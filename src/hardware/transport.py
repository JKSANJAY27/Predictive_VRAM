"""
Safe Authenticated Framed TCP Transport for Module 11.

Implements MigrationTransferProvider with:
- Explicit framed binary wire protocol (magic bytes, version, header checksums)
- HMAC-SHA256 authentication (rejects unauthorized connections)
- SHA-256 payload integrity verification (rejects corrupted transmissions)
- Pure binary tensor serialization (NEVER uses untrusted pickle)
- Strict payload length validation
- Connection and transfer timeouts
- Cancellation / abort support
- Localhost default binding (never binds to 0.0.0.0 without explicit opt-in)
"""

from __future__ import annotations

import hashlib
import hmac
import json
import socket
import struct
import threading
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.migration.transfer import LocalTensorTransfer, MigrationTransferProvider
from src.runtime.tier import TierId
from src.telemetry.types import DataSource


# ---------------------------------------------------------------------------
# Protocol Constants
# ---------------------------------------------------------------------------

MAGIC_BYTES = b"PVTM"  # Predictive VRAM Tensor Message
PROTOCOL_VERSION = 1

HEADER_STRUCT = struct.Struct("!4sHH32sIQ32s")
HEADER_SIZE = HEADER_STRUCT.size  # 84 bytes


class MessageType(IntEnum):
    PING = 1
    PONG = 2
    TENSOR = 3
    MODULE = 4
    ACK = 5
    ERROR = 6


class TransferError(Exception):
    """Base error for TCP transfer failures."""
    pass


class AuthenticationError(TransferError):
    """Raised when HMAC authentication fails."""
    pass


class IntegrityError(TransferError):
    """Raised when SHA-256 checksum verification fails."""
    pass


class PayloadTooLargeError(TransferError):
    """Raised when incoming payload exceeds configured maximum size."""
    pass


# ---------------------------------------------------------------------------
# Safe Tensor Serialization (Zero Pickle)
# ---------------------------------------------------------------------------

TORCH_DTYPE_MAP = {
    "float32": (torch.float32, np.float32),
    "float64": (torch.float64, np.float64),
    "float16": (torch.float16, np.float16),
    "int32": (torch.int32, np.int32),
    "int64": (torch.int64, np.int64),
    "int16": (torch.int16, np.int16),
    "int8": (torch.int8, np.int8),
    "uint8": (torch.uint8, np.uint8),
    "bool": (torch.bool, np.bool_),
}


def serialize_tensor_safe(tensor: torch.Tensor) -> Tuple[Dict[str, Any], bytes]:
    """
    Serialize a PyTorch tensor into structured JSON metadata and raw byte buffer.
    Completely avoids pickle.
    """
    t_cpu = tensor.detach().cpu().contiguous()
    dtype_str = str(t_cpu.dtype).replace("torch.", "")
    if dtype_str not in TORCH_DTYPE_MAP:
        raise ValueError(f"Unsupported tensor dtype for safe serialization: {dtype_str}")

    _, np_dtype = TORCH_DTYPE_MAP[dtype_str]
    np_arr = t_cpu.numpy().astype(np_dtype, copy=False)
    raw_bytes = np_arr.tobytes()

    metadata = {
        "shape": list(t_cpu.shape),
        "dtype": dtype_str,
        "numel": t_cpu.numel(),
    }
    return metadata, raw_bytes


def deserialize_tensor_safe(
    metadata: Dict[str, Any], raw_bytes: bytes, target_device: str = "cpu"
) -> torch.Tensor:
    """
    Safely reconstruct a PyTorch tensor from metadata and raw byte buffer.
    """
    shape = tuple(metadata["shape"])
    dtype_str = metadata["dtype"]
    if dtype_str not in TORCH_DTYPE_MAP:
        raise ValueError(f"Unsupported tensor dtype: {dtype_str}")

    torch_dtype, np_dtype = TORCH_DTYPE_MAP[dtype_str]
    np_arr = np.frombuffer(raw_bytes, dtype=np_dtype).reshape(shape).copy()
    tensor = torch.from_numpy(np_arr).to(dtype=torch_dtype)
    if target_device != "cpu":
        tensor = tensor.to(target_device)
    return tensor


def serialize_module_safe(module: nn.Module) -> Tuple[Dict[str, Any], bytes]:
    """
    Safely serialize a Module's state_dict into concatenated byte buffer and metadata.
    """
    state_dict = module.state_dict()
    params_meta = []
    buffer_parts = []
    current_offset = 0

    for name, tensor in state_dict.items():
        meta, raw = serialize_tensor_safe(tensor)
        byte_len = len(raw)
        params_meta.append({
            "name": name,
            "shape": meta["shape"],
            "dtype": meta["dtype"],
            "offset": current_offset,
            "length": byte_len,
        })
        buffer_parts.append(raw)
        current_offset += byte_len

    combined_bytes = b"".join(buffer_parts)
    metadata = {
        "params": params_meta,
        "total_bytes": current_offset,
    }
    return metadata, combined_bytes


def deserialize_module_safe(
    module: nn.Module,
    metadata: Dict[str, Any],
    raw_bytes: bytes,
    target_device: str = "cpu",
) -> nn.Module:
    """
    Safely load weights into an existing Module instance.
    """
    new_state = {}
    for p_meta in metadata["params"]:
        name = p_meta["name"]
        offset = p_meta["offset"]
        length = p_meta["length"]
        part = raw_bytes[offset : offset + length]
        tensor = deserialize_tensor_safe(p_meta, part, target_device=target_device)
        new_state[name] = tensor

    module.load_state_dict(new_state)
    module.to(target_device)
    return module


# ---------------------------------------------------------------------------
# Framing and Wire Protocol Helpers
# ---------------------------------------------------------------------------

def compute_auth_hmac(secret_key: bytes, msg_type: int, meta_len: int, payload_len: int, checksum: bytes) -> bytes:
    h = hmac.new(secret_key, digestmod=hashlib.sha256)
    h.update(struct.pack("!H I Q", msg_type, meta_len, payload_len))
    h.update(checksum)
    return h.digest()


def pack_frame(
    msg_type: MessageType,
    metadata: Dict[str, Any],
    payload: bytes,
    auth_key: bytes,
) -> bytes:
    meta_json = json.dumps(metadata).encode("utf-8")
    meta_len = len(meta_json)
    payload_len = len(payload)
    checksum = hashlib.sha256(payload).digest()
    auth_hmac = compute_auth_hmac(auth_key, int(msg_type), meta_len, payload_len, checksum)

    header = HEADER_STRUCT.pack(
        MAGIC_BYTES,
        PROTOCOL_VERSION,
        int(msg_type),
        auth_hmac,
        meta_len,
        payload_len,
        checksum,
    )
    return header + meta_json + payload


def read_exact(sock: socket.socket, num_bytes: int, cancel_event: Optional[threading.Event] = None) -> bytes:
    buf = bytearray(num_bytes)
    view = memoryview(buf)
    pos = 0
    while pos < num_bytes:
        if cancel_event and cancel_event.is_set():
            raise TransferError("Transfer aborted by cancellation request.")
        try:
            chunk = sock.recv_into(view[pos:], num_bytes - pos)
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError) as e:
            raise TransferError(f"Connection reset/aborted by remote peer: {e}") from e

        if chunk == 0:
            raise TransferError(f"Connection closed prematurely (received {pos}/{num_bytes} bytes).")
        pos += chunk
    return bytes(buf)


# ---------------------------------------------------------------------------
# Transfer Server
# ---------------------------------------------------------------------------

class TransferServer:
    """
    Lightweight, authenticated TCP receiver server for inter-tier state transfers.

    Runs in a dedicated thread, safely validating frames and verifying SHA-256 checksums.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,  # 0 binds to ephemeral free port
        auth_key: bytes = b"default_predictive_vram_secret",
        max_payload_bytes: int = 512 * 1024 * 1024,  # 512 MB
        timeout_seconds: float = 10.0,
    ) -> None:
        self.host = host
        self.requested_port = port
        self.auth_key = auth_key
        self.max_payload_bytes = max_payload_bytes
        self.timeout_seconds = timeout_seconds

        self._server_sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self.bound_port: int = 0

        self.received_tensors: List[Tuple[Dict[str, Any], torch.Tensor]] = []
        self.received_frames_count: int = 0
        self.total_bytes_received: int = 0

    def start(self) -> int:
        """Start listening server thread. Returns assigned port."""
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind((self.host, self.requested_port))
        self.bound_port = self._server_sock.getsockname()[1]
        self._server_sock.listen(5)
        self._server_sock.settimeout(0.5)

        self._running = True
        self._thread = threading.Thread(target=self._serve_loop, daemon=True)
        self._thread.start()
        return self.bound_port

    def stop(self) -> None:
        """Stop listening and close socket."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._server_sock:
            try:
                self._server_sock.close()
            except Exception:
                pass

    def _serve_loop(self) -> None:
        while self._running:
            try:
                conn, _ = self._server_sock.accept()
            except socket.timeout:
                continue
            except Exception:
                break

            conn.settimeout(self.timeout_seconds)
            try:
                self._handle_client(conn)
            except Exception as e:
                err_type_name = type(e).__name__
                try:
                    err_frame = pack_frame(
                        MessageType.ERROR,
                        {"error": str(e), "error_type": err_type_name},
                        b"",
                        self.auth_key,
                    )
                    conn.sendall(err_frame)
                except Exception:
                    pass
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

    def _handle_client(self, conn: socket.socket) -> None:
        # Read header
        raw_header = read_exact(conn, HEADER_SIZE)
        magic, version, msg_type, rcv_hmac, meta_len, payload_len, checksum = HEADER_STRUCT.unpack(raw_header)

        if magic != MAGIC_BYTES:
            raise TransferError(f"Invalid magic bytes: {magic}")
        if version != PROTOCOL_VERSION:
            raise TransferError(f"Unsupported protocol version: {version}")

        # Check payload length
        if payload_len > self.max_payload_bytes:
            raise PayloadTooLargeError(
                f"Payload size {payload_len} exceeds max allowed {self.max_payload_bytes}"
            )

        # Verify HMAC Authentication
        expected_hmac = compute_auth_hmac(self.auth_key, msg_type, meta_len, payload_len, checksum)
        if not hmac.compare_digest(rcv_hmac, expected_hmac):
            raise AuthenticationError("HMAC signature verification failed. Unauthorized peer.")

        # Read metadata and payload
        meta_bytes = read_exact(conn, meta_len)
        metadata = json.loads(meta_bytes.decode("utf-8"))
        payload_bytes = read_exact(conn, payload_len)

        # Verify SHA-256 integrity
        actual_checksum = hashlib.sha256(payload_bytes).digest()
        if not hmac.compare_digest(checksum, actual_checksum):
            raise IntegrityError("Payload SHA-256 checksum mismatch! Data corrupted.")

        # Dispatch
        if msg_type == MessageType.TENSOR:
            t = deserialize_tensor_safe(metadata, payload_bytes, target_device="cpu")
            self.received_tensors.append((metadata, t))
        elif msg_type == MessageType.PING:
            pass

        self.received_frames_count += 1
        self.total_bytes_received += HEADER_SIZE + meta_len + payload_len

        # Send ACK frame
        ack_frame = pack_frame(
            MessageType.ACK,
            {"status": "ok", "bytes_received": payload_len},
            b"",
            self.auth_key,
        )
        conn.sendall(ack_frame)


# ---------------------------------------------------------------------------
# Real TCP Migration Transfer Provider
# ---------------------------------------------------------------------------

class TCPNetworkTransferProvider(MigrationTransferProvider):
    """
    Real TCP-backed transport provider implementing MigrationTransferProvider.

    Connects to remote tier server, authenticates via HMAC, verifies SHA-256 checksums,
    and returns exact byte counts and durations tagged DataSource.MEASURED.
    """

    def __init__(
        self,
        endpoint_host: str = "127.0.0.1",
        endpoint_port: int = 8000,
        auth_key: bytes = b"default_predictive_vram_secret",
        timeout_seconds: float = 10.0,
        max_payload_bytes: int = 512 * 1024 * 1024,
    ) -> None:
        self.host = endpoint_host
        self.port = endpoint_port
        self.auth_key = auth_key
        self.timeout = timeout_seconds
        self.max_payload = max_payload_bytes
        self._local_fallback = LocalTensorTransfer()

    def set_endpoint(self, host: str, port: int) -> None:
        """Update destination endpoint host and port."""
        self.host = host
        self.port = port

    def transfer_tensor(
        self,
        tensor: torch.Tensor,
        source_tier: TierId,
        destination_tier: TierId,
        target_device: str = "cpu",
        cancel_event: Optional[threading.Event] = None,
    ) -> Tuple[torch.Tensor, int, float, DataSource]:
        """
        Transfer tensor over authenticated framed TCP connection.
        """
        if cancel_event and cancel_event.is_set():
            raise TransferError("Transfer cancelled prior to transmission.")

        start = time.perf_counter()

        meta, payload = serialize_tensor_safe(tensor)
        meta["source_tier"] = source_tier.value
        meta["destination_tier"] = destination_tier.value
        meta["target_device"] = target_device

        if len(payload) > self.max_payload:
            raise PayloadTooLargeError(f"Payload size {len(payload)} exceeds max limit {self.max_payload}")

        frame = pack_frame(MessageType.TENSOR, meta, payload, self.auth_key)
        total_wire_bytes = len(frame)

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        try:
            try:
                sock.connect((self.host, self.port))
            except (ConnectionRefusedError, socket.gaierror, socket.timeout) as e:
                raise TransferError(f"Failed to connect to endpoint {self.host}:{self.port}: {e}") from e

            if cancel_event and cancel_event.is_set():
                raise TransferError("Transfer cancelled prior to transmission.")

            sock.sendall(frame)

            # Wait for ACK or ERROR
            ack_header = read_exact(sock, HEADER_SIZE, cancel_event=cancel_event)
            _, _, ack_type, _, ack_meta_len, _, _ = HEADER_STRUCT.unpack(ack_header)
            if ack_type == MessageType.ERROR:
                err_bytes = read_exact(sock, ack_meta_len, cancel_event=cancel_event)
                try:
                    err_info = json.loads(err_bytes.decode("utf-8"))
                    err_msg = err_info.get("error", "Unknown error")
                    err_type = err_info.get("error_type", "")
                except Exception:
                    err_msg = err_bytes.decode("utf-8")
                    err_type = ""

                if err_type == "AuthenticationError" or "HMAC" in err_msg:
                    raise AuthenticationError(err_msg)
                elif err_type == "PayloadTooLargeError" or "exceeds max" in err_msg:
                    raise PayloadTooLargeError(err_msg)
                elif err_type == "IntegrityError":
                    raise IntegrityError(err_msg)
                raise TransferError(f"Remote server reported transfer error: {err_msg}")

            if ack_type != MessageType.ACK:
                raise TransferError(f"Unexpected response frame type: {ack_type}")

        finally:
            try:
                sock.close()
            except Exception:
                pass

        duration_ms = (time.perf_counter() - start) * 1000.0
        moved = tensor.clone().to(target_device)
        return moved, total_wire_bytes, duration_ms, DataSource.MEASURED

    def transfer_module(
        self,
        module: nn.Module,
        source_tier: TierId,
        destination_tier: TierId,
        target_device: str = "cpu",
        cancel_event: Optional[threading.Event] = None,
    ) -> Tuple[nn.Module, int, float, DataSource]:
        """
        Transfer entire module state over authenticated framed TCP connection.
        """
        if cancel_event and cancel_event.is_set():
            raise TransferError("Transfer cancelled prior to transmission.")

        start = time.perf_counter()

        meta, payload = serialize_module_safe(module)
        meta["source_tier"] = source_tier.value
        meta["destination_tier"] = destination_tier.value
        meta["target_device"] = target_device

        if len(payload) > self.max_payload:
            raise PayloadTooLargeError(f"Payload size {len(payload)} exceeds max limit {self.max_payload}")

        frame = pack_frame(MessageType.MODULE, meta, payload, self.auth_key)
        total_wire_bytes = len(frame)

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        try:
            try:
                sock.connect((self.host, self.port))
            except (ConnectionRefusedError, socket.gaierror, socket.timeout) as e:
                raise TransferError(f"Failed to connect to endpoint {self.host}:{self.port}: {e}") from e

            if cancel_event and cancel_event.is_set():
                raise TransferError("Transfer cancelled prior to transmission.")

            sock.sendall(frame)

            ack_header = read_exact(sock, HEADER_SIZE, cancel_event=cancel_event)
            _, _, ack_type, _, ack_meta_len, _, _ = HEADER_STRUCT.unpack(ack_header)
            if ack_type == MessageType.ERROR:
                err_bytes = read_exact(sock, ack_meta_len, cancel_event=cancel_event)
                try:
                    err_info = json.loads(err_bytes.decode("utf-8"))
                    err_msg = err_info.get("error", "Unknown error")
                    err_type = err_info.get("error_type", "")
                except Exception:
                    err_msg = err_bytes.decode("utf-8")
                    err_type = ""

                if err_type == "AuthenticationError" or "HMAC" in err_msg:
                    raise AuthenticationError(err_msg)
                elif err_type == "PayloadTooLargeError" or "exceeds max" in err_msg:
                    raise PayloadTooLargeError(err_msg)
                elif err_type == "IntegrityError":
                    raise IntegrityError(err_msg)
                raise TransferError(f"Remote server error: {err_msg}")

            if ack_type != MessageType.ACK:
                raise TransferError(f"Unexpected response frame type: {ack_type}")
        finally:
            try:
                sock.close()
            except Exception:
                pass

        duration_ms = (time.perf_counter() - start) * 1000.0
        module.to(target_device)
        return module, total_wire_bytes, duration_ms, DataSource.MEASURED
