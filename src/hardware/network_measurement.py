"""
Real Network Measurement Interface and Probing for Module 11.

Measures RTT, effective throughput, jitter, packet loss, and connection state.
Preserves strict data provenance (DataSource.MEASURED vs EMULATED vs UNAVAILABLE).
Distinguishes round-trip time from one-way latency, marking one-way latency
unavailable unless clock synchronization is present.
"""

from __future__ import annotations

import math
import socket
import statistics
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.telemetry.types import DataSource, NetworkConditionSnapshot, TaggedValue


@dataclass(frozen=True)
class NetworkMeasurementResult:
    """
    Structured outcome of an active network measurement probe.

    Research Boundaries & Invariants:
    - one_way_latency_ms is marked UNAVAILABLE unless hardware clock synchronization
      (PTP IEEE 1588 / synchronized GPS clocks) is explicitly verified.
    - Throughput measurements reflect effective transport goodput, not radio-layer link capacity.
    """
    rtt_ms: TaggedValue
    one_way_latency_ms: TaggedValue
    throughput_mbps: TaggedValue
    jitter_ms: TaggedValue
    packet_loss: TaggedValue

    duration_seconds: float
    sample_count: int
    timestamp: float = field(default_factory=time.monotonic)
    connection_state: str = "unknown"  # 'connected', 'refused', 'timeout', 'unreachable'
    source_host: str = "127.0.0.1"
    target_host: str = "127.0.0.1"
    target_port: int = 0
    measurement_method: str = "direct_probe"
    error_message: Optional[str] = None
    limitations: Tuple[str, ...] = (
        "TCP-based throughput reflects application-layer goodput influenced by congestion control and buffer sizing.",
        "One-way latency is unavailable without PTP or synchronized sub-millisecond clocks.",
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rtt_ms": self.rtt_ms.to_dict(),
            "one_way_latency_ms": self.one_way_latency_ms.to_dict(),
            "throughput_mbps": self.throughput_mbps.to_dict(),
            "jitter_ms": self.jitter_ms.to_dict(),
            "packet_loss": self.packet_loss.to_dict(),
            "duration_seconds": self.duration_seconds,
            "sample_count": self.sample_count,
            "timestamp": self.timestamp,
            "connection_state": self.connection_state,
            "source_host": self.source_host,
            "target_host": self.target_host,
            "target_port": self.target_port,
            "measurement_method": self.measurement_method,
            "error_message": self.error_message,
            "limitations": list(self.limitations),
        }

    def to_network_snapshot(self, scenario_label: str = "measured_link") -> NetworkConditionSnapshot:
        """
        Convert to Module 2/3 NetworkConditionSnapshot with preserved DataSource.MEASURED provenance.
        """
        return NetworkConditionSnapshot(
            rtt_ms=self.rtt_ms,
            bandwidth_mbps=self.throughput_mbps,
            packet_loss_rate=self.packet_loss,
            jitter_ms=self.jitter_ms,
            scenario_label=scenario_label,
            timestamp=self.timestamp,
        )


class NetworkProbe(ABC):
    """Abstract interface for active network measurement probes."""

    @abstractmethod
    def measure_rtt(
        self,
        host: str,
        port: int,
        count: int = 3,
        timeout_s: float = 1.0,
    ) -> NetworkMeasurementResult:
        """Measure round-trip time and jitter across multiple connection/ping samples."""
        pass

    @abstractmethod
    def measure_throughput(
        self,
        host: str,
        port: int,
        payload_bytes: int = 65536,
        timeout_s: float = 2.0,
    ) -> NetworkMeasurementResult:
        """Measure effective transmission throughput."""
        pass

    @abstractmethod
    def measure_link(
        self,
        host: str,
        port: int,
        rtt_count: int = 3,
        throughput_bytes: int = 65536,
        timeout_s: float = 2.0,
    ) -> NetworkMeasurementResult:
        """Measure composite link metrics (RTT, jitter, throughput, packet loss)."""
        pass


class SocketNetworkProbe(NetworkProbe):
    """
    Real socket-based network measurement probe.

    Uses TCP SYN/ACK handshake round-trip timing for non-intrusive RTT measurement,
    and framed payload transmission for throughput benchmarking.
    """

    def measure_rtt(
        self,
        host: str,
        port: int,
        count: int = 3,
        timeout_s: float = 1.0,
    ) -> NetworkMeasurementResult:
        start_time = time.perf_counter()
        rtts: List[float] = []
        failures = 0
        last_error: Optional[str] = None
        conn_state = "connected"

        for _ in range(max(1, count)):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(timeout_s)
            t0 = time.perf_counter()
            try:
                s.connect((host, port))
                t1 = time.perf_counter()
                rtts.append((t1 - t0) * 1000.0)
                s.close()
            except socket.timeout:
                failures += 1
                last_error = "Connection timed out"
                conn_state = "timeout"
            except ConnectionRefusedError:
                failures += 1
                last_error = "Connection refused"
                conn_state = "refused"
            except Exception as e:
                failures += 1
                last_error = str(e)
                conn_state = "unreachable"
            finally:
                try:
                    s.close()
                except Exception:
                    pass

        total_dur = time.perf_counter() - start_time
        loss_rate = failures / count

        if rtts:
            mean_rtt = float(statistics.mean(rtts))
            jitter = float(statistics.stdev(rtts)) if len(rtts) > 1 else 0.0
            rtt_rec = TaggedValue.measured(round(mean_rtt, 3), "ms")
            jitter_rec = TaggedValue.measured(round(jitter, 3), "ms")
        else:
            rtt_rec = TaggedValue.unavailable("ms")
            jitter_rec = TaggedValue.unavailable("ms")

        loss_rec = TaggedValue.measured(round(loss_rate, 4), "ratio")
        tp_rec = TaggedValue.unavailable("Mbps")
        owl_rec = TaggedValue.unavailable("ms")  # One-way latency unavailable without clock sync

        return NetworkMeasurementResult(
            rtt_ms=rtt_rec,
            one_way_latency_ms=owl_rec,
            throughput_mbps=tp_rec,
            jitter_ms=jitter_rec,
            packet_loss=loss_rec,
            duration_seconds=total_dur,
            sample_count=count,
            timestamp=time.monotonic(),
            connection_state=conn_state,
            target_host=host,
            target_port=port,
            measurement_method="tcp_syn_ack_probe",
            error_message=last_error,
        )

    def measure_throughput(
        self,
        host: str,
        port: int,
        payload_bytes: int = 65536,
        timeout_s: float = 2.0,
    ) -> NetworkMeasurementResult:
        start_time = time.perf_counter()
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout_s)

        conn_state = "connected"
        last_error: Optional[str] = None
        throughput_mbps: Optional[float] = None

        try:
            s.connect((host, port))
            # Send payload probe
            data = b"X" * payload_bytes
            t_tx0 = time.perf_counter()
            s.sendall(data)
            # Receive 4-byte ack or echo
            ack = s.recv(4)
            t_tx1 = time.perf_counter()
            tx_dur = max(1e-6, t_tx1 - t_tx0)
            bits = payload_bytes * 8
            throughput_mbps = (bits / tx_dur) / 1e6
        except socket.timeout:
            conn_state = "timeout"
            last_error = "Throughput probe timed out"
        except ConnectionRefusedError:
            conn_state = "refused"
            last_error = "Connection refused"
        except Exception as e:
            conn_state = "failed"
            last_error = str(e)
        finally:
            try:
                s.close()
            except Exception:
                pass

        total_dur = time.perf_counter() - start_time
        if throughput_mbps is not None:
            tp_rec = TaggedValue.measured(round(throughput_mbps, 3), "Mbps")
            loss_rec = TaggedValue.measured(0.0, "ratio")
        else:
            tp_rec = TaggedValue.unavailable("Mbps")
            loss_rec = TaggedValue.measured(1.0, "ratio")

        return NetworkMeasurementResult(
            rtt_ms=TaggedValue.unavailable("ms"),
            one_way_latency_ms=TaggedValue.unavailable("ms"),
            throughput_mbps=tp_rec,
            jitter_ms=TaggedValue.unavailable("ms"),
            packet_loss=loss_rec,
            duration_seconds=total_dur,
            sample_count=1,
            timestamp=time.monotonic(),
            connection_state=conn_state,
            target_host=host,
            target_port=port,
            measurement_method="tcp_stream_transfer_probe",
            error_message=last_error,
        )

    def measure_link(
        self,
        host: str,
        port: int,
        rtt_count: int = 3,
        throughput_bytes: int = 65536,
        timeout_s: float = 2.0,
    ) -> NetworkMeasurementResult:
        rtt_res = self.measure_rtt(host, port, count=rtt_count, timeout_s=timeout_s)
        if rtt_res.connection_state != "connected":
            return rtt_res

        tp_res = self.measure_throughput(host, port, payload_bytes=throughput_bytes, timeout_s=timeout_s)
        return NetworkMeasurementResult(
            rtt_ms=rtt_res.rtt_ms,
            one_way_latency_ms=TaggedValue.unavailable("ms"),
            throughput_mbps=tp_res.throughput_mbps,
            jitter_ms=rtt_res.jitter_ms,
            packet_loss=rtt_res.packet_loss,
            duration_seconds=rtt_res.duration_seconds + tp_res.duration_seconds,
            sample_count=rtt_count + 1,
            timestamp=time.monotonic(),
            connection_state=tp_res.connection_state,
            target_host=host,
            target_port=port,
            measurement_method="composite_link_probe",
            error_message=tp_res.error_message or rtt_res.error_message,
        )


class MockNetworkProbe(NetworkProbe):
    """
    Configurable mock probe for testing without real network access.
    """

    def __init__(
        self,
        simulated_rtt_ms: float = 18.5,
        simulated_jitter_ms: float = 1.2,
        simulated_throughput_mbps: float = 120.0,
        simulated_loss_rate: float = 0.0,
        simulate_failure: bool = False,
        failure_type: str = "refused",
    ) -> None:
        self.rtt_ms = simulated_rtt_ms
        self.jitter_ms = simulated_jitter_ms
        self.throughput_mbps = simulated_throughput_mbps
        self.loss_rate = simulated_loss_rate
        self.simulate_failure = simulate_failure
        self.failure_type = failure_type

    def measure_rtt(
        self,
        host: str,
        port: int,
        count: int = 3,
        timeout_s: float = 1.0,
    ) -> NetworkMeasurementResult:
        if self.simulate_failure:
            return NetworkMeasurementResult(
                rtt_ms=TaggedValue.unavailable("ms"),
                one_way_latency_ms=TaggedValue.unavailable("ms"),
                throughput_mbps=TaggedValue.unavailable("Mbps"),
                jitter_ms=TaggedValue.unavailable("ms"),
                packet_loss=TaggedValue.measured(1.0, "ratio"),
                duration_seconds=0.01,
                sample_count=count,
                connection_state=self.failure_type,
                target_host=host,
                target_port=port,
                measurement_method="mock_probe",
                error_message=f"Simulated network error: {self.failure_type}",
            )

        return NetworkMeasurementResult(
            rtt_ms=TaggedValue.measured(self.rtt_ms, "ms"),
            one_way_latency_ms=TaggedValue.unavailable("ms"),
            throughput_mbps=TaggedValue.unavailable("Mbps"),
            jitter_ms=TaggedValue.measured(self.jitter_ms, "ms"),
            packet_loss=TaggedValue.measured(self.loss_rate, "ratio"),
            duration_seconds=0.005,
            sample_count=count,
            connection_state="connected",
            target_host=host,
            target_port=port,
            measurement_method="mock_probe",
            error_message=None,
        )

    def measure_throughput(
        self,
        host: str,
        port: int,
        payload_bytes: int = 65536,
        timeout_s: float = 2.0,
    ) -> NetworkMeasurementResult:
        if self.simulate_failure:
            return NetworkMeasurementResult(
                rtt_ms=TaggedValue.unavailable("ms"),
                one_way_latency_ms=TaggedValue.unavailable("ms"),
                throughput_mbps=TaggedValue.unavailable("Mbps"),
                jitter_ms=TaggedValue.unavailable("ms"),
                packet_loss=TaggedValue.measured(1.0, "ratio"),
                duration_seconds=0.01,
                sample_count=1,
                connection_state=self.failure_type,
                target_host=host,
                target_port=port,
                measurement_method="mock_probe",
                error_message=f"Simulated network error: {self.failure_type}",
            )

        return NetworkMeasurementResult(
            rtt_ms=TaggedValue.unavailable("ms"),
            one_way_latency_ms=TaggedValue.unavailable("ms"),
            throughput_mbps=TaggedValue.measured(self.throughput_mbps, "Mbps"),
            jitter_ms=TaggedValue.unavailable("ms"),
            packet_loss=TaggedValue.measured(self.loss_rate, "ratio"),
            duration_seconds=0.005,
            sample_count=1,
            connection_state="connected",
            target_host=host,
            target_port=port,
            measurement_method="mock_probe",
            error_message=None,
        )

    def measure_link(
        self,
        host: str,
        port: int,
        rtt_count: int = 3,
        throughput_bytes: int = 65536,
        timeout_s: float = 2.0,
    ) -> NetworkMeasurementResult:
        if self.simulate_failure:
            return self.measure_rtt(host, port, count=rtt_count, timeout_s=timeout_s)

        return NetworkMeasurementResult(
            rtt_ms=TaggedValue.measured(self.rtt_ms, "ms"),
            one_way_latency_ms=TaggedValue.unavailable("ms"),
            throughput_mbps=TaggedValue.measured(self.throughput_mbps, "Mbps"),
            jitter_ms=TaggedValue.measured(self.jitter_ms, "ms"),
            packet_loss=TaggedValue.measured(self.loss_rate, "ratio"),
            duration_seconds=0.01,
            sample_count=rtt_count + 1,
            connection_state="connected",
            target_host=host,
            target_port=port,
            measurement_method="mock_composite_probe",
            error_message=None,
        )
