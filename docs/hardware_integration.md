# Module 11: Real-Hardware Integration, Network Testbeds & 6G-Oriented Validation

## 1. Overview & Research Boundaries

Module 11 provides a hardware-agnostic integration, real network transport, telemetry probing, and validation layer for the **Predictive VRAM & Network-Aware Dynamic Split Inference for Edge LLMs** system.

### Core Objectives
1. **CPU-Only Execution**: Maintain 100% deterministic, offline functionality on development environments without discrete accelerators.
2. **Real CUDA Hardware Telemetry**: Interrogate physical GPU allocation, caching allocator reserves, and free headroom when CUDA is present.
3. **Real Network Probing**: Actively sample transport round-trip time (RTT), jitter, packet loss, and goodput over physical sockets.
4. **Safe Framed Transport**: Implement an authenticated, zero-pickle TCP backend for physical tensor and module parameter migration across network endpoints.
5. **Configurable Operating Profiles**: Characterize 10 distinct wireless/wired operating conditions and exploratory 6G operating scenarios.
6. **Kernel Network Shaping Adapter**: Provide an optional, platform-guarded Linux traffic control (`tc`/`netem`) interface with user opt-in and safety protections.

### Critical Research Boundaries & Invariants
- **6G Emulation vs Physical Radio**: Configured 6G profiles (`six_g_sub_thz_extreme_bandwidth`, `six_g_high_mobility_doppler`) emulate prospective high-throughput and sub-millisecond conditions for algorithm exploration; they **do not establish or claim performance on physical 6G radio hardware**.
- **No VRAM Substitution**: Host RAM is **never** substituted for GPU VRAM. When CUDA is absent, VRAM metrics are tagged `DataSource.UNAVAILABLE` with `value=None`.
- **No Zero Fabrication**: Unsupported GPU compute utilization is tagged `DataSource.UNAVAILABLE`, **never** reported as `0.0`.
- **One-Way Latency Invariant**: One-way latency is explicitly marked `DataSource.UNAVAILABLE` unless hardware clock synchronization (PTP IEEE 1588 or GPS-disciplined clocks) is verified.
- **Unknown Capacity Invariant**: Unknown device capacity is **never** interpreted as unlimited capacity.

---

## 2. Architecture & Component Placement

Module 11 serves as the physical substrate interface beneath the closed-loop runtime:

```
+---------------------------------------------------------------------------------------+
|  Module 9: Closed-Loop Orchestrator (ClosedLoopRuntime & ControlScheduler)            |
+---------------------------------------------------------------------------------------+
     │                                                                   │
     ▼                                                                   ▼
+----------------------------------+          +-----------------------------------------+
|  Module 7 & 8: Migration Engine  |          |  Modules 2, 3 & 4: Telemetry & Forecaster|
+-----------------+----------------+          +--------------------+--------------------+
                  │                                                │
                  ▼                                                ▼
+---------------------------------------------------------------------------------------+
|  MODULE 11: HARDWARE INTEGRATION & NETWORK VALIDATION LAYER (src/hardware/)           |
|                                                                                       |
|  [Hardware Telemetry Provider]              [Device Capability Registry]              |
|  - CPUHardwareProvider                      - TierCapabilities                        |
|  - CUDAHardwareProvider                     - Capacity & Dtype Verification           |
|  - MockCUDAHardwareProvider                                                           |
|                                                                                       |
|  [Network Measurement Engine]               [Framed TCP Transport Provider]           |
|  - SocketNetworkProbe                       - TCPNetworkTransferProvider              |
|  - MockNetworkProbe                         - TransferServer (HMAC + SHA-256)         |
|                                                                                       |
|  [Network Profile Catalog]                  [Linux Netem Adapter (Optional)]          |
|  - 10 Edge Profiles + 6G Exploratory        - Platform-guarded dry-run & cleanup      |
+---------------------------------------------------------------------------------------+
```

---

## 3. Provider Interfaces & Hardware Discovery

### Abstract Contract: `HardwareTelemetryProvider`
Located in `src/hardware/provider.py`:
- `discover_devices() -> List[DeviceDescriptor]`
- `get_memory_telemetry(device_id) -> HardwareMemoryTelemetry`
- `get_compute_telemetry(device_id) -> HardwareComputeTelemetry`
- `get_telemetry_capabilities(device_id) -> TelemetryCapabilities`
- `collect_snapshot(device_id) -> HardwareSnapshot`

### Memory Categories
The provider explicitly distinguishes non-interchangeable memory quantities:
1. **Total Device Memory**: Total physical capacity reported by driver.
2. **Allocated VRAM**: Bytes currently occupied by active PyTorch tensors.
3. **Reserved VRAM**: Bytes reserved by the PyTorch caching allocator from the OS/driver.
4. **Free VRAM**: Remaining unreserved headroom (`mem_get_info`).
5. **Host RAM Total / Used / Available**: System physical RAM sampled via `psutil`.

```python
from src.hardware.factory import HardwareProviderFactory

provider = HardwareProviderFactory.create_provider("auto")
mem = provider.get_memory_telemetry()
print(f"Total VRAM: {mem.total_vram_bytes.value} ({mem.total_vram_bytes.source})")
print(f"Host RAM  : {mem.host_ram_total_bytes.value} ({mem.host_ram_total_bytes.source})")
```

---

## 4. Device Capability Registry

Located in `src/hardware/capabilities.py`, `DeviceCapabilityRegistry` maintains validated specifications for each execution tier (`TierId`):

- **Observed vs Operator-Configured**: Capabilities derived directly from hardware carry `is_observed=True` and `DataSource.MEASURED`. Configured estimates carry `DataSource.ESTIMATED`.
- **Capacity Bounds Checking**: `validate_capacity(tier_id, required_bytes)` rejects unknown capacities and enforces memory safety margins.

```python
from src.hardware.capabilities import DeviceCapabilityRegistry
from src.runtime.tier import TierId

registry = DeviceCapabilityRegistry()
registry.discover_tier(TierId.USER_DEVICE, provider)
ok, reason = registry.validate_capacity(TierId.USER_DEVICE, 512 * 1024 * 1024)
```

---

## 5. Real Network Measurement & Probing

Located in `src/hardware/network_measurement.py`:
- **`SocketNetworkProbe`**: Measures real TCP connection round-trip times and effective throughput across physical sockets without blocking the inference loop.
- **`MockNetworkProbe`**: Injects deterministic RTT, jitter, and packet loss for unit testing without network sockets.
- **Transport Limitations**: Throughput measurements represent application goodput subject to socket buffer sizing, TCP slow start, and congestion control dynamics.

---

## 6. Authenticated Framed TCP Transport

Located in `src/hardware/transport.py`, `TCPNetworkTransferProvider` implements `MigrationTransferProvider` to enable physical tensor migration across network boundaries.

### Binary Framing Specification (84-Byte Fixed Header)
```
+---------------+---------------+---------------+-------------------------------+
|  Magic (4B)   |  Ver (2B: u16)| MsgType(2B:u16|     HMAC-SHA256 (32 Bytes)    |
|   b"PVTM"     |       1       | 1..6          |                               |
+---------------+---------------+---------------+-------------------------------+
| Meta Len (4B) |        Payload Length (8B: uint64)                            |
+---------------+---------------------------------------------------------------+
|                      Payload SHA-256 Checksum (32 Bytes)                      |
+-------------------------------------------------------------------------------+
|                     JSON Metadata (Meta Len UTF-8 Bytes)                      |
+-------------------------------------------------------------------------------+
|                     Raw Binary Tensor Buffer (Payload Len)                    |
+-------------------------------------------------------------------------------+
```

### Security Properties
1. **Zero-Pickle Serialization**: Deserialization parses tensor dimensions and dtypes from JSON and reconstructs tensors from contiguous raw binary bytes (`np.frombuffer`). Arbitrary code execution via pickle is completely prevented.
2. **HMAC-SHA256 Authentication**: Connections must authenticate using a shared secret session token. Unauthorized peers are immediately rejected.
3. **Payload Integrity Verification**: Payloads are hashed with SHA-256 and matched against header checksums prior to reconstruction.
4. **Length Validation**: Payloads exceeding `max_payload_bytes` are rejected before reading stream bytes.
5. **Default Loopback Binding**: `TransferServer` binds to `127.0.0.1` by default; binding to public interfaces requires explicit operator configuration.

---

## 7. Network Profile Catalog

Located in `src/hardware/network_profiles.py`, `NetworkProfileCatalog` standardizes operating environments:

| Profile Identifier | Bandwidth (Mbps) | RTT (ms) | Jitter (ms) | Packet Loss | Provenance | Research Use Case |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `ideal_lan` | 1000.0 | 1.0 | 0.1 | 0.000 | Illustrative | On-premise cluster baseline |
| `stable_wifi` | 150.0 | 15.0 | 2.0 | 0.001 | Illustrative | Office / Home Wi-Fi 6 |
| `congested_wifi` | 25.0 | 60.0 | 18.0 | 0.030 | Illustrative | Multi-user channel contention |
| `mobile_high_bandwidth`| 300.0 | 25.0 | 5.0 | 0.005 | Illustrative | 5G Sub-6GHz eMBB line-of-sight |
| `mobile_low_bandwidth` | 15.0 | 90.0 | 25.0 | 0.020 | Illustrative | Cell-edge LTE coverage |
| `high_rtt_backhaul` | 80.0 | 180.0 | 12.0 | 0.008 | Illustrative | Satellite / WAN backhaul |
| `intermittent_connectivity`| 50.0 | 40.0 | 30.0 | 0.050 | Illustrative | Periodic beam blockage / outages |
| `bandwidth_degradation`| 120 -> 5 | 20 -> 120 | 10.0 | 0.010 | Illustrative | Proactive migration trigger stress |
| `bandwidth_recovery` | 8 -> 100 | 110 -> 15 | 8.0 | 0.005 | Illustrative | Opportunistic edge offload |
| `oscillating_link` | 10 <-> 80 | 15 <-> 75 | 12.0 | 0.015 | Illustrative | Controller anti-thrashing test |
| `six_g_sub_thz_extreme_bandwidth`| 5000.0 | 0.8 | 0.05 | 0.000 | Illustrative | 6G Sub-THz exploratory study |
| `six_g_high_mobility_doppler` | 1200.0 | 3.0 | 4.0 | 0.008 | Illustrative | 6G High-mobility beam tracking |

---

## 8. Optional Linux Network Emulation (`tc` / `netem`)

Located in `src/hardware/linux_netem.py`:
- **Platform Guarding**: Detects OS platform (`platform.system() == "Linux"`), utility availability (`shutil.which("tc")`), and privileges.
- **Opt-In Requirement**: Kernel modifications require `opt_in=True` and `dry_run=False`. Ordinary test runs execute safely in dry-run mode.
- **Primary Interface Protection**: Refuses by default to shape primary interfaces (`eth0`, `wlan0`).
- **Restoration Guidance**: Emits exact shell instructions (`sudo tc qdisc del dev <interface> root`) to restore clean interfaces.

---

## 9. VRAM & Memory Predictability Validation

Located in `src/hardware/memory_validation.py`:
- Evaluates `KVCacheAwareMemoryPredictor` against real or simulated step-by-step memory growth.
- Quantifies KV-cache memory expansion per token step.
- Measures sampling duration overhead (<5.0 ms target).
- Computes warning lead time before memory risk thresholds (e.g., 85% of capacity) are breached.

---

## 10. CLI Utilities & Benchmarks

1. **Hardware Inspection**:
   ```bash
   python scripts/inspect_hardware.py --provider auto
   ```
2. **Network Probing Benchmark**:
   ```bash
   python scripts/benchmark_network.py --host 127.0.0.1 --port 8000 --trials 5 --mock
   ```
3. **Real TCP Transfer Benchmark**:
   ```bash
   python scripts/benchmark_real_transfer.py --tensor-dim 512 --trials 5
   ```
4. **End-to-End Evaluation Harness**:
   ```bash
   python scripts/evaluate_hardware_integration.py --provider auto --steps 16
   ```

---

## 11. Test Suite Execution

Run hardware-independent unit tests (100% offline, zero GPU required):
```bash
uv run pytest tests/test_hardware.py -v
```

Run optional physical integration tests (requires CUDA GPU / physical sockets):
```bash
uv run pytest tests/test_hardware_integration.py -v
```
If physical CUDA hardware is absent, physical tests skip cleanly with explicit reasons.
