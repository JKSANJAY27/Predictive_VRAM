"""
Optional Linux Network Emulation (tc / netem) Adapter for Module 11.

Allows controlled physical network shaping on Linux testbeds using tc/netem.
Strict safety constraints:
- Refuses to run on non-Linux platforms (returns diagnostic status).
- Requires explicit user opt-in (opt_in=True). Never runs in ordinary unit tests.
- Protected interface guard: protects primary network interfaces by default.
- Provides dry-run inspection and explicit restoration/cleanup instructions.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.hardware.network_profiles import NetworkProfile


@dataclass(frozen=True)
class NetemSupportStatus:
    """Diagnostic status of platform support for Linux tc/netem."""
    is_supported: bool
    os_platform: str
    is_linux: bool
    tc_available: bool
    has_net_admin: bool
    reasons: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_supported": self.is_supported,
            "os_platform": self.os_platform,
            "is_linux": self.is_linux,
            "tc_available": self.tc_available,
            "has_net_admin": self.has_net_admin,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class NetemCommandResult:
    """Result of a netem configuration command."""
    success: bool
    dry_run: bool
    applied_command: str
    interface: str
    profile_id: Optional[str]
    timestamp: float = field(default_factory=time.monotonic)
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "dry_run": self.dry_run,
            "applied_command": self.applied_command,
            "interface": self.interface,
            "profile_id": self.profile_id,
            "timestamp": self.timestamp,
            "error_message": self.error_message,
        }


class LinuxNetemAdapter:
    """
    Optional adapter for kernel-level network emulation using Linux Traffic Control (tc).

    Safety Invariant:
    - Default behavior is dry-run. Requires opt_in=True and dry_run=False to execute kernel commands.
    """

    PROTECTED_DEFAULT_INTERFACES = {"eth0", "ens3", "ens33", "enp0s3", "wlan0", "wifi0"}

    def __init__(
        self,
        interface: str = "lo",
        opt_in: bool = False,
        allow_primary_interface: bool = False,
    ) -> None:
        self.interface = interface
        self.opt_in = opt_in
        self.allow_primary_interface = allow_primary_interface
        self._last_applied_command: Optional[str] = None

    def check_support(self) -> NetemSupportStatus:
        """
        Verify operating system compatibility and required tooling.
        """
        os_name = platform.system()
        is_linux = (os_name == "Linux")
        tc_found = shutil.which("tc") is not None
        has_net_admin = False
        reasons: List[str] = []

        if not is_linux:
            reasons.append(f"Operating system '{os_name}' is not Linux. Linux tc/netem is unavailable.")
        if not tc_found:
            reasons.append("The 'tc' (traffic control) binary was not found in PATH.")

        if is_linux:
            # Check for root / CAP_NET_ADMIN privileges
            has_net_admin = (hasattr(os, "geteuid") and os.geteuid() == 0)
            if not has_net_admin:
                reasons.append("Process does not have root / CAP_NET_ADMIN privileges required for tc modification.")

        is_supported = is_linux and tc_found and has_net_admin
        if is_supported:
            reasons.append("Linux tc/netem environment verified and supported.")

        return NetemSupportStatus(
            is_supported=is_supported,
            os_platform=os_name,
            is_linux=is_linux,
            tc_available=tc_found,
            has_net_admin=has_net_admin,
            reasons=tuple(reasons),
        )

    def build_tc_command(self, profile: NetworkProfile) -> str:
        """
        Format the tc netem shell command for a given NetworkProfile.
        """
        one_way_delay = max(0.1, profile.rtt_ms / 2.0)
        jitter = max(0.0, profile.jitter_ms)
        loss_pct = max(0.0, min(100.0, profile.packet_loss_rate * 100.0))
        rate_mbit = max(0.1, profile.bandwidth_mbps)

        cmd = (
            f"tc qdisc replace dev {self.interface} root netem "
            f"delay {one_way_delay:.2f}ms {jitter:.2f}ms "
            f"loss {loss_pct:.2f}% "
            f"rate {rate_mbit:.1f}mbit"
        )
        return cmd

    def apply_profile(self, profile: NetworkProfile, dry_run: bool = True) -> NetemCommandResult:
        """
        Apply traffic shaping rules corresponding to the given profile.
        """
        cmd = self.build_tc_command(profile)

        if dry_run:
            return NetemCommandResult(
                success=True,
                dry_run=True,
                applied_command=cmd,
                interface=self.interface,
                profile_id=profile.profile_id,
            )

        if not self.opt_in:
            raise RuntimeError(
                "LinuxNetemAdapter requires explicit user opt-in (opt_in=True) to apply network configuration."
            )

        status = self.check_support()
        if not status.is_supported:
            raise RuntimeError(
                f"Cannot apply Linux netem rules: {'; '.join(status.reasons)}"
            )

        if self.interface in self.PROTECTED_DEFAULT_INTERFACES and not self.allow_primary_interface:
            raise PermissionError(
                f"Refusing to apply traffic shaping to primary interface '{self.interface}' "
                "without explicit allow_primary_interface=True override."
            )

        try:
            subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            self._last_applied_command = cmd
            return NetemCommandResult(
                success=True,
                dry_run=False,
                applied_command=cmd,
                interface=self.interface,
                profile_id=profile.profile_id,
            )
        except subprocess.CalledProcessError as e:
            return NetemCommandResult(
                success=False,
                dry_run=False,
                applied_command=cmd,
                interface=self.interface,
                profile_id=profile.profile_id,
                error_message=e.stderr.strip() or str(e),
            )

    def clear_rules(self, dry_run: bool = True) -> NetemCommandResult:
        """
        Remove netem qdisc rules and restore default network behavior.
        """
        cmd = f"tc qdisc del dev {self.interface} root"

        if dry_run or not self.opt_in:
            return NetemCommandResult(
                success=True,
                dry_run=True,
                applied_command=cmd,
                interface=self.interface,
                profile_id=None,
            )

        status = self.check_support()
        if not status.is_supported:
            raise RuntimeError(f"Cannot clear netem rules: {'; '.join(status.reasons)}")

        try:
            subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            self._last_applied_command = None
            return NetemCommandResult(
                success=True,
                dry_run=False,
                applied_command=cmd,
                interface=self.interface,
                profile_id=None,
            )
        except subprocess.CalledProcessError as e:
            return NetemCommandResult(
                success=False,
                dry_run=False,
                applied_command=cmd,
                interface=self.interface,
                profile_id=None,
                error_message=e.stderr.strip() or str(e),
            )

    def get_cleanup_instructions(self) -> str:
        """
        Return user-facing guidance for restoring interface settings.
        """
        return (
            f"To restore network interface '{self.interface}' to default unshaped behavior, run:\n"
            f"  sudo tc qdisc del dev {self.interface} root\n"
            f"To inspect active qdisc rules:\n"
            f"  tc qdisc show dev {self.interface}"
        )
