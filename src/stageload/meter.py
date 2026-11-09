"""Process and system memory on macOS, sampled into a JSONL trace.

``process_footprint`` is ``ri_phys_footprint`` from ``proc_pid_rusage``: the number Activity
Monitor shows in its Memory column. It includes Metal allocations of the process.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import functools
import json
import os
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from .events import Event
from .trace import summarize

_RUSAGE_INFO_V2 = 2
_RUSAGE_FIELDS = (
    "ri_user_time",
    "ri_system_time",
    "ri_pkg_idle_wkups",
    "ri_interrupt_wkups",
    "ri_pageins",
    "ri_wired_size",
    "ri_resident_size",
    "ri_phys_footprint",
    "ri_proc_start_abstime",
    "ri_proc_exit_abstime",
    "ri_child_user_time",
    "ri_child_system_time",
    "ri_child_pkg_idle_wkups",
    "ri_child_interrupt_wkups",
    "ri_child_pageins",
    "ri_child_elapsed_abstime",
    "ri_diskio_bytesread",
    "ri_diskio_byteswritten",
)


class _RusageInfoV2(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in _RUSAGE_FIELDS]


class _XswUsage(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_uint64),
        ("avail", ctypes.c_uint64),
        ("used", ctypes.c_uint64),
        ("pagesize", ctypes.c_uint32),
        ("encrypted", ctypes.c_int32),
    ]


def _require_macos() -> None:
    if sys.platform != "darwin":
        raise OSError("stageload's memory readers use macOS interfaces (libproc and sysctl)")


@functools.cache
def _libc() -> ctypes.CDLL:
    return ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)


@functools.cache
def _libproc() -> ctypes.CDLL:
    return ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)


def process_footprint(pid: int | None = None) -> int:
    _require_macos()
    target = pid or os.getpid()
    info = _RusageInfoV2()
    if _libproc().proc_pid_rusage(target, _RUSAGE_INFO_V2, ctypes.byref(info)) != 0:
        raise OSError(ctypes.get_errno(), f"proc_pid_rusage failed for pid {target}")
    return int(info.ri_phys_footprint)


def _sysctl(name: str, value: Any) -> None:
    size = ctypes.c_size_t(ctypes.sizeof(value))
    rc = _libc().sysctlbyname(
        name.encode(), ctypes.byref(value), ctypes.byref(size), None, ctypes.c_size_t(0)
    )
    if rc != 0:
        raise OSError(ctypes.get_errno(), f"sysctlbyname({name!r}) failed")


def swap_used() -> int:
    _require_macos()
    usage = _XswUsage()
    _sysctl("vm.swapusage", usage)
    return int(usage.used)


def memorystatus_level() -> int:
    """Percent of memory available (0-100), the gauge the kernel's memory pressure logic uses."""
    _require_macos()
    level = ctypes.c_int32()
    _sysctl("kern.memorystatus_level", level)
    return int(level.value)


class _VmStatistics64(ctypes.Structure):
    """``struct vm_statistics64`` of <mach/vm_statistics.h>, what ``vm_stat`` prints."""

    _fields_ = [
        ("free_count", ctypes.c_uint32),
        ("active_count", ctypes.c_uint32),
        ("inactive_count", ctypes.c_uint32),
        ("wire_count", ctypes.c_uint32),
        ("zero_fill_count", ctypes.c_uint64),
        ("reactivations", ctypes.c_uint64),
        ("pageins", ctypes.c_uint64),
        ("pageouts", ctypes.c_uint64),
        ("faults", ctypes.c_uint64),
        ("cow_faults", ctypes.c_uint64),
        ("lookups", ctypes.c_uint64),
        ("hits", ctypes.c_uint64),
        ("purges", ctypes.c_uint64),
        ("purgeable_count", ctypes.c_uint32),
        ("speculative_count", ctypes.c_uint32),
        ("decompressions", ctypes.c_uint64),
        ("compressions", ctypes.c_uint64),
        ("swapins", ctypes.c_uint64),
        ("swapouts", ctypes.c_uint64),
        ("compressor_page_count", ctypes.c_uint32),
        ("throttled_count", ctypes.c_uint32),
        ("external_page_count", ctypes.c_uint32),
        ("internal_page_count", ctypes.c_uint32),
        ("total_uncompressed_pages_in_compressor", ctypes.c_uint64),
    ]


_HOST_VM_INFO64 = 4


def _vm_statistics() -> tuple[_VmStatistics64, int]:
    """The machine's page counts (``host_statistics64``) and the page size in bytes."""
    _require_macos()
    libc = _libc()
    libc.mach_host_self.restype = ctypes.c_uint32
    info = _VmStatistics64()
    count = ctypes.c_uint32(ctypes.sizeof(info) // 4)
    rc = libc.host_statistics64(
        ctypes.c_uint32(libc.mach_host_self()), _HOST_VM_INFO64, ctypes.byref(info),
        ctypes.byref(count),
    )
    if rc != 0:
        raise OSError(f"host_statistics64 failed: kern_return {rc}")
    page = ctypes.c_uint64()
    _sysctl("hw.pagesize", page)
    return info, int(page.value)


def compressor_used() -> int:
    """Bytes the memory compressor occupies: vm_stat's «Pages occupied by compressor».

    2026-10-24: the Mac hung with the compressor at some 15 GB while the swap the guard watched
    looked tolerable and the level read 40 %."""
    info, page = _vm_statistics()
    return int(info.compressor_page_count) * page


def wired_used() -> int:
    """Bytes of wired memory: vm_stat's «Pages wired down», Metal buffers among them."""
    info, page = _vm_statistics()
    return int(info.wire_count) * page


def machine_ram() -> int:
    """The machine's RAM in bytes (``hw.memsize``; the page count elsewhere)."""
    if sys.platform == "darwin":
        size = ctypes.c_uint64()
        _sysctl("hw.memsize", size)
        return int(size.value)
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


@dataclass(frozen=True)
class Readers:
    footprint: Callable[[], int]
    swap_used: Callable[[], int]
    level: Callable[[], int]
    # written as compressor_gb and wired_gb in each sample when given
    compressor: Callable[[], int] | None = None
    wired: Callable[[], int] | None = None


def system_readers(pid: int | None = None) -> Readers:
    return Readers(
        footprint=lambda: process_footprint(pid), swap_used=swap_used, level=memorystatus_level,
        compressor=compressor_used, wired=wired_used,
    )


class MemoryMeter:
    """Writes this process's footprint and the system's swap to a JSONL trace, with events.

    Use it as a context manager; every line is flushed as it is written, so the trace survives
    a kill.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        interval: float = 0.5,
        meta: dict[str, Any] | None = None,
        readers: Readers | None = None,
    ) -> None:
        self.path = Path(path)
        self.interval = interval
        self.meta = meta or {}
        self.readers = readers or system_readers()
        self._records: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._file: TextIO | None = None
        self._t0 = time.monotonic()
        self._last_error: str | None = None

    def __enter__(self) -> MemoryMeter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8")
        self._t0 = time.monotonic()
        utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._write({"type": "start", "utc": utc, "interval": self.interval, **self.meta})
        try:
            self.sample()
        except BaseException:
            self._file.close()
            self._file = None
            raise
        self._thread = threading.Thread(target=self._run, name="stageload-meter", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._sample_or_note()
        self._write({"type": "end"})
        if self._file is not None:
            self._file.close()
            self._file = None

    def sample(self) -> dict[str, Any]:
        record = {
            "type": "sample",
            "t": round(time.monotonic() - self._t0, 3),
            "footprint": self.readers.footprint(),
            "swap_used": self.readers.swap_used(),
            "level": self.readers.level(),
        }
        for key, reader in (("compressor_gb", self.readers.compressor),
                            ("wired_gb", self.readers.wired)):
            if reader is not None:
                try:
                    record[key] = round(reader() / 2**30, 2)
                except OSError:
                    pass  # the sample keeps what the job's own readings say
        self._write(record)
        return record

    def emit(self, event: Event) -> None:
        record = {"type": "event", **event.as_dict()}
        record["t"] = round(event.t - self._t0, 3)
        self._write(record)

    def mark(self, label: str) -> None:
        self.emit(Event("mark", note=label))

    def summary(self) -> dict[str, Any]:
        with self._lock:
            records = list(self._records)
        return summarize(records)

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self._sample_or_note()

    def _sample_or_note(self) -> None:
        """A failed read is written to the trace, once until a read succeeds again, and sampling
        goes on."""
        try:
            self.sample()
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            if message != self._last_error:
                self._last_error = message
                t = round(time.monotonic() - self._t0, 3)
                self._write({"type": "error", "t": t, "error": message})
        else:
            self._last_error = None

    def _write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, separators=(",", ":"))
        with self._lock:
            self._records.append(record)
            if self._file is not None:
                self._file.write(line + "\n")
                self._file.flush()
