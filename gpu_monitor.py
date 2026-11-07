#!/usr/bin/env python3
"""
gpu_monitor.py -- DGX Spark (GB10) GPU sampler, driven by gpu_monitor.timer.

One invocation = one sample:
  * util / temp / power  via `nvidia-smi --query-gpu=...`
  * GPU memory + procs   via `nvidia-smi --query-compute-apps=...`
    (GB10's memory.used query returns [N/A] -- never rely on it)
  * appends ONE compact line to ~/gpu_monitor_log.txt
  * writes ONE report file to ~/gpu_reports/
 * self-prunes ~/gpu_reports/ older than REPORT_KEEP_D days
 * samples MemAvailable / MemFree / PSI from /proc and runs the unified-memory
 * guard: when allocatable memory drops under the floor (or GPU memory passes
 * the ceiling) on two consecutive samples, SIGTERM the top GPU consumer and
 *   escalate to SIGKILL after a grace period.  See the GUARD_* block below.
 * counts NV_ERR_NO_MEMORY lines the kernel log gained since the previous
 *   sample -- literal evidence the driver is refusing allocations.  Joins
 *   the breach set with the normal two-sample confirmation, so a lone
 *   transient refusal recovers instead of killing.  Technique credited to
 *   MiaAI Lab's memwatch.sh (AGPL-3.0-or-later,
 *   github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark);
 *   reimplemented from scratch, no code taken.
 * stall guard: the engine's own log silent for STALL_QUIET_S while the GPU
 *   is pegged (>= STALL_UTIL_PCT) means the engine stopped making progress
 *   with memory still healthy -- the signature the box hard-locked from on
 *   2026-10-24/25, which no memory floor ever saw.  Snapshots the engine's
 *   tail, SIGTERMs the engine, escalates to SIGKILL after STALL_GRACE_S, and
 *   keeps a rolling copy of the engine tail in ~/gpu_engine_tail.txt (a hard
 *   lock loses the engine log's unflushed buffer as NUL padding).
 *   GPU_MONITOR_STALL_GUARD=0 disarms it.
    (2026-09-20 fix: old build wrote an unbounded report every run -> 1.45 M
    files / 5.7 GB; also old regexes matched the May nvidia-smi layout and
    logged only N/A ever since)

Failure mode: if every GPU query fails the sample line still lands (all N/A)
but exit code is 1 -> systemd Restart=on-failure retries in 60 s.
Never use Restart=always with a script that exits 0 immediately: it becomes
a hot self-restarting loop.

This unit uses Restart=always + RestartSec=10 to get a ~10 s sampling cadence;
gpu_monitor.timer's 30 min OnUnitInactiveSec never fires because the service
never stays inactive that long.
"""

import ctypes
import errno
import fcntl
import glob
import math
import os
import platform
import re
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime

HOME         = "/home/cpatriaw"
LOG_PATH     = os.path.join(HOME, "gpu_monitor_log.txt")
REPORT_DIR   = os.path.join(HOME, "gpu_reports")
REPORT_KEEP_D = 7                      # days of per-sample reports to keep

# --- unified-memory guard (2026-10-24) --------------------------------------
# GB10 has ONE memory pool: the GPU allocates from the same DRAM the CPU
# uses, and the NVIDIA driver does not charge those pages to the allocating
# process's cgroup (maya holds ~90 GiB while its cgroup reports ~4 GiB).  A
# cgroup MemoryMax therefore cannot see this coming -- only the real
# MemAvailable reading can.  When the driver runs out of pages,
# _memdescAllocInternal returns NV_ERR_NO_MEMORY and its error path
# deadlocks: the box hard-locks with nothing on the console, and since the
# kernel watchdog is disabled ("NMI not fully supported") it has to be
# power-cycled.  Cheaper to kill one server than to lose the whole uptime.
#
# MemFree is NOT a usable trigger: the kernel pins it at ~1.2 GiB by design
# and pushes everything else into page cache.  MemAvailable is the kernel's
# own estimate of what is allocatable without swapping.
#
# Two consecutive breaches inside GUARD_CONSEC_S are required, so a
# transient dip never costs a running server.  GPU_MONITOR_GUARD=0 disarms
# the whole thing without editing this file.
#
# THE SIGNAL THAT MATTERS IS DEVICE-FREE, NOT MemAvailable.  On GB10 the
# driver allocates GPU memory out of the same DRAM the CPU uses, so when the
# engine has taken 115 of 119.7 GiB the OS still reports a comfortable
# MemAvailable (~20 GiB, mostly page cache) while the driver is in fact a
# couple of GiB from NV_ERR_NO_MEMORY.  nvidia-smi and NVML both answer
# [N/A] for memory.total/used/free on this part, and the engine prints its
# own "N MiB of VRAM free" exactly once (glm_pack_generate runs a single
# time), so cudaMemGetInfo through ctypes is the only cheap live source.
# Measured: ~0.3 s and ~170 MiB of device memory per call, released at once.
#
# DEVICE_FREE_FLOOR_MIB 3072 sits above the engine's own "LOW" warning
# (256 MiB) and above the ~1021 MiB the box was at when it hard-locked.
def _env_float(name, default, *, minimum=0.0, exclusive=False):
    raw = os.environ.get(name, default)
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a finite number (got {raw!r})") from exc
    if not math.isfinite(value) or (value <= minimum if exclusive else value < minimum):
        relation = "greater than" if exclusive else "at least"
        raise ValueError(f"{name} must be finite and {relation} {minimum} (got {raw!r})")
    return value


def _env_bool(name, default):
    raw = os.environ.get(name, default).strip().lower()
    if raw in ("1", "yes", "true", "on"):
        return True
    if raw in ("0", "no", "false", "off"):
        return False
    raise ValueError(f"{name} must be a boolean value (got {raw!r})")


GUARD_ENABLED         = _env_bool("GPU_MONITOR_GUARD", "1")
DEVICE_FREE_FLOOR_MIB = _env_float("GPU_MONITOR_DEVICE_FREE_FLOOR_MIB", "3072", exclusive=True)
MEM_FLOOR_GIB         = _env_float("GPU_MONITOR_MEM_FLOOR_GIB", "6", exclusive=True)
GPU_CEIL_MIB          = _env_float("GPU_MONITOR_GPU_CEIL_MIB", "112640", exclusive=True)
WARN_FREE_MIB         = _env_float("GPU_MONITOR_WARN_FREE_MIB", "4096")
WARN_FLOOR_GIB        = _env_float("GPU_MONITOR_WARN_FLOOR_GIB", "12")
WARN_EVERY_S          = _env_float("GPU_MONITOR_WARN_EVERY_S", "300", exclusive=True)
GUARD_CONSEC_S        = _env_float("GPU_MONITOR_GUARD_CONSEC_S", "30", exclusive=True)
GUARD_GRACE_S         = _env_float("GPU_MONITOR_GUARD_GRACE_S", "30", exclusive=True)

# 2026-10-30 crash-loop fixes (14:11-15:01 SIGTERM/SIGKILL loop, seven
# engines, each killed seconds after boot):
#
# The engine's prefill path borrows ~4.4 GiB transiently and every LOW sample
# in that loop read a healthy MemAvailable (~31 GiB) while the 2026-10-25
# hard locks read ~20 GiB -- device memory IS host RAM on GB10, so a
# device_free breach is only fatal when allocatable host RAM has dropped too.
# DEVICE_BREACH_MEM_FLOOR_GIB sits between those two readings: below it the
# breach kills exactly as before, above it the dip is downgraded to WARN (the
# prefill borrow, or page cache the kernel can reclaim).  NV_ERR_NO_MEMORY,
# the GPU ceiling and OS-side signals (mem/PSI/swap) kill regardless.
DEVICE_BREACH_MEM_FLOOR_GIB = _env_float("GPU_MONITOR_DEVICE_BREACH_MEM_FLOOR_GIB", "24")

# Every kill in that loop hit a freshly restarted engine seconds after boot --
# the previous engine's death left the guard armed and the next sample
# SIGTERMed an engine still warming up.  A victim younger than this on a
# device_free-only breach gets WARN treatment instead; if the breach is real
# it is caught right after warm-up, and any other breach signal ignores the
# grace and kills immediately.
GUARD_BOOT_GRACE_S     = _env_float("GPU_MONITOR_GUARD_BOOT_GRACE_S", "240")

# 2026-10-25: two guards against concurrent invocations (see
# docs/2026-10-25-guard-concurrency.md).  A second invocation running at the
# same time as the first used to read the first's freshly-written armed state
# as its own confirming sample, collapsing the two-sample rule into a
# one-sample kill.  (1) A whole-run exclusive flock: a concurrent invocation
# skips sampling and guard logic entirely.  (2) An armed-age floor: a
# confirming sample must observe an armed state at least this old, i.e.
# written by an EARLIER invocation, never a simultaneous one.
MIN_CONFIRM_AGE_S     = _env_float("GPU_MONITOR_MIN_CONFIRM_AGE_S", "8")
LOCK_PATH             = os.path.join(HOME, ".gpu_monitor_guard_lock")

# 2026-10-24 additions: two more breach signals and smarter victim choice.
# PSI memory "full" avg10 is the kernel saying allocation is already
# stalling -- the earliest hard-lock warning there is, visible before any
# floor in this file trips.  Swap growth is the thrash signature that
# precedes a lock.  And an OS-side breach (low MemAvailable / high swap /
# PSI) is often caused by something that holds no GPU memory at all -- a
# runaway build, say -- in which case killing the engine is wrong: pick the
# biggest-RSS process instead.  GPU_MONITOR_PROTECT is a comma list of pids
# or /proc names the OS-side victim pick will always skip.
PSI_FULL_FLOOR        = _env_float("GPU_MONITOR_PSI_FULL_FLOOR", "5.0")
SWAP_FLOOR_GIB        = _env_float("GPU_MONITOR_SWAP_FLOOR_GIB", "8")
OS_VICTIM_BY_RSS      = _env_bool("GPU_MONITOR_OS_VICTIM", "1")

# 2026-10-25: RSS-growth attribution for device-free breaches.  On GB10 the
# device pool IS system RAM, so a process that holds no GPU memory at all --
# an unwrapped build, say -- can drive device_free below the floor and the
# old victim pick (top GPU consumer) would kill the engine for it.  Every
# healthy sample writes a pid->RSS baseline; on a device-free breach the
# victim is the process whose RSS grew the most since the previous sample,
# and only when nothing grew does the old top-GPU-consumer pick apply (the
# engine's own CUDA allocations never show in RSS).  RSS_GROW_MIN_KIB keeps
# page-noise from framing a bystander; RSS_BASE_MAX_AGE_S discards a stale
# baseline (monitor was down) rather than mis-attribute.
RSS_BASE_PATH         = os.path.join(HOME, ".gpu_monitor_guard_rss")
RSS_BASE_MAX_AGE_S    = _env_float("GPU_MONITOR_RSS_BASE_MAX_AGE_S", "300", exclusive=True)
RSS_GROW_MIN_KIB      = _env_float("GPU_MONITOR_RSS_GROW_MIN_KIB", "262144")
_PROTECT_TOKENS       = {t.strip() for t in
                         os.environ.get("GPU_MONITOR_PROTECT", "").replace(",", " ").split()
                         if t.strip()}

STATE_PATH            = os.path.join(HOME, ".gpu_monitor_guard_state")
WARN_PATH             = os.path.join(HOME, ".gpu_monitor_guard_warned")
GUARD_LOG_PATH        = os.path.join(HOME, "gpu_monitor_guard.log")

# NV_ERR_NO_MEMORY journal watch (2026-10-25, technique credited to MiaAI
# Lab's memwatch.sh in the header): cursor file keeps the last window's
# timestamp so the one-shot process can count only NEW kernel-log lines.
NVRM_PATH             = os.path.join(HOME, ".gpu_monitor_guard_nvrm")
NVRM_JOURNAL_TIMEOUT  = 10

# --- stall / liveness guard (2026-10-25) -------------------------------------
# The guard above answers "is the box about to run out of device memory".
# It cannot see the failure the box has actually hard-locked from three
# times (2026-10-24 19:40, 2026-10-24 21:08, 2026-10-25 17:11): the engine
# stops making progress while the GPU stays pegged at 94-95%, with
# device-free still healthy (7-15 GiB), the driver refusing nothing, and the
# kernel log silent to the last line.  On GB10 device memory IS system RAM
# and the kernel watchdog is disabled, so a GPU-side fault takes the whole
# box with it and no userspace watchdog can save it -- but a hang that is
# still only the ENGINE (host-side deadlock, stuck copy lane) is
# recoverable, and in both cases the evidence worth having is the engine's
# own last output, which dies with the box.
#
# So: watch the file the engine actually writes to (its own stderr fd --
# /proc/<engine>/fd/2, the same inode the "[maya] ..." progress lines land
# in; fd 1 is a pipe to the server and is skipped).  When that file has been
# silent for STALL_QUIET_S while the GPU is at STALL_UTIL_PCT or more, treat
# it as a stall: snapshot the engine's tail, SIGTERM the engine, escalate to
# SIGKILL after STALL_GRACE_S.  The launcher restarts the engine on the next
# request, so a false positive costs one restart -- and the threshold sits
# well above the longest legitimate silent stretch there is (one
# 32768-token prefill chunk, ~46 s, with the chunk stage line silenced;
# normally the launcher prints a progress line every 10 s).
#
# An idle engine is silent by design, which is why the utilization gate and
# not the silence alone is what arms this.  A rolling copy of the engine's
# tail is kept on every sample regardless (gpu_engine_tail.txt): after a
# hard lock it is the only record left of what the model was doing, since
# the engine's own log loses its unflushed buffer as NUL padding.
STALL_ENABLED      = _env_bool("GPU_MONITOR_STALL_GUARD", "1")
STALL_UTIL_PCT     = _env_float("GPU_MONITOR_STALL_UTIL_PCT", "90", exclusive=True)
STALL_QUIET_S      = _env_float("GPU_MONITOR_STALL_QUIET_S", "90", exclusive=True)
STALL_GRACE_S      = _env_float("GPU_MONITOR_STALL_GRACE_S", "30", exclusive=True)
STALL_TAIL_LINES   = int(_env_float("GPU_MONITOR_STALL_TAIL_LINES", "40"))
STALL_LOG_PATH     = os.environ.get("GPU_MONITOR_STALL_LOG", "").strip()
STALL_STATE_PATH   = os.path.join(HOME, ".gpu_monitor_stall_state")
STALL_TAIL_PATH    = os.path.join(HOME, "gpu_engine_tail.txt")
STALL_SNAP_DIR     = REPORT_DIR          # stall_*.txt; pruned by nothing

if MIN_CONFIRM_AGE_S > GUARD_CONSEC_S:
    raise ValueError("GPU_MONITOR_MIN_CONFIRM_AGE_S must not exceed GPU_MONITOR_GUARD_CONSEC_S")
if WARN_FREE_MIB < DEVICE_FREE_FLOOR_MIB:
    raise ValueError("GPU_MONITOR_WARN_FREE_MIB must not be below GPU_MONITOR_DEVICE_FREE_FLOOR_MIB")
if WARN_FLOOR_GIB < MEM_FLOOR_GIB:
    raise ValueError("GPU_MONITOR_WARN_FLOOR_GIB must not be below GPU_MONITOR_MEM_FLOOR_GIB")


def _atomic_write(path, contents):
    """Atomically replace a small state file, durably before returning."""
    directory = os.path.dirname(path) or "."
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=directory,
                prefix=f".{os.path.basename(path)}.", delete=False) as f:
            temp_path = f.name
            f.write(contents)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
        temp_path = None
        dir_fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        if temp_path is not None:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass


def _q(args, timeout=15):
    try:
        r = subprocess.run(["nvidia-smi"] + args,
                           capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            return None
        out = r.stdout.strip()
        return out if out else None
    except Exception:
        return None


def gpu_row():
    """(util, temp, power) from one --query-gpu call (fields proven on GB10)."""
    out = _q(["--query-gpu=utilization.gpu,temperature.gpu,power.draw",
              "--format=csv,noheader"])
    if not out:
        return None, None, None
    parts = [p.strip() for p in out.split(",")]

    def num(i):
        if i >= len(parts):
            return None
        m = re.match(r"(\d+(?:\.\d+)?)", parts[i])
        return float(m.group(1)) if m else None

    return num(0), num(1), num(2)


def compute_apps():
    """[(pid, MiB, name)] via --query-compute-apps (GB10-safe memory source)."""
    out = _q(["--query-compute-apps=pid,used_gpu_memory,process_name",
              "--format=csv,noheader"])
    if not out:
        return []
    procs = []
    for line in out.splitlines():
        if "," not in line:
            continue
        parts = [p.strip() for p in line.split(",")]
        m = re.match(r"(\d+(?:\.\d+)?)", parts[1]) if len(parts) > 1 else None
        if not m:
            continue
        pid   = parts[0] if re.match(r"\d+$", parts[0]) else "?"
        mib   = float(m.group(1))
        pname = parts[2] if len(parts) > 2 else "?"
        procs.append((pid, mib, pname))
    return procs


def fmt(v, unit=""):
    if v is None:
        return "N/A"
    if float(v).is_integer():
        return f"{int(v)}{unit}"
    return f"{v}{unit}"


def prune_old_reports():
    cutoff = datetime.now().timestamp() - REPORT_KEEP_D * 86400
    n = 0
    try:
        names = os.listdir(REPORT_DIR)
    except OSError:
        return 0
    for fn in names:
        if not (fn.startswith("gpu_report_") and fn.endswith(".txt")):
            continue
        fp = os.path.join(REPORT_DIR, fn)
        try:
            if os.path.getmtime(fp) < cutoff:
                os.remove(fp)
                n += 1
        except OSError:
            pass
    return n


def kib(k, unit="MiB"):
    """KiB -> human string, or N/A."""
    if k is None:
        return "N/A"
    return f"{k / 1024:.1f}{unit}"


_CUDART = None
_CUDART_TRIED = False


def device_free():
    """(free MiB, total MiB) from cudaMemGetInfo, or (None, None) if unavailable.

    This is the pool the NVIDIA driver actually allocates from.  On GB10 it is
    system DRAM, so it is the number that ran out before the hard lockup - and
    the only one MemAvailable gets wrong, because MemAvailable counts the
    page cache as reclaimable while the driver could not reclaim it in time.
    """
    global _CUDART, _CUDART_TRIED
    if not _CUDART_TRIED:
        _CUDART_TRIED = True
        cands = sorted(glob.glob("/usr/local/cuda*/lib64/libcudart.so*"), reverse=True)
        cands.append("libcudart.so")
        for c in cands:
            try:
                lib = ctypes.CDLL(c)
                lib.cudaMemGetInfo.restype = ctypes.c_int
                lib.cudaMemGetInfo.argtypes = [ctypes.POINTER(ctypes.c_size_t)] * 2
                _CUDART = lib
                break
            except OSError:
                continue
    if _CUDART is None:
        return None, None
    free = ctypes.c_size_t(0)
    total = ctypes.c_size_t(0)
    if _CUDART.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)) != 0:
        return None, None
    return free.value >> 20, total.value >> 20


def meminfo():
    """{key: value_in_KiB} from /proc/meminfo."""
    d = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                k, _, v = line.partition(":")
                d[k.strip()] = int(v.strip().split()[0])
    except (OSError, ValueError):
        pass
    return d


def psi_memory():
    """(some avg10, full avg10) memory stall percentages, or (None, None)."""
    try:
        with open("/proc/pressure/memory") as f:
            txt = f.read()
    except OSError:
        return None, None
    out = []
    for key in ("some", "full"):
        m = re.search(rf"^{key}\s+avg10=([\d.]+)", txt, re.M)
        out.append(float(m.group(1)) if m else None)
    return out[0], out[1]


def nvrm_errors():
    """Count of NV_ERR_NO_MEMORY lines the kernel log gained since the
    previous sample, or None when the journal is unavailable.

    Windows are adjacent [last, now] via epoch cursors in NVRM_PATH.
    journalctl's --since/--until are inclusive at the boundary instant, so a
    line stamped exactly at the boundary can double-count -- harmless for a
    trigger, and preferred over a since=last+eps window that could silently
    drop a line arriving within the epsilon.  The first run only
    initialises the cursor: there is no lower bound, so counting then could
    replay an old flood as a fresh breach.

    The cursor advances only AFTER a successful query: a timed-out or
    failed journalctl leaves it at `last`, so the next sample re-covers the
    whole [last, next] window and no lines are lost -- a long journal
    outage counts its lines once, on the first sample that succeeds.
    """
    try:
        with open(NVRM_PATH) as f:
            last = float(f.read().strip())
    except (OSError, ValueError):
        last = None
    now = time.time()
    if last is None:
        try:
            _atomic_write(NVRM_PATH, f"{now}")
        except OSError:
            return None
        return 0
    if last >= now:
        return 0
    try:
        r = subprocess.run(
            ["journalctl", "-k", "-q", "--no-pager",
             f"--since=@{last}", f"--until=@{now}"],
            capture_output=True, text=True, timeout=NVRM_JOURNAL_TIMEOUT)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    try:
        _atomic_write(NVRM_PATH, f"{now}")
    except OSError:
        return None
    return sum(1 for ln in r.stdout.splitlines() if "NV_ERR_NO_MEMORY" in ln)


def guard_log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(GUARD_LOG_PATH, "a") as f:
            f.write(f"[{ts}] {msg}\n")
    except OSError:
        pass


def _state():
    try:
        with open(STATE_PATH) as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""


def _set_state(s):
    _atomic_write(STATE_PATH, s)


def _warn_due(now):
    """True at most once per WARN_EVERY_S, so a long stretch in the warning
    band does not flood the guard log (it samples every ~10 s)."""
    try:
        with open(WARN_PATH) as f:
            last = float(f.read().strip() or 0)
    except (OSError, ValueError):
        last = 0.0
    if now - last < WARN_EVERY_S:
        return False
    try:
        with open(WARN_PATH, "w") as f:
            f.write(f"{now}")
    except OSError:
        pass
    return True


def _clear_warn():
    try:
        if os.path.exists(WARN_PATH):
            os.remove(WARN_PATH)
    except OSError:
        pass


def _alive(pid):
    # 2026-10-24: a killed-but-unreaped victim is a zombie -- /proc/<pid>
    # still exists, so a pure existence check treated it as alive and the
    # guard kept "draining" and eventually SIGKILLed a process that was
    # already dead.  /proc/<pid>/stat state 'Z' is the reliable answer.
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return False


def _os_victim(self_pid):
    """(pid, name, rss_kib) of the biggest-RSS process, or None.

    Used when the breach is OS-side only (nothing ran out of the GPU pool):
    the offender is then likelier to be a plain memory hog than the engine.
    Skips the monitor itself, pid 1/2 (init, kthreadd) and anything in
    GPU_MONITOR_PROTECT (pid or /proc Name).  Kernel threads report no
    VmRSS and are skipped by the same check.
    """
    best = None
    try:
        entries = os.listdir("/proc")
    except OSError:
        return None
    for d in entries:
        if not d.isdigit():
            continue
        pid = int(d)
        if pid in (self_pid, 1, 2):
            continue
        try:
            with open(f"/proc/{pid}/status") as f:
                txt = f.read()
        except OSError:
            continue
        m = re.search(r"^VmRSS:\s+(\d+) kB", txt, re.M)
        if not m:
            continue
        rss = int(m.group(1))
        if rss <= 0:
            continue
        nm = ""
        mm = re.search(r"^Name:\s+(.+)", txt, re.M)
        if mm:
            nm = mm.group(1).strip()
        if f"{pid}" in _PROTECT_TOKENS or nm in _PROTECT_TOKENS:
            continue
        if best is None or rss > best[2]:
            best = (pid, nm, rss)
    return best


def _signal_pid(pid, sig, expected_starttime):
    """Signal only the process instance whose /proc start time was recorded."""
    pidfd = None
    try:
        pidfd = _pidfd_open(int(pid))
        if _starttime(int(pid)) != expected_starttime:
            return False
        _pidfd_send_signal(pidfd, sig)
        return True
    except (ProcessLookupError, OSError, ValueError):
        return False
    finally:
        if pidfd is not None:
            try:
                os.close(pidfd)
            except OSError:
                pass


_PIDFD_SYSCALLS = {
    # Linux assigns these syscall numbers on both supported 64-bit ABIs.
    "aarch64": (434, 424),
    "x86_64": (434, 424),
}


def _pidfd_open(pid):
    if hasattr(os, "pidfd_open"):
        return os.pidfd_open(pid, 0)
    syscall_numbers = _PIDFD_SYSCALLS.get(platform.machine().lower())
    if not sys.platform.startswith("linux") or syscall_numbers is None:
        raise OSError(errno.EOPNOTSUPP, "pidfd_open is unavailable")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(ctypes.c_long(syscall_numbers[0]), ctypes.c_int(pid),
                          ctypes.c_uint(0))
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return int(result)


def _pidfd_send_signal(pidfd, sig):
    if hasattr(signal, "pidfd_send_signal"):
        signal.pidfd_send_signal(pidfd, sig)
        return
    syscall_numbers = _PIDFD_SYSCALLS.get(platform.machine().lower())
    if not sys.platform.startswith("linux") or syscall_numbers is None:
        raise OSError(errno.EOPNOTSUPP, "pidfd_send_signal is unavailable")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(ctypes.c_long(syscall_numbers[1]), ctypes.c_int(pidfd),
                          ctypes.c_int(sig), ctypes.c_void_p(), ctypes.c_uint(0))
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


_LAST_RSS = None      # {pid: rss_kib} of this sample, written by memory_guard
_REBASE_OK = False    # True only on a fully healthy sample -> main() rebases


def _rss_map():
    """{pid: rss_kib} for every process with resident pages.

    Kernel threads report no VmRSS and are absent.  One /proc/*/status read
    each, a few ms for a few hundred processes -- negligible next to the
    nvidia-smi call in the same sample.
    """
    d = {}
    try:
        entries = os.listdir("/proc")
    except OSError:
        return d
    for e in entries:
        if not e.isdigit():
            continue
        try:
            with open(f"/proc/{e}/status") as f:
                txt = f.read()
        except OSError:
            continue
        m = re.search(r"^VmRSS:\s+(\d+) kB", txt, re.M)
        if m:
            rss = int(m.group(1))
            if rss > 0:
                d[int(e)] = rss
    return d


def _starttime(pid):
    """Clock-tick start time of pid (stat field 22), or None when it died.

    Field 22 sits after the last ')' (comm may contain them), at index 19 of
    the remaining tokens (index i is field 3+i; _alive uses the same trick).
    Constant for the process's lifetime within one boot, so a baseline entry
    whose start time no longer matches belongs to a REUSED pid.
    """
    try:
        with open(f"/proc/{pid}/stat") as f:
            return int(f.read().rsplit(")", 1)[1].split()[19])
    except (OSError, IndexError, ValueError):
        return None


_BOOT_TIME = None


def _boot_time():
    """Epoch second of the current boot, from /proc/stat's btime (cached)."""
    global _BOOT_TIME
    if _BOOT_TIME is None:
        try:
            with open("/proc/stat") as f:
                for line in f:
                    if line.startswith("btime"):
                        _BOOT_TIME = int(line.split()[1])
                        break
        except (OSError, IndexError, ValueError):
            return None
    return _BOOT_TIME


def _proc_age(pid, now):
    """Seconds since pid started, or None when it died / cannot be timed.

    /proc/<pid>/stat starttime is CLOCK TICKS since boot (field 22, the same
    tick unit _alive and the identity checks see), so divide by SC_CLK_TCK
    and subtract from the boot epoch -- treating ticks as seconds makes a
    53-minute-old engine read as freshly started.
    """
    st = _starttime(pid)
    if st is None:
        return None
    bt = _boot_time()
    if bt is None:
        return None
    hz = os.sysconf("SC_CLK_TCK") or 100
    return max(0.0, now - bt - st / hz)


def _load_rss_base():
    """(timestamp, {pid: (rss_kib, starttime)}) of the last baseline, or
    (0, {}).  Tolerates the pre-starttime two-token format (pid:rss) as a
    same-process delta with no identity check."""
    try:
        with open(RSS_BASE_PATH) as f:
            parts = f.read().split()
        ts = float(parts[0])
        base = {}
        for tok in parts[1:]:
            bits = tok.split(":")
            if len(bits) == 3:            # pid:starttime:rss
                base[int(bits[0])] = (int(bits[2]), int(bits[1]))
            elif len(bits) == 2:         # legacy pid:rss
                base[int(bits[0])] = (int(bits[1]), None)
        return ts, base
    except (OSError, ValueError, IndexError):
        return 0.0, {}


def _write_rss_base(cur):
    if not cur:
        return
    try:
        with open(RSS_BASE_PATH, "w") as f:
            f.write(f"{time.time()} " + " ".join(
                f"{p}:{_starttime(p)}:{r}"
                for p, r in sorted(cur.items()) if _starttime(p) is not None
            ) + "\n")
    except OSError:
        pass


def _rss_grow_victim(cur, self_pid):
    """(pid, name, growth_kib) of the process whose RSS grew the most since
    the previous sample, or None.

    Only called on a device-free breach: the baseline is deliberately NOT
    rewritten while a breach is active (main() rebases on healthy samples
    only), so growth keeps accumulating against the pre-breach snapshot and
    a slow leak is caught on the confirming sample too.  Requires a fresh
    baseline and growth above RSS_GROW_MIN_KIB; honours the protect list.

    Baseline entries carry the process start time (stat field 22), so a
    pid that died and was REUSED inside the 300 s window cannot inherit the
    old tenant's RSS: a mismatched start time means a fresh process whose
    full RSS counts as growth.  A pid with no /proc/<pid>/stat any more died
    between the sample and the pick and is skipped.
    """
    ts, base = _load_rss_base()
    if not base or time.time() - ts > RSS_BASE_MAX_AGE_S:
        return None
    best_pid, best_kib = None, 0
    for pid, rss in cur.items():
        if pid in (self_pid, 1, 2):
            continue
        st = _starttime(pid)
        if st is None:
            continue            # died between the sample and the pick
        b = base.get(pid)
        base_rss = 0
        if b is not None:
            brss, bst = b
            if bst is None or bst == st:
                base_rss = brss  # same process (or legacy entry): delta
            # else: pid reused -- full RSS counts as growth
        grow = rss - base_rss
        if grow > best_kib:
            best_pid, best_kib = pid, grow
    if best_pid is None or best_kib < RSS_GROW_MIN_KIB:
        return None
    nm = ""
    try:
        with open(f"/proc/{best_pid}/status") as f:
            txt = f.read()
        mm = re.search(r"^Name:\s+(.+)", txt, re.M)
        if mm:
            nm = mm.group(1).strip()
    except OSError:
        return None            # died between the scan and the pick
    if f"{best_pid}" in _PROTECT_TOKENS or nm in _PROTECT_TOKENS:
        return None
    return best_pid, nm, best_kib


_LOCK_FD = None


def _try_lock():
    """True if this invocation holds the whole-run guard lock.

    The lock makes every state-file and baseline read-modify-write
    single-threaded across invocations.  flock is released by the kernel
    when the fd is closed or the holder dies, so a SIGKILLed run can never
    wedge the next sample.  Returns False only when another invocation owns
    the lock; lock setup errors raise instead of running the guard unlocked.
    """
    global _LOCK_FD
    fd = os.open(LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        if exc.errno in (errno.EACCES, errno.EAGAIN):
            os.close(fd)
            return False
        try:
            os.close(fd)
        except OSError:
            pass
        raise
    _LOCK_FD = fd
    return True


def _unlock():
    global _LOCK_FD
    if _LOCK_FD is not None:
        try:
            fcntl.flock(_LOCK_FD, fcntl.LOCK_UN)
            os.close(_LOCK_FD)
        except OSError:
            pass
        _LOCK_FD = None


def memory_guard(mem, dfree, procs, psi_f=None, nvrm=None):
    """Kill the offending process before unified-memory exhaustion hard-locks
    the box.  Returns a short tag for the sample line, or None.

    Triggers, in the order they matter:
      1. device-free below DEVICE_FREE_FLOOR_MIB - the pool the driver
         allocates from, and the thing that actually ran out.  This is the
         only signal that saw the failure coming: MemAvailable still read a
         healthy ~20 GiB while the driver was ~1 GiB from NV_ERR_NO_MEMORY.
         2026-10-30: it kills only when MemAvailable has ALSO dropped below
         DEVICE_BREACH_MEM_FLOOR_GIB (or another signal fires alongside) --
         a dip against healthy host RAM is the prefill borrow, not death.
      2. MemAvailable below MEM_FLOOR_GIB - OS-side exhaustion.
      3. GPU compute-app total above GPU_CEIL_MIB - runaway backstop, mostly
         useful when the cudaMemGetInfo probe is unavailable.
      4. PSI memory full avg10 at/above PSI_FULL_FLOOR for two consecutive
         samples - the kernel saying allocation is already stalling.
      5. swap used at/above SWAP_FLOOR_GIB - the pre-lock thrash signature.
      6. NV_ERR_NO_MEMORY lines new in the kernel log since the previous
         sample - the driver literally refusing an allocation.  Arms like
         any breach; the two-sample confirmation still applies, so one
         transient refusal that the engine retried past recovers.

    Victim: a GPU-pool breach (1, 3 or 6) kills the top GPU consumer, as
    before -- with 2026-10-25 growth attribution first, since on GB10 the
    pool IS system RAM and an RSS hog can cause (1) and (6) without holding
    any GPU memory.  An OS-side-only breach (2, 4, 5) picks the biggest-RSS
    process instead (see _os_victim), because the offender there often
    holds no GPU memory.

    One invocation is one sample, so the two-sample confirmation has to live
    in the state file:
        armed:<ts>            first breach, waiting for a confirming sample
        termed:<pid>:<starttime>:<ts>
                              SIGTERM sent, waiting for exit / grace expiry
    """
    if not GUARD_ENABLED:
        return None

    global _LAST_RSS, _REBASE_OK
    _REBASE_OK = False
    cur_rss = _rss_map()
    _LAST_RSS = cur_rss
    now    = time.time()
    avail  = mem.get("MemAvailable")
    availg = avail / 1048576 if avail is not None else None
    gpu    = sum(p[1] for p in procs)
    state = _state().split(":")

    if state[0] == "armed":
        try:
            armed_at = float(state[1])
            valid_state = len(state) == 2 and math.isfinite(armed_at) and armed_at >= 0
        except (IndexError, ValueError):
            valid_state = False
        if not valid_state:
            guard_log("discarding malformed armed state")
            _set_state("")
            state = [""]
    elif state[0] == "termed":
        try:
            valid_state = (
                len(state) == 4 and state[1].isdigit() and int(state[1]) > 0
                and int(state[2]) >= 0 and math.isfinite(float(state[3]))
                and float(state[3]) >= 0
            )
        except (IndexError, ValueError):
            valid_state = False
        if not valid_state:
            guard_log("discarding malformed or legacy termed state")
            _set_state("")
            state = [""]
    elif state != [""]:
        guard_log("discarding unknown guard state")
        _set_state("")
        state = [""]

    # 1. finish a kill started on an earlier sample -- never block this run
    if state[0] == "termed":
        pid = state[1]
        expected_starttime = int(state[2])
        termed_at = float(state[3])
        if _starttime(int(pid)) != expected_starttime:
            guard_log(f"pid {pid} no longer matches the guarded process; "
                      "discarding stale termination state")
            _set_state("")
            return "guard:state-reset"
        if not _alive(pid):
            guard_log(f"pid {pid} exited after SIGTERM")
            _set_state("")
            return None
        if now - termed_at >= GUARD_GRACE_S:
            if _signal_pid(pid, signal.SIGKILL, expected_starttime):
                guard_log(f"pid {pid} still alive after "
                          f"{GUARD_GRACE_S:.0f}s -> SIGKILL")
            else:
                guard_log(f"pid {pid} was not SIGKILLed: process identity "
                          "changed or pidfd signaling is unavailable")
            _set_state("")
            return "guard:killed"
        return "guard:draining"

    low_free = dfree is not None and dfree < DEVICE_FREE_FLOOR_MIB
    low_avail = availg is not None and availg < MEM_FLOOR_GIB
    high_gpu = gpu > GPU_CEIL_MIB
    low_psi = psi_f is not None and psi_f >= PSI_FULL_FLOOR
    nvrm_hit = nvrm is not None and nvrm > 0
    swap_used = None
    st = mem.get("SwapTotal")
    sf = mem.get("SwapFree")
    if st is not None and sf is not None:
        swap_used = (st - sf) / 1048576.0
    high_swap = swap_used is not None and swap_used >= SWAP_FLOOR_GIB

    # 2026-10-30: downgrade a device_free breach against healthy host RAM to
    # the WARN band.  The engine's prefill path borrows ~4.4 GiB transiently
    # and every LOW sample in the 14:11-15:01 crash loop read MemAvailable
    # ~31 GiB, while the real hard locks read ~20 GiB; the co-floor sits
    # between.  Clearing any armed state here lets the next honest breach
    # re-arm from scratch instead of confirming on a stale one.
    tolerated = False
    if (low_free and not nvrm_hit and not high_gpu and not low_avail
            and not low_psi and not high_swap
            and availg is not None and availg >= DEVICE_BREACH_MEM_FLOOR_GIB):
        low_free = False
        tolerated = True
        if state[0] == "armed":
            _set_state("")
            guard_log(f"tolerated: device_free={dfree}MiB < "
                      f"{int(DEVICE_FREE_FLOOR_MIB)}MiB but mem_available="
                      f"{availg:.1f}GiB >= "
                      f"{DEVICE_BREACH_MEM_FLOOR_GIB:.0f}GiB (prefill borrow "
                      "against healthy host RAM)")
    pool_breach = low_free or nvrm_hit      # driver-pool family: (1), (3), (6)

    if not (low_free or low_avail or high_gpu or low_psi or high_swap
            or nvrm_hit):
        # A tolerated device_free dip is NOT a healthy sample: device_free
        # was below the kill floor this sample, so the pre-breach RSS
        # snapshot must survive for the next honest breach's growth
        # attribution (invariant 4: rebase on healthy samples only).
        if not tolerated:
            _REBASE_OK = True
        if state[0] == "armed":
            _set_state("")
            guard_log(f"recovered: device_free={dfree}MiB "
                      f"mem_available={availg if availg is None else round(availg, 1)}GiB "
                      f"gpu={int(gpu)}MiB")
            _clear_warn()
        elif _warn_due(now) and (
                (dfree is not None and dfree < WARN_FREE_MIB) or
                (availg is not None and availg < WARN_FLOOR_GIB)):
            if dfree is not None and dfree < WARN_FREE_MIB:
                guard_log(f"WARN device_free={dfree}MiB (kill floor "
                          f"{int(DEVICE_FREE_FLOOR_MIB)}MiB) gpu={int(gpu)}MiB")
            else:
                guard_log(f"WARN mem_available={availg:.1f}GiB gpu={int(gpu)}MiB "
                          f"(kill floor {MEM_FLOOR_GIB:.0f}GiB)")
        elif dfree is None and _warn_due(now):
            guard_log("WARN cudaMemGetInfo unavailable - falling back to "
                      "MemAvailable and the GPU ceiling only")
        return None

    armed_age = None
    if state[0] == "armed":
        armed_age = now - float(state[1])

    if armed_age is not None and armed_age < MIN_CONFIRM_AGE_S:
        # An armed state this fresh was written by a CONCURRENT invocation
        # seconds ago; treating it as this sample's confirmation would turn
        # the two-sample rule into a one-sample kill.  Keep the original
        # timestamp untouched so the honest first sighting survives.
        guard_log(f"armed state is only {armed_age:.1f}s old (< "
                  f"{MIN_CONFIRM_AGE_S:g}s) -- concurrent writer? not "
                  "confirming on it")
        return "guard:pending"

    if armed_age is not None and armed_age <= GUARD_CONSEC_S:
        victim = None
        if high_gpu:
            # GPU compute-app ceiling: whoever holds that memory is the
            # culprit by definition.
            if procs:
                victim = max(procs, key=lambda p: p[1])          # (pid, MiB, name)
        elif pool_breach and OS_VICTIM_BY_RSS:
            # 2026-10-25: device memory IS RAM on GB10, so a device-free
            # breach can be caused by a plain RSS hog that holds no GPU
            # memory at all (an unwrapped build, say) -- killing the engine
            # for it is exactly the failure this guard exists to prevent.
            # Kill whoever actually grew since the previous sample.  The
            # engine's CUDA allocations never show in RSS, so genuine
            # engine growth falls through to the old GPU-consumer pick.
            v = _rss_grow_victim(cur_rss, os.getpid())
            if v is not None:
                vk, vn, vgrow = v
                victim = (str(vk), vgrow / 1024.0, vn or "?")
                guard_log(f"victim by RSS growth: {vn}(pid {vk}, "
                          f"+{vgrow / 1048576:.1f}GiB since last sample)")
            elif procs:
                victim = max(procs, key=lambda p: p[1])
        elif pool_breach:
            # growth attribution disabled -> old behaviour
            if procs:
                victim = max(procs, key=lambda p: p[1])
        elif OS_VICTIM_BY_RSS:
            # OS-side-only breach (mem_available / PSI / swap): the offender
            # often holds no GPU memory -- pick the biggest-RSS process.
            # (2026-10-24 fix: the old condition also matched `not procs`,
            # which routed exactly this case into the GPU-consumer pick,
            # found no victim and cleared the state -- armed forever.)
            v = _os_victim(os.getpid())
            if v is not None:
                vk, vn, vrss = v
                victim = (str(vk), vrss / 1024.0, vn or "?")
                guard_log(f"victim by RSS: {vn}(pid {vk}, {vrss} kB) "
                          f"over GPU consumers (OS-side breach)")
            elif procs:
                victim = max(procs, key=lambda p: p[1])
                guard_log("no RSS victim found (protected/none) -> "
                          "falling back to the top GPU consumer")
        if victim is None:
            _set_state("")
            return None
        pid, mib, name = victim
        starttime = _starttime(int(pid))
        if starttime is None:
            guard_log(f"pid {pid} exited before it could be signaled")
            _set_state("")
            return "guard:term-failed"
        # 2026-10-30: skip a victim that is still warming up.  Every kill in
        # the 14:11-15:01 crash loop hit an engine started seconds earlier;
        # the guard kept SIGTERMing whatever it had just restarted.  On a
        # device_free-only breach a younger-than-grace victim gets WARN
        # treatment instead; if the breach persists past warm-up it kills
        # then.  Any other breach signal ignores the grace.
        age = _proc_age(int(pid), now)
        if (age is not None and age < GUARD_BOOT_GRACE_S
                and low_free and not nvrm_hit and not high_gpu
                and not low_avail and not low_psi and not high_swap):
            guard_log(f"boot grace: {name}(pid {pid}) is {int(age)}s old (< "
                      f"{GUARD_BOOT_GRACE_S:.0f}s) -- warming up, not killing "
                      f"(device_free {dfree}MiB, mem_available "
                      f"{availg if availg is None else round(availg, 1)}GiB)")
            _set_state("")
            return "guard:boot-grace"
        why = []
        if low_free:
            why.append(f"device_free {dfree}MiB < {int(DEVICE_FREE_FLOOR_MIB)}MiB")
        if low_avail:
            why.append(f"mem_available {availg:.1f}GiB < {MEM_FLOOR_GIB:.0f}GiB")
        if high_gpu:
            why.append(f"gpu {int(gpu)}MiB > {int(GPU_CEIL_MIB)}MiB")
        if low_psi:
            why.append(f"psi_full {psi_f}% >= {PSI_FULL_FLOOR}%")
        if high_swap:
            why.append(f"swap {swap_used:.1f}GiB >= {SWAP_FLOOR_GIB:.0f}GiB")
        if nvrm_hit:
            why.append(f"nvrm {nvrm} NV_ERR_NO_MEMORY since last sample")
        guard_log(f"LOW {'; '.join(why)} -> SIGTERM "
                  f"{name}(pid {pid}, {int(mib)}MiB)")
        if _signal_pid(pid, signal.SIGTERM, starttime):
            _set_state(f"termed:{pid}:{starttime}:{now}")
            return "guard:terming"
        guard_log(f"pid {pid} was not SIGTERM'ed: process identity changed "
                  "or pidfd signaling is unavailable")
        _set_state("")
        return "guard:term-failed"

    _set_state(f"armed:{now}")
    why = ("device_free " + str(dfree) + "MiB") if low_free else \
          (f"mem_available {availg:.1f}GiB" if low_avail else
           (f"gpu {int(gpu)}MiB" if high_gpu else
            (f"psi_full {psi_f}%" if low_psi else
             (f"nvrm {nvrm} since last sample" if nvrm_hit else
              f"swap {swap_used:.1f}GiB"))))
    guard_log(f"LOW {why} -> armed, needs a confirming sample within "
              f"{GUARD_CONSEC_S:.0f}s")
    return "guard:armed"


def _engine_log_file(pid):
    """The regular file this process writes its log to, or None.

    The engine's stderr is the file it and its launcher both append to:
    /proc/<engine>/fd/2 resolves to maya-maya-s-v2-iq2_xxs-256k.log, the
    same inode the "[maya] ..." progress lines land in, while fd 1 is a pipe
    to the server and fd 0 is /dev/null.  When several regular files are
    open, the most recently written *.log wins, so a wrapper's own capture
    is not mistaken for the engine's stream.  A pipe, socket or deleted
    target yields None -- the stall guard then stays out of the way rather
    than guessing.
    """
    best = None
    try:
        fds = os.listdir(f"/proc/{pid}/fd")
    except OSError:
        return None
    for fd in fds:
        try:
            target = os.readlink(f"/proc/{pid}/fd/{fd}")
        except OSError:
            continue
        if not target.startswith("/"):
            continue
        try:
            if not os.path.isfile(target):
                continue
            mtime = os.stat(target).st_mtime_ns
        except OSError:
            continue
        score = (target.endswith(".log"), mtime)
        if best is None or score > best[0]:
            best = (score, target)
    return best[1] if best else None


def _log_tail(path, lines):
    """The last `lines` non-empty lines of a file, NUL padding removed.

    A hard lock costs the engine its unflushed stdio buffer, and the kernel
    leaves the lost region as NULs inside the file (a 1806-byte run was
    measured at the 2026-10-25 lock), so reading the tail raw can return
    mostly zeros.
    """
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            if size > 262144:
                f.seek(size - 262144)
            else:
                f.seek(0)
            data = f.read()
    except OSError:
        return []
    out = []
    for raw in data.replace(b"\x00", b"").decode("utf-8", "replace").split("\n"):
        if raw.strip():
            out.append(raw)
    return out[-lines:] if lines > 0 else out


def _stall_snapshot(pid, path, util, temp, dfree, mem, why):
    """Write ~/gpu_reports/stall_<ts>.txt and return its path (or None)."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rp = os.path.join(STALL_SNAP_DIR, f"stall_{stamp}.txt")
    bar = "=" * 64
    avail = mem.get("MemAvailable") if mem else None
    try:
        with open(rp, "w") as f:
            f.write(f"{bar}\nDGX Spark engine stall -- "
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{bar}\n")
            f.write(f"{'why':<18}{why}\n")
            f.write(f"{'engine pid':<18}{pid}\n")
            f.write(f"{'engine log':<18}{path}\n")
            f.write(f"{'utilization':<18}{fmt(util, '%'):>16}\n")
            f.write(f"{'temperature':<18}{fmt(temp, '\N{DEGREE SIGN}C'):>16}\n")
            f.write(f"{'device free':<18}"
                    f"{('N/A' if dfree is None else f'{dfree}MiB'):>16}\n")
            f.write(f"{'mem available':<18}{kib(avail):>16}\n")
            f.write("-" * 30 + "\n")
            try:
                with open(f"/proc/{pid}/status") as pf:
                    for key in ("Name", "State", "VmRSS", "Threads"):
                        m = re.search(rf"^{key}:\s*(.+)$", pf.read(), re.M)
                        if m:
                            f.write(f"{key:<18}{m.group(1):>16}\n")
            except OSError:
                f.write("(no /proc status: the process is gone)\n")
            f.write("-" * 30 + f"\nlast {STALL_TAIL_LINES} engine log lines:\n")
            for line in _log_tail(path, STALL_TAIL_LINES):
                f.write(f"  {line}\n")
            f.write(f"{bar}\n")
    except OSError:
        return None
    return rp


def _stall_state():
    try:
        with open(STALL_STATE_PATH) as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""


def _set_stall_state(s):
    _atomic_write(STALL_STATE_PATH, s)


def _roll_engine_tail(path):
    """Keep a rolling copy of the engine's tail for post-mortem forensics."""
    if not path:
        return
    lines = _log_tail(path, STALL_TAIL_LINES)
    if not lines:
        return
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    body = f"# engine log tail ({path}) as of {stamp}\n" + "\n".join(lines) + "\n"
    try:
        _atomic_write(STALL_TAIL_PATH, body)
    except OSError:
        pass


def stall_guard(util, temp, procs, dfree, mem):
    """Kill an engine that stopped making progress while the GPU is pegged.

    Returns a short tag for the sample line, or None.  Separate state file
    from the memory guard, so the two never fight over one another's armed
    states; a stall kill also clears a stale memory-guard arm so it cannot
    immediately frame someone else for the engine's silence.
    """
    if not STALL_ENABLED:
        return None
    now = time.time()
    state = _stall_state().split(":")

    if state[0] == "stall-termed":
        try:
            pid, expected, termed_at = state[1], int(state[2]), float(state[3])
            valid = pid.isdigit() and int(pid) > 0 and expected >= 0 and termed_at >= 0
        except (IndexError, ValueError):
            valid = False
        if not valid:
            guard_log("discarding malformed stall state")
            _set_stall_state("")
            return None
        if _starttime(int(pid)) != expected:
            _set_stall_state("")          # the engine restarted on its own
            return None
        if not _alive(pid):
            guard_log(f"pid {pid} exited after the stall SIGTERM")
            _set_stall_state("")
            return None
        if now - termed_at >= STALL_GRACE_S:
            if _signal_pid(pid, signal.SIGKILL, expected):
                guard_log(f"pid {pid} still alive {STALL_GRACE_S:.0f}s after "
                          "the stall SIGTERM -> SIGKILL")
            else:
                guard_log(f"pid {pid} was not SIGKILLed after the stall "
                          "SIGTERM: process identity changed or pidfd "
                          "signaling is unavailable")
            _set_stall_state("")
            return "stall:killed"
        return "stall:draining"

    # Not mid-kill: find what to watch.  The top GPU consumer is the engine,
    # and its own stderr fd is the file it and its launcher append to.
    if not procs:
        return None
    pid, mib, name = max(procs, key=lambda p: p[1])
    path = STALL_LOG_PATH or _engine_log_file(int(pid))
    if path:
        # Kept on every sample, stall or not: after a hard lock it is the
        # only record of what the engine was doing.
        _roll_engine_tail(path)

    # An idle engine is silent by design -- this is the false-positive gate.
    if util is None or util < STALL_UTIL_PCT:
        if state[0]:
            _set_stall_state("")
        return None
    if _state().startswith("termed:"):
        return None                          # the memory guard is killing it
    if not path:
        return None                          # nothing watchable (pipe, gone)
    try:
        quiet = now - os.stat(path).st_mtime
    except OSError:
        return None
    if quiet < STALL_QUIET_S:
        if state[0]:
            _set_stall_state("")
        return None

    starttime = _starttime(int(pid))
    if starttime is None:
        return None
    why = (f"engine log silent {quiet:.0f}s (>= {STALL_QUIET_S:.0f}s) at "
           f"{util:.0f}% GPU, device_free "
           f"{'N/A' if dfree is None else str(dfree) + 'MiB'}")
    rp = _stall_snapshot(pid, path, util, temp, dfree, mem, why)
    guard_log(f"STALL {name}(pid {pid}): {why} -> SIGTERM"
              + (f"; snapshot {rp}" if rp else ""))
    if _signal_pid(pid, signal.SIGTERM, starttime):
        _set_stall_state(f"stall-termed:{pid}:{starttime}:{now}")
        armed = _state()
        if armed == "" or armed.startswith("armed:"):
            _set_state("")                   # do not let a stale arm frame another
        return "stall:terming"
    _set_stall_state("")
    return "stall:term-failed"


def main():
    # Whole-run exclusive lock (2026-10-25): if another invocation is
    # mid-sample, it owns this instant -- skip everything (no sample line,
    # no report, no guard decision) and exit 0 quietly.
    try:
        acquired = _try_lock()
    except OSError as exc:
        print(f"ERROR: cannot acquire GPU monitor guard lock: {exc}",
              file=sys.stderr)
        return 1
    if not acquired:
        return 0
    try:
        return _main_locked()
    finally:
        _unlock()


def _main_locked():
    util, temp, power = gpu_row()
    procs     = compute_apps()
    mem_total = sum(p[1] for p in procs) if procs else None
    mem       = meminfo()
    psi_s, psi_f = psi_memory()
    dfree, dtotal = device_free()
    nvrm_new  = nvrm_errors()
    guard     = memory_guard(mem, dfree, procs, psi_f, nvrm_new)
    stall     = stall_guard(util, temp, procs, dfree, mem)
    # Rebase the RSS baseline only on a healthy sample: mid-breach the
    # pre-breach snapshot is what makes growth attribution work on the
    # confirming sample, so it must survive.
    if _REBASE_OK and _LAST_RSS:
        _write_rss_base(_LAST_RSS)
    ts        = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stamp     = datetime.now().strftime("%Y%m%d_%H%M%S")

    deg = "\N{DEGREE SIGN}"
    top = sorted(procs, key=lambda p: -p[1])[:3]
    plist = " | ".join(f"{n}(pid {pid}, {int(m)}MiB)" for pid, m, n in top) \
        if top else "none"

    avail   = mem.get("MemAvailable")
    avail_s = f"{avail / 1048576:.1f}GiB" if avail is not None else "N/A"
    dfree_s = f"{dfree / 1024:.2f}GiB" if dfree is not None else "N/A"
    psi_s   = f"{fmt(psi_s, '%')}/{fmt(psi_f, '%')}"
    nvrm_s  = "N/A" if nvrm_new is None else str(nvrm_new)

    line = (f"[{ts}] GPU: NVIDIA GB10 | U:{fmt(util, '%')} | "
            f"T:{fmt(temp, deg + 'C')} | P:{fmt(power, 'W')} | "
            f"M:{fmt(mem_total, 'MiB')} | F:{dfree_s} | A:{avail_s} | "
            f"N:{nvrm_s} | proc: {plist}")
    if guard:
        line += f" | {guard}"
    if stall:
        line += f" | {stall}"
    line += "\n"

    with open(LOG_PATH, "a") as f:
        f.write(line)
    print(line.rstrip())

    os.makedirs(REPORT_DIR, exist_ok=True)
    rp = os.path.join(REPORT_DIR, f"gpu_report_{stamp}.txt")
    with open(rp, "w") as f:
        bar = "=" * 64
        f.write(f"{bar}\nDGX Spark GPU snapshot -- {ts}\n{bar}\n")
        f.write(f"{'Metric':<14}{'Value':>16}\n{'-' * 30}\n")
        f.write(f"{'Utilization':<14}{fmt(util, '%'):>16}\n")
        f.write(f"{'Temperature':<14}{fmt(temp, deg + 'C'):>16}\n")
        f.write(f"{'Power':<14}{fmt(power, 'W'):>16}\n")
        f.write(f"{'Mem used':<14}{fmt(mem_total, 'MiB'):>16}\n")
        f.write(f"{'DeviceFree':<14}{dfree_s:>16}\n")
        f.write(f"{'DeviceTotal':<14}{(f'{dtotal / 1024:.2f}GiB' if dtotal else 'N/A'):>16}\n")
        f.write(f"{'MemAvailable':<14}{avail_s:>16}\n")
        f.write(f"{'MemFree':<14}{kib(mem.get('MemFree')):>16}\n")
        f.write(f"{'PSI some/full':<14}{psi_s:>16}\n")
        f.write(f"{'NV_ERR_NO_MEMORY':<14}{nvrm_s:>16}\n")
        f.write("-" * 30 + "\n")
        for pid, m, n in sorted(procs, key=lambda p: -p[1]):
            f.write(f"{n[:20]:<20} pid {pid:<9} {int(m):>8} MiB\n")
        if not procs:
            f.write("(no GPU compute processes)\n")
        if guard:
            f.write(f"{'GUARD':<14}{guard:>16}\n")
        if stall:
            f.write(f"{'STALL':<14}{stall:>16}\n")
        f.write(f"{bar}\n")
    print(f"report: {rp}")

    rm = prune_old_reports()
    if rm:
        print(f"pruned {rm} report(s) older than {REPORT_KEEP_D}d")

    if util is None and mem_total is None:
        print("WARN: all GPU queries failed -- logged as N/A, exit 1",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
