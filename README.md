# gpu-monitor

A self-restarting GPU/memory guard for NVIDIA **DGX Spark (GB10)** -- a
unified-memory box where the GPU allocates from the same DRAM the CPU uses,
`nvidia-smi` answers `[N/A]` for memory queries, and a driver allocation
failure does not produce an error message: it deadlocks the driver and the
whole box has to be power-cycled.

`gpu_monitor.py` samples the box every ~10 s (one invocation = one sample,
driven by a systemd restart loop), and when it sees the signature of those
failures it kills the offending process before the box dies.

> Technique credited to MiaAI Lab's `memwatch.sh` (AGPL-3.0-or-later,
> github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark) for the kernel-log
> cursor idea. Reimplemented from scratch; no code taken.

## What it watches

| Signal | Source | Why this one |
|---|---|---|
| util / temp / power | `nvidia-smi --query-gpu` | fields proven to work on GB10 |
| per-process GPU memory | `nvidia-smi --query-compute-apps` | GB10's `memory.used` query returns `[N/A]` -- never trust it |
| **device-free** | `cudaMemGetInfo` via ctypes | **the pool the driver allocates from** -- on GB10 that is system DRAM |
| MemAvailable / SwapUsed | `/proc/meminfo` | OS-side exhaustion; `MemFree` is pinned by the kernel and useless as a trigger |
| PSI memory some/full | `/proc/pressure/memory` | "full" avg10 >= 5% is allocation already stalling |
| NV_ERR_NO_MEMORY | `journalctl -k` with an epoch cursor | literal evidence the driver is refusing allocations |

## The guards

### Memory guard

Two consecutive breaches within 30 s (so transient dips never kill a running
server) on any of:

1. device-free below floor (default 3072 MiB) -- the pool that actually ran
   out,
2. MemAvailable below floor (default 6 GiB),
3. GPU compute-app total above ceiling (default 112640 MiB),
4. PSI memory full avg10 >= 5%,
5. swap used >= 8 GiB,
6. new NV_ERR_NO_MEMORY lines in the kernel log,

-> SIGTERM the victim, escalating to SIGKILL after a 30 s grace.

Victim selection by breach family: GPU-pool breaches pick the **top GPU
consumer** -- unless RSS-growth attribution says a plain RSS hog (holding no
GPU memory at all) drove the breach, since on unified memory device RAM *is*
host RAM.  OS-side breaches pick the **biggest-RSS process**.  Signaling goes
through pidfd with a start-time identity check, so a reused pid is never
signaled and a stale kill state never drains forever.

Two refinements the field tests bought:

- **The tolerated dip** -- a device-free-only breach with healthy host RAM
  (>= 24 GiB) is downgraded to WARN: on GB10 a transient prefill borrow can
  read below the floor while everything is fine.  A breach that persists
  past the boot grace (240 s, so a freshly restarted engine is never shot
  for warming up) kills then.
- **The concurrent-writer age check** -- a freshly-written `armed` state
  younger than 8 s belongs to a simultaneous invocation and is never
  confirmed on; without it, two invocations collapse the two-sample rule
  into a one-sample kill.

### Stall guard

The memory guard cannot see the failure the box actually hard-locked from:
the engine stops making progress while the GPU stays pegged (>= 90%),
device-free stays healthy, the driver refuses nothing, and the kernel log is
silent.  This guard arms only when the engine's own log file (its stderr fd,
auto-detected via `/proc/<pid>/fd/*`) has been silent for 90 s **while GPU
util is at/above the threshold** -- the utilization gate is the
false-positive gate against an idle engine.

On a stall it snapshots the engine's `/proc` status and its last 40 log lines
(NUL padding stripped -- a hard lock loses the engine's unflushed buffer as
NULs), SIGTERMs the engine, escalates to SIGKILL after 30 s, and keeps a
rolling copy of the engine tail in `~/gpu_engine_tail.txt` on **every**
sample, stall or not -- after a hard lock that is the only record of what
the model was doing.

## Files it owns

| Path | Purpose |
|---|---|
| `~/gpu_monitor_log.txt` | one compact line per sample |
| `~/gpu_reports/gpu_report_<ts>.txt` | full per-sample snapshot (pruned after 7 days) |
| `~/gpu_reports/stall_<ts>.txt` | stall post-mortems |
| `~/gpu_engine_tail.txt` | rolling engine-log tail (forensics) |
| `~/gpu_monitor_guard.log` | every guard decision, with timestamps and reasons |

State files (`.gpu_monitor_*` in `$HOME`) carry the two-sample rule across
samples and invocations; all writes are atomic (`os.replace` + `fsync`), so
a power cut mid-write cannot leave a half-written armed state.

## Install (systemd)

```ini
# /etc/systemd/system/gpu_monitor.service
[Unit]
Description=DGX Spark GPU/memory guard
After=multi-user.target

[Service]
Type=oneshot
User=YOUR_USER
ExecStart=/usr/bin/python3 /home/YOUR_USER/gpu_monitor.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Do **not** use `Restart=on-failure` with this script: it exits 0 on every
healthy sample, so a restart-on-failure unit becomes a hot self-restarting
loop the moment one sample goes bad.  `Restart=always` + `RestartSec=10` is
the intended driver of the ~10 s cadence.  (The companion
`gpu_monitor.timer` is unnecessary -- its 30 min period never fires because
the service never stays inactive that long.)

Then:

```sh
sudo cp gpu_monitor.py /home/YOUR_USER/
sudo systemctl daemon-reload && sudo systemctl enable --now gpu_monitor.service
```

## Reading a sample line

```
[2026-11-07 06:55:02] GPU: NVIDIA GB10 | U:95% | T:74°C | P:46.68W |
M:76294MiB | F:5.91GiB | A:31.2GiB | N:0 | proc: build/maya(pid 1583451,
76124MiB)
```

`F` (device-free via cudaMemGetInfo) is the number the guard acts on;
`A` (MemAvailable) is the OS's -- and on unified memory the two can diverge
wildly, which is the whole point.  Trailing `guard:` / `stall:` tags carry
that sample's guard decisions; no tags means a healthy sample.

## Tuning

Every floor is a `GPU_MONITOR_*` environment variable -- see the config
block at the top of the script for the full list, including
`GPU_MONITOR_GUARD=0` / `GPU_MONITOR_STALL_GUARD=0` to disarm either guard
without editing the file.

Porting notes:

- **Another model**: the guard is model-agnostic -- it watches the top GPU
  consumer and a log file's mtime; it never parses model-specific content.
  Re-tune the floors for the new workload's footprint.
- **Another architecture**: the code runs on any NVIDIA box, but the memory
  guard's *semantics* are unified-memory.  On a discrete-GPU box,
  device-free and MemAvailable describe separate pools: remove the
  tolerated-dip co-floor (it would wrongly suppress real VRAM breaches) and
  reconsider the RSS-growth attribution premise.  The stall guard ports
  as-is to any GPU.  Non-aarch64/non-x86_64 machines need the two pidfd
  syscall numbers added (or a kernel/glibc with `os.pidfd_open`).
- **Another box**: re-measure every floor; a slower nvidia-smi needs a
  longer `RestartSec` so one sample never overlaps the next.

## License

AGPL-3.0-or-later -- see [LICENSE](LICENSE).
