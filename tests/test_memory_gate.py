"""The absolute start gate, the machine-wide budget and the pause on absolute readings.

2026-10-24: the Mac hung and rebooted. A GPU video render (ram), a 12-worker CPU job (a private lane), a
1.4 GB build and Blender jobs (light) ran at once; the compressor reached some 15 GB, 0.1 GB was
free. The guard saw none of it: the render's footprint read 6.43 GB (Metal is missed), the swap
budget counted from 8.3 GB at the start, and the level read 40 %.
"""

import json
import os
import re
import sys
import threading
import time

import pytest

from stageload import queue
from stageload.cli import _parser, main
from stageload.guard import EXIT_NO_ROOM, GuardConfig, GuardIO, run_guarded
from stageload.meter import Readers
from stageload.queue import (
    DEFAULT_PEAK,
    HISTORY_FACTOR,
    list_queue,
    predict_peak,
    run_queued,
)

GB = 2**30
GATE_ON = {"STAGELOAD_GATE_SWAP_GB": "6", "STAGELOAD_GATE_COMPRESSOR_GB": "8",
           "STAGELOAD_GATE_LEVEL": "35", "STAGELOAD_PAUSE_SWAP_GB": "9",
           "STAGELOAD_PAUSE_COMPRESSOR_GB": "12", "STAGELOAD_PAUSE_LEVEL": "25",
           "STAGELOAD_RESERVE_GB": "12"}
TRUE = [sys.executable, "-c", "pass"]
# runs for 1.5 s of its own time: it ends only if a pause is resumed
WORK = [sys.executable, "-c", "import time\nfor _ in range(30): time.sleep(0.05)"]
# the line `stageload queue` prints, as the command center at :4420 reads it
CENTER = re.compile(
    r"^(\S+)\s+(\d+)\s+(running|waiting|gpu-wait|mem-wait|paused)\s+(\S+)(?:\s+\[gpu\])?\s+"
    r"pid (\d+)\s+since (\S+)"
)


@pytest.fixture
def gate(monkeypatch):
    for name, value in GATE_ON.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(queue, "_ram", lambda: 48 * GB)


def _after(seconds):
    began = time.monotonic()
    return lambda: time.monotonic() - began >= seconds


class Readings:
    """Fake machine readings that change with time; ``seen`` keeps the queue rows read meanwhile."""

    def __init__(self, home=None, swap=0, compressor=2 * GB, level=70, footprint=None):
        self.home = home
        self.swap, self.compressor, self.level = swap, compressor, level
        self.footprint = footprint
        self.logs, self.seen = [], []
        self.phase = None

    def read_swap(self):
        if self.phase is not None:
            self.phase(self)
        if self.home is not None:
            self.seen.append(list_queue(self.home))
        return self.swap

    def io(self):
        return GuardIO(
            level=lambda: self.level, swap_used=self.read_swap, processes=lambda: [],
            log=self.logs.append, compressor=lambda: self.compressor, wired=lambda: 9 * GB,
            footprint=None if self.footprint is None else (lambda pid: self.footprint),
            neighbours=lambda: [(4242, int(5.1 * GB), "python3 sim.py --workers 12")],
        )


def _running_ticket(home, lane, who, mem_peak=None, cmd="python3 job.py"):
    """A running job of another session: its run is this test process, so it stays alive."""
    lane_dir = home / "queue" / lane
    lane_dir.mkdir(parents=True, exist_ok=True)
    ticket = lane_dir / f"{time.time_ns():020d}-{os.getpid()}.json"
    data = {"pid": os.getpid(), "who": who, "lane": lane, "cmd": cmd, "state": "running",
            "started": "2026-10-24T00:30:00+00:00"}
    if mem_peak is not None:
        data["mem_peak"] = mem_peak
    ticket.write_text(json.dumps(data))
    return ticket


def _quick(**fields):
    return GuardConfig(poll=0.05, grace=1.0, **fields)


# --- the start gate ------------------------------------------------------------------------


def test_the_gate_waits_while_swap_is_high_and_starts_once_it_falls(tmp_path, gate, monkeypatch):
    _running_ticket(tmp_path, "light", "blender-grey", mem_peak=0)  # not idle: swap counts
    # over the level gate (35 %) and under the stale-swap level of 50 %: from there on, swap with the compressor
    # under its gate is stale and holds nothing (the machine's 35 % is set in guard.py)
    monkeypatch.setenv("STAGELOAD_GATE_STALE_SWAP_LEVEL", "50")
    r = Readings(home=tmp_path, swap=int(8.3 * GB), level=40)
    falls = _after(0.4)
    r.phase = lambda m: setattr(m, "swap", 2 * GB if falls() else int(8.3 * GB))
    began = time.monotonic()
    assert run_queued(TRUE, _quick(), io=r.io(), home=tmp_path, who="box-s1-c003",
                      trace=None) == 0
    assert time.monotonic() - began >= 0.4
    wait = next(line for line in r.logs if "waiting for memory" in line)
    assert "swap 8.3 GB > 6.0 GB" in wait
    waiting = [row for rows in r.seen for row in rows
               if row["who"] == "box-s1-c003" and row["state"] == "mem-wait"]
    assert waiting and "swap 8.3 GB > 6.0 GB" in waiting[0]["mem_wait"]


def test_the_gate_waits_for_the_compressor_and_the_level(tmp_path, gate):
    r = Readings(home=tmp_path, compressor=15 * GB, level=30)
    calm = _after(0.4)

    def phase(m):
        if calm():
            m.compressor, m.level = 3 * GB, 60

    r.phase = phase
    assert run_queued(TRUE, _quick(wait_free=0), io=r.io(), home=tmp_path, who="w08-c014",
                      trace=None) == 0
    wait = next(line for line in r.logs if "waiting for memory" in line)
    assert "compressor 15.0 GB > 8.0 GB" in wait and "memory available 30% < 35%" in wait


def test_the_gate_gives_up_after_the_start_timeout(tmp_path, gate):
    _running_ticket(tmp_path, "light", "blender-grey", mem_peak=0)  # not idle: swap counts
    r = Readings(home=tmp_path, swap=10 * GB, level=34)
    assert run_queued(TRUE, _quick(start_timeout=0.3), io=r.io(), home=tmp_path, who="x",
                      trace=None) == EXIT_NO_ROOM
    assert any("no memory after" in line for line in r.logs)
    assert [row["who"] for row in list_queue(tmp_path)] == ["blender-grey"]


def test_the_queue_line_of_a_run_waiting_for_memory(tmp_path, monkeypatch, capsys):
    lane = tmp_path / "queue" / "ram"
    lane.mkdir(parents=True)
    ticket = {"pid": os.getpid(), "who": "box-s1-c003", "owner": "box", "lane": "ram",
              "cmd": "python3 h3_shot.py", "state": "mem-wait", "gpu": True, "gpu_turn": True,
              "turn": "2026-10-24T03:00:00+00:00", "mem_peak": 25 * GB,
              "mem_wait": "swap 8.3 GB > 6.0 GB"}
    (lane / f"{1:020d}-{os.getpid()}.json").write_text(json.dumps(ticket))
    monkeypatch.setenv("STAGELOAD_HOME", str(tmp_path))
    assert main(["queue"]) == 0
    line = capsys.readouterr().out.splitlines()[0]
    assert line.startswith("ram     0 mem-wait box-s1-c003 [gpu]    pid ")
    assert line.endswith("since 2026-10-24T03:00:00+00:00  memory: swap 8.3 GB > 6.0 GB  "
                         "python3 h3_shot.py")
    found = CENTER.match(line)
    assert found and found.group(3) == "mem-wait" and found.group(4) == "box-s1-c003"


# --- the budget ----------------------------------------------------------------------------


def test_the_budget_holds_a_second_big_job_and_lets_a_small_one_through(tmp_path, gate):
    render = _running_ticket(tmp_path, "ram", "box-s1-c003", mem_peak=25 * GB)
    # 25 + 4 + 12 = 41 GB of 48: the small job starts at once, in another lane
    small = Readings(home=tmp_path)
    assert run_queued(TRUE, _quick(start_timeout=5), io=small.io(), home=tmp_path, lane="light",
                      who="wanderer-camp-build", trace=None, mem_peak=4 * GB) == 0
    assert not any("waiting for memory" in line for line in small.logs)
    # 25 + 25 + 12 = 62 GB of 48: the second render waits until the first is gone
    big = Readings(home=tmp_path)
    gone = threading.Timer(0.4, render.unlink)
    gone.start()
    began = time.monotonic()
    try:
        assert run_queued(TRUE, _quick(start_timeout=5), io=big.io(), home=tmp_path, lane="h3b",
                          who="box-s1-c004", trace=None, mem_peak=25 * GB) == 0
    finally:
        gone.cancel()
    assert time.monotonic() - began >= 0.4
    wait = next(line for line in big.logs if "waiting for memory" in line)
    assert ("budget: running 25.0 GB (box-s1-c003 (ram) 25 GB) + ours 25.0 GB + reserve "
            "12.0 GB > RAM 48.0 GB") in wait


def test_a_job_of_a_private_lane_counts_even_without_a_written_peak(tmp_path, gate):
    # a run of the stageload before this one wrote no peak: it is predicted from its history
    (tmp_path / "history.jsonl").write_text(json.dumps(
        {"lane": "now-cpu", "who": "now-cpu-sim", "peak_footprint": int(5.1 * GB),
         "end": "2026-10-23T00:00:00+00:00", "run_s": 66600}) + "\n")
    _running_ticket(tmp_path, "now-cpu", "now-cpu-sim", cmd="python3 sim.py --workers 12")
    r = Readings(home=tmp_path)
    # 6.6 + 30 + 12 = 48.6 GB of 48
    assert run_queued(TRUE, _quick(start_timeout=0.3), io=r.io(), home=tmp_path,
                      who="big", trace=None, mem_peak=30 * GB) == EXIT_NO_ROOM
    wait = next(line for line in r.logs if "waiting for memory" in line)
    assert "now-cpu-sim (now-cpu) 7 GB" in wait


def test_a_job_bigger_than_the_budget_starts_alone(tmp_path, gate):
    r = Readings(home=tmp_path)
    assert run_queued(TRUE, _quick(start_timeout=5), io=r.io(), home=tmp_path, who="ltx",
                      trace=None, mem_peak=43 * GB) == 0
    assert any("it starts alone" in line for line in r.logs)
    _running_ticket(tmp_path, "light", "blender-grey", mem_peak=4 * GB)
    r = Readings(home=tmp_path)
    assert run_queued(TRUE, _quick(start_timeout=0.3), io=r.io(), home=tmp_path, who="ltx",
                      trace=None, mem_peak=43 * GB) == EXIT_NO_ROOM


def test_the_predicted_peak_is_written_into_the_ticket(tmp_path, gate):
    r = Readings(home=tmp_path)
    h3 = ["nice", "-n", "15", sys.executable, "-c", "pass",
          "/Users/me/local-llm/h3-forks/lachlanstuart/h3", "--profile"]
    assert run_queued(h3, _quick(), io=r.io(), home=tmp_path, who="box-s1-c003",
                      trace=None) == 0
    rows = [rows[0] for rows in r.seen if rows]
    assert rows and all(row["mem_peak"] == 25 * GB for row in rows)


# --- the predicted peak --------------------------------------------------------------------


def test_the_peak_comes_from_the_flag_then_the_engine_then_the_history_then_the_default():
    peaks = {"now-cpu-sim": int(5.1 * GB), "wide-w08-c025b": 2 * GB, "wide-w08-c036": 3 * GB}
    assert predict_peak("anything", "x", 10 * GB, peaks) == (10 * GB, "--mem-peak")
    for cmd, gb in (
        ("nice -n 15 /Users/me/local-llm/h3-forks/lachlanstuart/h3 --profile", 25),
        ("zsh /Users/me/.claude/skills/h3-video/scripts/h3_exec_convrot.sh -d w", 25),
        ("/Users/me/ltx/.venv/bin/ltx-2-mlx --distilled", 43),
        ("/Applications/Blender.app/Contents/MacOS/Blender -b set.blend -P shot.py", 4),
        ("realcugan-ncnn-vulkan -i a -o b", 3),
        ("rife-ncnn-vulkan -i a", 3),
        ("/Users/me/local-llm/mmaudio-venv/bin/python foley_mac.py", 8),
        ("python -c import mlx_whisper", 3),
    ):
        assert predict_peak(cmd, "x", None, peaks)[0] == gb * GB, cmd
    peak, source = predict_peak("python3 sim.py", "now-cpu-sim", None, peaks)
    assert peak == int(int(5.1 * GB) * HISTORY_FACTOR) and source.startswith("history")
    # a new shot of a known series: the largest of its --who prefix
    assert predict_peak("python3 x.py", "wide-w08-c040", None, peaks)[0] == int(
        3 * GB * HISTORY_FACTOR)
    assert predict_peak("python3 x.py", "new-job", None, peaks, lane="light") == (
        DEFAULT_PEAK, "default")


def test_the_history_peak_is_read_from_history_jsonl(tmp_path, gate):
    lines = [{"who": "now-cpu-sim", "peak_footprint": int(5.1 * GB)},
             {"who": "now-cpu-sim", "peak_footprint": 3 * GB}, {"who": "other", "exit": 1}]
    (tmp_path / "history.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    r = Readings(home=tmp_path)
    assert run_queued(TRUE, _quick(), io=r.io(), home=tmp_path, lane="now-cpu",
                      who="now-cpu-sim", trace=None) == 0
    row = next(rows[0] for rows in r.seen if rows)
    assert row["mem_peak"] == int(int(5.1 * GB) * HISTORY_FACTOR)


def test_the_mem_peak_flag():
    args = _parser().parse_args(["run", "--mem-peak", "25G", "--", "true"])
    assert args.mem_peak == 25 * GB
    assert _parser().parse_args(["run", "--", "true"]).mem_peak is None


# --- the pause on absolute readings --------------------------------------------------------


def test_absolute_swap_pauses_a_flat_command_and_resumes_it_under_the_gate(gate):
    # swap 5 GB at the start, budget 8 GB: the budget alone would wait for 13 GB
    r = Readings(swap=5 * GB, footprint=17 * GB)
    rises, falls = _after(0.4), _after(1.2)
    def phase(m):
        # 29.10: a pause ends at the pause floor (25 %), so the level stays under it meanwhile
        m.swap, m.level = (10 * GB, 20) if rises() and not falls() else (5 * GB, 70)

    r.phase = phase
    began = time.monotonic()
    assert run_guarded(WORK, _quick(pause_max=30), r.io()) == 0
    assert time.monotonic() - began >= 1.5 + 0.5
    pause = next(line for line in r.logs if "pausing pid" in line)
    assert "swap 10.0 GB > 9.0 GB machine-wide" in pause
    # 29.10: a pause ends at the pause floor (25 %), not at the start gate's 35 %
    assert "Resuming once memory available is 25% or more" in pause
    assert any("resuming pid" in line and "memory available 70% >= 25%" in line
               for line in r.logs)
    assert not any("killing pid" in line for line in r.logs)


def test_the_compressor_pauses_a_flat_command(gate):
    r = Readings(swap=0, footprint=17 * GB)
    rises, falls = _after(0.4), _after(1.2)
    r.phase = lambda m: setattr(m, "compressor", 15 * GB if rises() and not falls() else 2 * GB)
    assert run_guarded(WORK, _quick(pause_max=30), r.io()) == 0
    assert any("pausing pid" in line and "compressor 15.0 GB > 12.0 GB" in line
               for line in r.logs)
    assert any("resuming pid" in line for line in r.logs)


def test_a_pause_that_does_not_clear_stops_the_command_after_the_pause_max(gate):
    r = Readings(swap=5 * GB, footprint=17 * GB)
    rises = _after(0.3)
    def phase(m):
        m.swap, m.level = (10 * GB, 20) if rises() else (7 * GB, 70)

    r.phase = phase
    slow = [sys.executable, "-c", "import time; time.sleep(60)"]
    assert run_guarded(slow, _quick(pause_max=0.5), r.io()) == 3
    kill = next(line for line in r.logs if "killing pid" in line)
    assert "still memory available 20% < 25%" in kill


def test_a_command_that_grows_under_absolute_pressure_is_not_paused(gate):
    r = Readings(swap=5 * GB, footprint=10 * GB)
    rises = _after(0.3)

    def phase(m):
        # 2 GB a sample: more than a quarter of the swap growth, so the growth is its own
        if rises():
            m.swap, m.footprint = 10 * GB, m.footprint + 2 * GB

    r.phase = phase
    assert run_guarded(WORK, _quick(pause_max=30), r.io()) == 0
    assert not any("pausing" in line or "killing" in line for line in r.logs)


# --- the trace -----------------------------------------------------------------------------


def test_the_trace_records_the_compressor_and_wired_memory(tmp_path, gate):
    r = Readings(home=tmp_path, compressor=int(4.5 * GB))
    trace = tmp_path / "t.jsonl"
    assert run_queued(TRUE, _quick(), io=r.io(), home=tmp_path, who="x", trace=trace) == 0
    samples = [json.loads(line) for line in trace.read_text().splitlines()]
    samples = [s for s in samples if s["type"] == "sample"]
    assert samples and all(s["compressor_gb"] == 4.5 and s["wired_gb"] == 9.0 for s in samples)
    assert "footprint" in samples[0] and "swap_used" in samples[0]


def test_a_failing_compressor_reader_leaves_the_rest_of_the_sample():
    from stageload.meter import MemoryMeter

    def broken():
        raise OSError("host_statistics64 failed")

    readers = Readers(footprint=lambda: 1, swap_used=lambda: 2, level=lambda: 3,
                      compressor=broken)
    meter = MemoryMeter("/dev/null", readers=readers)
    sample = meter.sample()
    assert sample["footprint"] == 1 and "compressor_gb" not in sample


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS memory counters")
def test_the_compressor_and_wired_readers_read_the_machine():
    from stageload.meter import compressor_used, machine_ram, wired_used

    ram = machine_ram()
    assert ram > 4 * GB
    assert 0 <= compressor_used() < ram and 0 < wired_used() < ram
