"""Pressure the queue's own jobs make is blamed on the job that rose, and a pause resumes on swap
and the level, not on the compressor.

2026-10-24, 03:06-03:21Z: w08-post-audio (ram, ``zsh audio/run_ram_audio.sh`` with MMAudio
inside, real peak 23.4 GB) and code-isekai-omb-s1-4k-build (light, real peak 7.7 GB) ran side by
side. The budget took both at the 4 GB default. At 03:11 the compressor reached 14.5 GB; the audio
job's guard saw its own footprint flat (+0.2 GB in 61 s), called the pressure outside and paused
its command, although every process outside the queue held under 0.5 GB and the build beside it
had risen 1 GB. Paused, the command's idle pages were compressed: the compressor rose to 21.9 GB,
the resume condition (compressor under 8 GB) could never be met, and after 600 s both jobs were
killed with exit 3.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

import pytest

from stageload import queue
from stageload.guard import (
    COMPRESSOR_REARM,
    Growth,
    GuardConfig,
    GuardIO,
    queue_culprit,
    run_guarded,
    window_rise,
)
from stageload.queue import (
    DEFAULT_PEAK,
    HISTORY_FACTOR,
    RAM_LANE_PEAK,
    budget_reason,
    predict_peak,
    run_queued,
    running_jobs,
    trace_stats,
)

GB = 2**30
MB = 2**20
GATE = {"STAGELOAD_GATE_SWAP_GB": "6", "STAGELOAD_GATE_COMPRESSOR_GB": "8",
        "STAGELOAD_GATE_LEVEL": "35", "STAGELOAD_PAUSE_SWAP_GB": "9",
        "STAGELOAD_PAUSE_COMPRESSOR_GB": "12", "STAGELOAD_PAUSE_LEVEL": "25"}
# runs for 1.5 s of its own time: it ends only if a pause is resumed
WORK = [sys.executable, "-c", "import time\nfor _ in range(30): time.sleep(0.05)"]
TRUE = [sys.executable, "-c", "pass"]
STARTED_AUDIO = datetime(2026, 10, 24, 3, 6, 53, tzinfo=timezone.utc).timestamp()
STARTED_BUILD = datetime(2026, 10, 24, 3, 6, 47, tzinfo=timezone.utc).timestamp()
AUDIO_PEAK = 25079861472  # history.jsonl, w08-post-audio: «peak 23.4 GB»
BUILD_PEAK = 8306928056  # history.jsonl, code-isekai-omb-s1-4k-build: 7.7 GB


@pytest.fixture
def gate(monkeypatch):
    for name, value in GATE.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("STAGELOAD_RESERVE_GB", raising=False)
    monkeypatch.setattr(queue, "_ram", lambda: 48 * GB)


def audio(me=False, rise=200 * MB):
    return Growth("w08-post-audio", grown=23 * GB, recent=rise, me=me, rise=rise,
                  started=STARTED_AUDIO)


def build(me=False, rise=GB):
    return Growth("code-isekai-omb-s1-4k-build", grown=5 * GB, recent=0, me=me, rise=rise,
                  started=STARTED_BUILD)


class Machine:
    """The readings of 03:11Z: swap empty, the level 64 %, the compressor jumping to 14.5 GB and
    growing to 21.9 GB while a command is paused. Each guard's footprint stays flat."""

    def __init__(self, jobs, footprint=23 * GB, compressor=int(14.5 * GB)):
        self.compressor = compressor
        self.paused_compressor = int(21.9 * GB)
        self.footprint = footprint
        self.logs = []
        self.jobs = jobs

    def paused(self):
        for line in reversed(self.logs):
            if "resuming pid" in line:
                return False
            if "pausing pid" in line:
                return True
        return False

    def read_compressor(self):
        return self.paused_compressor if self.paused() else self.compressor

    def io(self):
        return GuardIO(
            level=lambda: 64, swap_used=lambda: 170 * MB, processes=lambda: [],
            log=self.logs.append, compressor=self.read_compressor, wired=lambda: 4 * GB,
            footprint=lambda pid: self.footprint, jobs=self.jobs,
            neighbours=lambda: [(2760, int(0.4 * GB), "claude-code/2.1.295")],
        )


def _quick(**fields):
    return GuardConfig(poll=0.05, grace=1.0, **fields)


def _lines(logs, text):
    return [line for line in logs if text in line]


# --- the decision --------------------------------------------------------------------------


def test_the_rise_is_from_a_trough_to_a_later_peak():
    assert window_rise([6, 4, 5, 3, 5]) == 2
    assert window_rise([5, 4, 3]) == 0
    assert window_rise([]) == 0


def test_the_audio_job_blames_the_build_beside_it_and_the_build_blames_itself():
    slack = 512 * MB
    verdict, target, growing = queue_culprit(0, [audio(me=True), build()], slack)
    assert verdict == "theirs" and target.who == "code-isekai-omb-s1-4k-build"
    assert [job.who for job in growing] == ["code-isekai-omb-s1-4k-build"]
    verdict, target, _ = queue_culprit(0, [audio(), build(me=True)], slack)
    assert verdict == "ours" and target.me


def test_no_queue_job_rose_is_outside_and_a_job_alone_acts_as_before():
    slack = 512 * MB
    assert queue_culprit(0, [audio(me=True), build(rise=100 * MB)], slack)[0] == "outside"
    assert queue_culprit(0, [audio(me=True)], slack)[0] == "alone"
    assert queue_culprit(0, None, slack)[0] == "alone"


def test_of_two_jobs_that_rose_the_one_started_last_is_the_target():
    later = Growth("box-s1-c004", grown=2 * GB, recent=GB, rise=GB, started=STARTED_AUDIO + 60)
    verdict, target, growing = queue_culprit(0, [audio(me=True, rise=3 * GB), later],
                                             512 * MB)
    assert verdict == "theirs" and target is later and len(growing) == 2
    # our own live samples count when our trace says less
    verdict, _, _ = queue_culprit(2 * GB, [audio(me=True, rise=0), build(rise=0)], 512 * MB)
    assert verdict == "ours"


# --- the guard -----------------------------------------------------------------------------


def test_the_audio_job_runs_on_while_the_build_beside_it_rose(gate):
    m = Machine(lambda: [audio(me=True), build()])
    assert run_guarded(WORK, _quick(pause_max=30), m.io()) == 0
    assert not _lines(m.logs, "pausing") and not _lines(m.logs, "killing")
    blamed = _lines(m.logs, "not ours to pause or stop")
    assert len(blamed) == 1  # the compressor reading is dealt with: it is not logged every poll
    assert "compressor 14.5 GB > 12.0 GB machine-wide" in blamed[0]
    assert "code-isekai-omb-s1-4k-build +1.0 GB" in blamed[0]
    assert "outside" not in blamed[0]


def test_the_build_pauses_itself_and_resumes_although_its_pages_fill_the_compressor(gate):
    m = Machine(lambda: [audio(), build(me=True)], footprint=5 * GB)
    began = time.monotonic()
    assert run_guarded(WORK, _quick(pause_max=30), m.io()) == 0
    assert time.monotonic() - began >= 1.5
    pause = _lines(m.logs, "pausing pid")
    assert len(pause) == 1, m.logs  # resumed with 21.9 GB compressed, not paused again at once
    assert "ours started last: the pressure is ours" in pause[0]
    assert "Resuming once memory available is 25% or more" in pause[0]
    resume = _lines(m.logs, "resuming pid")
    assert resume and "memory available 64% >= 25%" in resume[0]  # 29.10: the pause floor
    assert not _lines(m.logs, "killing")


def test_with_no_queue_job_risen_the_pause_is_for_outside_pressure(gate):
    m = Machine(lambda: [audio(me=True, rise=0), build(rise=100 * MB)])
    assert run_guarded(WORK, _quick(pause_max=30), m.io()) == 0
    pause = _lines(m.logs, "pausing pid")
    assert len(pause) == 1
    assert "no queue job rose more than 0.5 GB" in pause[0]
    assert "the pressure is from outside the queue" in pause[0] and "claude-code" in pause[0]
    assert _lines(m.logs, "resuming pid") and not _lines(m.logs, "killing")


def test_the_incident_alone_resumes_instead_of_being_killed_after_the_pause_max(gate):
    # the guard of 03:11Z without the other job: paused for outside pressure, as e44f1e3 does;
    # the compressor grows to 21.9 GB meanwhile and no longer keeps it paused
    m = Machine(None)
    assert run_guarded(WORK, _quick(pause_max=0.6), m.io()) == 0
    assert len(_lines(m.logs, "pausing pid")) == 1
    assert _lines(m.logs, "resuming pid") and not _lines(m.logs, "killing")


def test_a_compressor_that_grows_again_after_a_resume_counts_again(gate):
    m = Machine(None)
    began = time.monotonic()

    def compressor():
        # 2 GB more than the reading at the resume, paused or not
        if _lines(m.logs, "resuming pid") and time.monotonic() - began > 0.6:
            return m.paused_compressor + COMPRESSOR_REARM + GB
        return m.read_compressor()

    io = m.io()
    io.compressor = compressor
    assert run_guarded(WORK, _quick(pause_max=30), io) == 0
    assert len(_lines(m.logs, "pausing pid")) == 2


def test_a_level_that_stays_low_still_stops_the_command_after_the_pause_max(gate):
    m = Machine(None)
    io = m.io()
    readings = iter([64, 64])
    io.level = lambda: next(readings, 20)  # room to start, then 20 %
    slow = [sys.executable, "-c", "import time; time.sleep(60)"]
    assert run_guarded(slow, _quick(pause_max=0.5), io) == 3
    kill = _lines(m.logs, "killing pid")[0]
    assert "still memory available 20% < 25%" in kill and "compressor" not in kill


# --- the other jobs, read from the queue ---------------------------------------------------


def _trace(path, footprints, interval=0.5):
    began = time.time() - interval * (len(footprints) - 1)
    utc = datetime.fromtimestamp(began, timezone.utc).isoformat(timespec="seconds")
    lines = [{"type": "start", "utc": utc, "interval": interval}]
    lines += [{"type": "sample", "t": i * interval, "footprint": fp, "swap_used": 0, "level": 50}
              for i, fp in enumerate(footprints)]
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    return path


def test_the_trace_gives_the_rise_of_the_last_minute(tmp_path):
    # the build of 03:10-03:11Z: 6.2 GB, down to 3.95 GB, back up to 4.97 GB
    fps = [int(6.2 * GB)] * 40 + [int(3.95 * GB)] * 40 + [int(4.97 * GB)] * 40
    grown, recent, rise = trace_stats(_trace(tmp_path / "t.jsonl", fps))
    assert rise == int(4.97 * GB) - int(3.95 * GB)
    assert recent == 0


def test_running_jobs_carry_when_they_started_and_whether_they_are_paused(tmp_path):
    traces = tmp_path / "traces"
    traces.mkdir()
    lane_dir = tmp_path / "queue" / "light"
    lane_dir.mkdir(parents=True)
    trace = _trace(traces / "b.jsonl", [3 * GB, 4 * GB])
    ticket = lane_dir / f"{1:020d}-{os.getpid()}.json"
    ticket.write_text(json.dumps({"pid": os.getpid(), "who": "code-isekai-omb-s1-4k-build",
                                  "state": "paused", "started": "2026-10-24T03:06:47+00:00",
                                  "trace": str(trace)}))
    (job,) = running_jobs(tmp_path)()
    assert job.started == STARTED_BUILD and job.paused and job.rise == GB


# --- the defaults and the history ----------------------------------------------------------


def test_the_reserve_is_16_gb_and_an_unknown_ram_job_is_16_gb(monkeypatch):
    monkeypatch.delenv("STAGELOAD_RESERVE_GB", raising=False)
    assert GuardConfig().reserve == 16 * GB
    assert predict_peak("zsh audio/run_ram_audio.sh", "w08-post-audio", None, {}) == (
        RAM_LANE_PEAK, "default of the ram lane")
    assert RAM_LANE_PEAK == 16 * GB
    assert predict_peak("python3 build_short.py S1", "code-isekai-omb-s1-4k-build", None, {},
                        lane="light") == (DEFAULT_PEAK, "default")


def test_the_next_audio_run_is_predicted_from_its_23_4_gb(tmp_path, gate):
    record = {"end": "2026-10-24T03:21:27+00:00", "lane": "ram", "who": "w08-post-audio",
              "cmd": "zsh audio/run_ram_audio.sh", "exit": 3, "peak_footprint": AUDIO_PEAK}
    (tmp_path / "history.jsonl").write_text(json.dumps(record) + "\n")
    expected = int(AUDIO_PEAK * HISTORY_FACTOR)
    peaks = queue.history_peaks(tmp_path)
    assert predict_peak("zsh audio/run_ram_audio.sh", "w08-post-audio", None, peaks) == (
        expected, "history of w08-post-audio x1.3")
    # a command line that names a smaller engine
    assert predict_peak("python foley_mac.py", "w08-post-audio", None, peaks)[0] == expected
    rows = []

    def swap():
        rows.extend(queue.list_queue(tmp_path))
        return 0

    io = GuardIO(level=lambda: 90, swap_used=swap, processes=lambda: [], log=lambda line: None,
                 compressor=lambda: GB)
    assert run_queued(TRUE, _quick(), io=io, home=tmp_path, who="w08-post-audio",
                      trace=None) == 0
    assert rows and all(row["mem_peak"] == expected for row in rows)


def test_the_two_jobs_of_the_incident_no_longer_fit_together(tmp_path, gate):
    lines = [{"who": "w08-post-audio", "peak_footprint": AUDIO_PEAK},
             {"who": "code-isekai-omb-s1-4k-build", "peak_footprint": BUILD_PEAK}]
    (tmp_path / "history.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    lane_dir = tmp_path / "queue" / "ram"
    lane_dir.mkdir(parents=True)
    running = lane_dir / f"{1:020d}-{os.getpid()}.json"
    running.write_text(json.dumps({"pid": os.getpid(), "who": "w08-post-audio", "lane": "ram",
                                   "cmd": "zsh audio/run_ram_audio.sh", "state": "running"}))
    own = predict_peak("python3 build_short.py S1 --name S1_4k", "code-isekai-omb-s1-4k-build",
                       None, queue.history_peaks(tmp_path), lane="light")[0]
    reason = budget_reason(tmp_path, tmp_path / "queue" / "light" / "x.json", own, GuardConfig(),
                           48 * GB)
    # 30.4 + 10.1 + 16 > 48; with the 4 GB defaults and a 12 GB reserve, 4 + 4 + 12 fitted
    assert reason.startswith("budget: running 30.4 GB (w08-post-audio (ram) 30 GB)")
    assert "+ ours 10.1 GB + reserve 16.0 GB > RAM 48.0 GB" in reason


def test_the_prefix_predicts_numbered_siblings_and_not_other_steps_of_a_chain():
    # 2026-10-24: w08-post-release (light, 3-4 GB) was counted at 31.9 GB from w08-post-audio
    peaks = {"w08-post-audio": int(24.5 * 2**30 / 1.0), "wide-w09-c010": 20 * GB,
             "w08-lips": 6 * GB}
    assert predict_peak("python3 release.py", "w08-post-release", None, peaks,
                        lane="light") == (DEFAULT_PEAK, "default")
    assert predict_peak("python3 vc.py", "w08-post-vc", None, peaks) == (
        RAM_LANE_PEAK, "default of the ram lane")
    assert predict_peak("python3 x.py", "wide-w09-c011", None, peaks)[0] == int(
        20 * GB * HISTORY_FACTOR)
    # a numbered who learns only from numbered siblings, not from a named step of the prefix
    assert predict_peak("python3 x.py", "w08-post-03", None, peaks, lane="light") == (
        DEFAULT_PEAK, "default")
    assert predict_peak("python3 x.py", "code-isekai-omb-s2", None,
                        {"code-isekai-omb-s1": 3 * GB})[0] == int(3 * GB * HISTORY_FACTOR)


# --- remote lanes --------------------------------------------------------------------------


def test_a_remote_lane_job_is_never_paused_or_stopped_for_this_machines_memory(tmp_path, gate,
                                                                                 monkeypatch):
    # 2026-10-24 03:11-03:21Z: a watcher script of 0.0 GB following a run on
    # another machine, was paused and then killed after 605.9 s (exit 3)
    monkeypatch.delenv("STAGELOAD_REMOTE_LANES", raising=False)
    bad = time.monotonic() + 0.3
    logs = []

    def worse(good, awful):
        return lambda: good if time.monotonic() < bad else awful

    io = GuardIO(level=worse(90, 5), swap_used=worse(0, 30 * GB), processes=lambda: [],
                 log=logs.append, compressor=worse(GB, 30 * GB), footprint=lambda pid: 0)
    trace = tmp_path / "arch.jsonl"
    assert run_queued(WORK, _quick(pause_max=0.3), io=io, home=tmp_path, lane="arch",
                      who="now-nrsur-arch", trace=trace) == 0
    assert not _lines(logs, "pausing") and not _lines(logs, "killing"), logs
    assert "never paused or stopped" in _lines(logs, "queue: start in arch")[0]
    record = json.loads((tmp_path / "history.jsonl").read_text().splitlines()[-1])
    assert record["who"] == "now-nrsur-arch" and record["exit"] == 0
    assert any(json.loads(line)["type"] == "sample" for line in trace.read_text().splitlines())


def test_remote_lane_jobs_are_neither_blamed_nor_counted_in_the_budget(tmp_path, monkeypatch):
    monkeypatch.delenv("STAGELOAD_REMOTE_LANES", raising=False)
    traces = tmp_path / "traces"
    traces.mkdir()
    for lane, who in (("arch", "now-nrsur-arch"), ("ram", "w08-post-audio")):
        lane_dir = tmp_path / "queue" / lane
        lane_dir.mkdir(parents=True)
        trace = _trace(traces / f"{lane}.jsonl", [0, 5 * GB])
        (lane_dir / f"{1:020d}-{os.getpid()}.json").write_text(json.dumps(
            {"pid": os.getpid(), "who": who, "lane": lane, "state": "running",
             "mem_peak": 20 * GB, "trace": str(trace)}))
    assert [job.who for job in running_jobs(tmp_path)()] == ["w08-post-audio"]
    reason = budget_reason(tmp_path, tmp_path / "queue" / "light" / "x.json", 10 * GB,
                           GuardConfig(reserve=16 * GB), 48 * GB)
    assert reason == ""  # 20 + 10 + 16 fit; the arch job's 20 GB are not on this machine


@pytest.mark.parametrize("our_start", [STARTED_BUILD - 3600, STARTED_BUILD, STARTED_BUILD + 3600])
def test_a_job_that_did_not_grow_is_never_the_one_paused_while_another_grew(our_start):
    flat = Growth("ours", grown=20 * GB, recent=0, me=True, rise=100 * MB, started=our_start)
    for mine in (None, 0, 100 * MB):
        verdict, target, _ = queue_culprit(mine, [flat, build()], 512 * MB)
        assert verdict == "theirs" and not target.me
