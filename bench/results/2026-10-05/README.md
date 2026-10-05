# Pixal3D `1024_cascade`, eager against staged (2026-10-05)

## Setup

- Apple M5 Pro with 48 GB of memory, torch 2.12.0 in the port's environment.
- Pixal3D-mac at commit `0be9e69`. The checkout has two local changes that only act when asked for
  (a `--free-spent-models` flag and a `PIXAL3D_REMESH_RES` variable); neither was used.
- Input: the port's example image `assets/images/0_img.png` (not copied into this repository).
  Pipeline `1024_cascade`, seed 7, texture size 2048.
- Every run is its own process under
  `stageload guard --wait-free 40 --swap-budget 8G --quiet-for 3m`, with `--busy` patterns for
  the other heavy jobs on this machine. A video generation job was running on the machine between
  our runs; each run started after three quiet minutes.
- Eager runs stop where the texture stage would begin (`--stop-at texture`). On this machine
  the port's own loading does not get through the texture stage within the guard's limits; see
  [A full eager run](#a-full-eager-run).
- GB means 2^30 bytes. Footprint is the process's `phys_footprint`, the number Activity Monitor
  shows, which includes memory macOS has compressed.

## Memory

| mode | runs | ran to | peak footprint (GB) | swap rise (GB) | time (s) |
|---|---|---|---|---|---|
| eager | 2 | texture (stopped) / texture (stopped) | 30.55 / 30.53 | 2.27 / 3.87 | 351.0 / 410.5 |
| staged | 2 | end / end | 69.57 / 70.19 | 0.0 / 2.03 | 595.9 / 631.9 |

| stage | eager (GB) | staged (GB) |
|---|---|---|
| setup | 20.41 / 20.4 | 2.93 / 2.93 |
| camera | 23.18 / 23.18 | 6.72 / 6.72 |
| structure | 22.21 / 22.21 | 10.9 / 10.91 |
| shape_512 | 26.6 / 26.6 | 11.46 / 11.44 |
| shape_1024 | 30.55 / 30.53 | 13.06 / 13.06 |
| texture | – / – | 30.08 / 30.1 |
| decode | – / – | 17.87 / 18.35 |
| export | – / – | 69.57 / 70.19 |

![Peak footprint per stage, eager against staged](stages.png)

In every stage both modes ran, eager holds between 11.3 and 17.5 GB more than staged. The staged
runs loaded eight models (seven, plus the shape decoder a second time for decoding), 13.2 GB in
24 s, out of about ten minutes per run.

Most of the texture stage's 30 GB is not weights: the texture model is 2.6 GB and the image
feature extractors 1.1 GB. When the decode stage began, stageload released both and emptied the
MPS allocator's cache, and within two seconds the footprint fell from 30.1 GB to 4.5 GB in one
staged run and to 8.0 GB in the other.

![Process footprint over time](memory.png)

The highest staged footprint is in the export, after every model weight has been released. It
comes from the port's own mesh processing (remeshing, BVH, cleanup, UV unwrapping, baking), which
stageload does not touch. During part of the export the meter took no samples: it runs as a
thread inside the process, and the port's native code held the interpreter lock. Those are the
straight segments in the chart, and the export peak may be higher than measured.

## A full eager run

On 2026-10-04 an eager run with the same settings went into the texture stage. Within 10 s its
footprint rose from 27 to 44.2 GB and available memory fell to 6 %; swap, 14.8 GB when the run
started, reached 27.0 GB, and the guard stopped it
([trace](full-eager-attempt/2026-10-04-eager.trace.jsonl), middle panel of the chart). Both staged
runs went through the texture stage at 30.1 GB with at least 32 % of memory available.

## Output

Every run saved a hash and a copy of each sampler's output (`--fingerprint`), and
[`compare_fingerprints.py`](../../compare_fingerprints.py) compared them pair by pair
(`compare-*.json`).

| pair | sparse structure latent | 512 shape latent, largest difference | 1024 shape tokens |
|---|---|---|---|
| eager-1 / eager-2 | identical | 1.77 | 13,755 / 13,748 |
| eager-1 / staged-1 | identical | 1.24 | 13,755 / 13,740 |
| eager-1 / staged-3 | identical | 1.86 | 13,755 / 13,754 |
| eager-2 / staged-1 | identical | 1.85 | 13,748 / 13,740 |
| eager-2 / staged-3 | identical | 1.82 | 13,748 / 13,754 |
| staged-1 / staged-3 | identical | 2.04 | 13,740 / 13,754 |

The sparse structure latent is the same bit for bit in all four runs. From the 512 shape stage on,
any two runs differ: with the same seed the port itself does not repeat its sparse stages on MPS.
The one pair of eager runs differs by 1.77, and a staged run differs from an eager one by 1.24
to 1.86. So these runs cannot show whether staged loading changes the later stages; they show
that any change is no larger than the port's own variation, which was measured on that single
eager pair. The texture latent and the GLB were not compared, because the eager runs stop before
the texture stage on this machine.

## Building on the CPU (experiment)

The second staged run, in [`cpu-build/`](cpu-build), built every model on the CPU and then moved
it to MPS, the way the port builds them. Its peaks were higher in four of the five stages that
load models, by 2.6 to 11.7 GB (shape_512 14.1 GB, shape_1024 16.0, texture 33.9, decode 29.5),
and 3.0 GB lower in the structure stage (7.9 GB); loading took 82 s instead of 24 s. Its 512
shape latent differs from the two eager runs by 1.84 and 1.12, inside the range above.
stageload builds each model directly on the device.

## Another image

[`showcase/`](showcase) holds the trace of a staged run on the character shown at the top of the
main README (same settings). It took 416 s and loaded the same eight models, 13.2 GB in 37 s. Its
peaks were 13.0 GB in shape_1024, 30.1 GB in texture and 20.0 GB in decode; the highest, 59.3 GB,
was again in the export. Available memory stayed at 22 % or more.

## Files

- `*.trace.jsonl`: one trace per run, readable with `stageload summary`.
- `summary.json`, `table.md`: written by `bench/summarize_runs.py`.
- `compare-*.json`, `cpu-build/compare-*.json`: fingerprint comparisons.
- `stages.svg`, `stages.png`: written by `bench/plot_stages.py`.
- `memory.svg`, `memory.png`: written by `bench/plot_trace.py`.
- `runs.json`: the runs and how each ended.
