# What PYTORCH_MPS_HIGH_WATERMARK_RATIO changes on a 48 GB Mac (13.10.2026)

MacBook Pro M5 Pro, 48 GB, macOS 26.5.2, PyTorch 2.14.0. `bench/mps-watermark/fill.py` allocates 512 MiB float16
tensors of random values on MPS, one every half second, until PyTorch raises or 46 GiB are held. Every run went
through `stageload run --swap-budget 2G`, so a run that pushed the Mac into swap was stopped by the guard rather than
by PyTorch. `torch.mps.recommended_max_memory()` reported 37.44 GiB in every run. Other programs were running on the
Mac at the same time (as on any working Mac), so the point where swap starts depends on what else is open.

| setting | what stopped the run | tensors held | peak footprint | swap during the run |
|---|---|---|---|---|
| `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5` alone | PyTorch at start: `invalid low watermark ratio 1.4` | 0 | 0.1 GB | no change |
| `PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0` alone | PyTorch at start: `invalid low watermark ratio 1.4` | 0 | 0.1 GB | no change |
| high 0.5, low 0.5 | PyTorch: `MPS backend out of memory (MPS allocated: 18.50 GiB, other allocations: 448.00 KiB, max allowed: 18.72 GiB)` | 18.5 GiB | 20.1 GB | 3.7 → 3.7 GB |
| high 0.8, low 0.8 | PyTorch: `MPS backend out of memory (MPS allocated: 29.50 GiB, other allocations: 448.00 KiB, max allowed: 29.95 GiB)` | 29.5 GiB | 31.9 GB | 5.6 → 5.6 GB |
| high 1.0, low 1.0 | the guard, swap over budget | 34.5 GiB | 37.2 GB | 4.6 → 8.7 GB |
| default (no variable) | the guard, swap over budget | 38.5 GiB | 41.5 GB | 6.1 → 12.0 GB |
| high 0.0 | the guard, swap over budget | 36.0 GiB | 38.9 GB | 5.1 → 9.8 GB |

PyTorch's own description (https://docs.pytorch.org/docs/2.14/mps_environment_variables.html): the high ratio is a
hard limit, "Default is 1.7", and "0.0 : disables high watermark limit (may cause system failure if system-wide OOM
occurs)"; the low ratio is a soft limit, "Default is 1.4 (unified) or 1.0 (discrete)", and "setting 0.0 disables
adaptive commit and garbage collection".

What the numbers say:

- PyTorch's limit is `ratio × recommended_max_memory`: 0.5 × 37.44 = 18.72 GiB, the "max allowed" in the error.
- The default ratio is 1.7, so the default limit here is about 63.6 GiB, more than the Mac's 48 GB. With the default
  setting the Mac went into swap long before PyTorch would have raised anything; it behaved like `0.0`, which removes
  the limit.
- Even 1.0 (37.44 GiB) was more than this Mac had free: in the 1.0, 0.0 and default runs swap began to grow once the
  process footprint passed 33.5–36.2 GB (first sample with swap 0.25 GB above its start), before any PyTorch limit.
- Setting only the high ratio below the default low ratio (1.4) does not lower the limit: MPS refuses to start.
  The low ratio has to be set too, and no higher than the high one.
- On this Mac 0.8 (29.95 GiB) turned the slide into swap into a clean error with no swap at all.
- PyTorch reads both variables when MPS is first used, not at `import torch`: set from Python with `os.environ`
  after the import but before the first MPS tensor, the high ratio alone still gave the low-ratio error; set after
  the first MPS tensor, it was ignored (checked 13.10 with the same PyTorch).

A first run filled the tensors with ones instead of random values and reached the 46 GiB cap with the default setting,
no error and no swap: macOS compresses pages of identical values almost to nothing. Its files are kept as
`default-ones.*`; a memory test on a Mac has to use data that does not compress.

Files: `<setting>.jsonl` (one line per 512 MiB step, then the end and PyTorch's error text), `<setting>.stderr.txt`
(the queue and the guard), `<setting>.trace.jsonl` (the guard's memory trace, `stageload summary` reads it).
