# Qwen-Image-2.1 through diffusers, eager against staged (2026-10-06)

## Setup

- Apple M5 Pro with 48 GB of memory; torch 2.14.0, diffusers 0.41.0.dev0 (from main, which has
  `QwenImage21Pipeline`), transformers 5.17.0.
- `Qwen/Qwen-Image-2.1` from the Hugging Face cache, in bfloat16: a text encoder (Qwen3-VL,
  16.3 GB), a transformer (13.3 GB) and a VAE (1.3 GB). Text to image without reference images,
  1024 x 1024, 20 steps, seed 7, the prompt in [`bench/qwen_image.py`](../../qwen_image.py).
- Eager is the usual `QwenImage21Pipeline.from_pretrained(...).to("mps")`. Staged keeps the VAE
  and loads the text encoder and the transformer through `StagedModels`. Its stages begin at
  `encode_prompt`, `prepare_latents` and `_unpack_latents`; the pipeline's code is unchanged, and
  its two large models are stand-ins that answer `.config` from the checkpoint and load the model
  on any other use.
- Every run is its own process under `stageload guard --wait-free 40 --swap-budget 8G`, with
  `--busy` patterns for the other heavy jobs on this machine. A video job that shares the machine
  paused for these runs.
- GB means 2^30 bytes. Footprint is the process's `phys_footprint`, the number Activity Monitor
  shows, which includes memory macOS has compressed.

## Memory

| mode | runs | ran to | peak footprint (GB) | swap rise (GB) | time (s) |
|---|---|---|---|---|---|
| eager | 1 | decode (stopped by the guard) | 43.57 | 7.97 | 118.4 |
| staged | 2 | end / end | 19.0 / 19.0 | 0.0 / 0.0 | 124.3 / 94.3 |

| stage | eager (GB) | staged (GB) |
|---|---|---|
| load | 31.4 | 2.08 / 2.08 |
| encode | 31.44 | 19.0 / 19.0 |
| denoise | 32.52 | 16.2 / 18.62 |
| decode | 43.57 | 14.41 / 14.43 |

(`summarize_runs.py` names the first stage `setup`; [table.md](table.md) has its output.)

Eager loaded the pipeline in 20 s and ran all 20 steps. In the VAE decode its footprint rose from
32.5 to 43.6 GB and swap grew by 8 GB, and the guard stopped it 10 s into the decode, before it
wrote an image. At 1024 px the decode needs about 11 GB on top of the weights that are resident.
Staged released the transformer before the decode and peaked at 14.4 GB there. Neither staged run
added any swap, and neither loaded a model outside the stage that lists it.

An earlier eager run, started a minute after the video job's last process had ended
([first-attempt/eager-1.trace.jsonl](first-attempt/eager-1.trace.jsonl)), did not get past
loading: swap grew from 8.6 to 18.5 GB in 18 s, and the guard stopped it at a footprint of
27.4 GB.

## Time

| stage | eager (s) | staged run 1 (s) | staged run 2 (s) |
|---|---|---|---|
| load | 19.9 | 4.5 | 4.5 |
| encode | 10.9 | 8.4 | 8.8 |
| denoise | 77.5 | 109.2 | 79.0 |
| decode | 10.1 (stopped) | 2.2 | 1.9 |

Staged loads the text encoder in its encode stage and the transformer in its denoise stage; the
two loads took 12.6 s in the first run and 13.0 s in the second. The second run's denoising took
79 s against eager's 77.5 s, so a step costs the same. The first run's denoising took 109 s, and
its trace does not show why.

## Output

The two staged runs produced the same image, bit for bit. Eager produced none, so these runs do
not show that both modes give the same image.

## Reproduce

```bash
python bench/qwen_image_ab.py --out RESULTS --runs 1 --steps 20 --size 1024
python bench/summarize_runs.py RESULTS
python bench/plot_qwen_stages.py RESULTS/summary.json --out RESULTS/stages
```

The A/B runner stops after a run that fails, as the eager run here did; the two staged runs were
started with `bench/qwen_image.py --mode staged` under the same guard.
