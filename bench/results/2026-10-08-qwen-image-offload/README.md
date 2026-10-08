# Qwen-Image-2.1 with model CPU offload, 2026-10-08

One run of [`bench/qwen_image_offload.py`](../../qwen_image_offload.py):
`QwenImage21Pipeline.from_pretrained(...)`, then `pipe.enable_model_cpu_offload(device="mps")`. The
prompt, size (1024 x 1024), steps (20), seed (7), weights and versions (torch 2.14.0, diffusers
0.41.0.dev0, stageload 0.1.0) are those of the [2026-10-06 runs](../2026-10-06-qwen-image/README.md),
on the same Apple M5 Pro with 48 GB. The guard had the same limits: start at 40 % available memory,
stop the run once swap has grown by 8 GB.

| run | peak footprint | swap growth | lowest available memory | outcome |
|---|---|---|---|---|
| eager, 2026-10-06 | 43.6 GB | 8.0 GB | 6 % | stopped by the guard during the VAE decode |
| model CPU offload, 2026-10-08 | 18.5 GB | 9.6 GB | 22 % | stopped by the guard during the VAE decode |
| staged, 2026-10-06, two runs | 19.0 GB | none | 51 and 56 % | finished in 94 and 124 s |

The offload run by stage: the pipeline loaded in 1.9 s. Encoding took 8 s at 18.5 GB with no swap.
Denoising took 87 s at up to 18.5 GB while swap grew by 3.5 GB and available memory fell to 27 %. The
VAE decode reached 16.2 GB, swap grew to 9.6 GB and available memory fell to 22 %, and the guard
stopped the run 105 s in. No image was written.

The process footprint looks like staged; the machine does not. Footprint does not count pages
already in swap, so a footprint-only comparison would make offload look like staged. Which pages
went to swap was not measured. This is one run: a second one would push the machine into swap again
for an outcome the guard had already decided.

`offload-1.trace.jsonl` is the meter's trace: every 0.5 s the process footprint, the swap in use and
`kern.memorystatus_level`, plus the stage events.
