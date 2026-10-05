# Pixal3D-mac with stageload

[Pixal3D-mac](https://github.com/pawel-mazurkiewicz/Pixal3D-mac) is the Apple Silicon port of
TencentARC's Pixal3D image-to-3D model. `stageload-pixal3d` runs it on one image with only the
current stage's weights in memory. The port's own code does the work; stageload changes when
models are loaded and released.

## Install

1. Install Pixal3D-mac by following its README. stageload is tested with commit
   `0be9e69a729432323dfb60a418801e5e9e7a1437`.
2. Install stageload into the port's environment without touching its packages:

   ```bash
   ~/local-llm/Pixal3D-mac/.venv/bin/pip install --no-deps git+https://github.com/nefayran/stageload
   ```

3. If the port is not at `~/local-llm/Pixal3D-mac`, pass `--port DIR` or set `PIXAL3D_MAC_DIR`.

## Run

```bash
stageload-pixal3d photo.png -o photo.glb
stageload-pixal3d photo.png -o photo.glb --load eager     # the port's own loading, for comparison
```

| Flag | Default | Meaning |
|---|---|---|
| `--load staged\|eager` | `staged` | staged loading, or the port's `load_pipeline` untouched |
| `--pipeline` | `1024_cascade` | passed to the port as `--pipeline-type` |
| `--seed` | `7` | passed to the port |
| `--texture-size` | `2048` | passed to the port |
| `--trace FILE` | next to the output | the JSONL trace of this run |
| `--port DIR` | `$PIXAL3D_MAC_DIR`, then `~/local-llm/Pixal3D-mac` | the port checkout |
| `--stop-at STAGE` | none | end the run where that stage would begin; no GLB is written |
| `--fingerprint DIR` | none | save a hash and a copy of every sampler's output |

Any other flag goes to the port's own argument parser, so its defaults and checks apply.

Refused in both modes, because they select other branches of the port's `main()` that this
command does not repeat: `--flash-sdpa`, `--load-mesh`, `--load-fixture-07`, and
`--free-spent-models` where a local copy of the port has it. Run `generate_mps.py` for those.

Refused in staged mode only: the port's diagnostic switches `PIXAL3D_FP32_MODELS`,
`PIXAL3D_CPU_MODELS`, `PIXAL3D_NAF_ANE_REPLACE`, `PIXAL3D_NAF_ANE_WHOLE`, `PIXAL3D_NAF_METAL` and
`PIXAL3D_DUMP_FIXTURES`. They change how models are built or moved, which staged loading replaces.

## Stages

| Stage | Begins when | In memory |
|---|---|---|
| setup | the command starts | nothing yet; the pipeline is built without reading any of its seven models |
| preprocess | `preprocess_image` is called | RMBG-2, released when it returns |
| camera | `preprocess_image` returns | MoGe-2, loaded and freed by the port |
| structure | `get_proj_cond_ss` is called | DINOv3, `sparse_structure_flow_model`, `sparse_structure_decoder` |
| shape_512 | `get_proj_cond_shape` gets the 512 extractor | DINOv3, NAF, `shape_slat_flow_model_512`, `shape_slat_decoder` |
| shape_1024 | `get_proj_cond_shape` gets the 1024 extractor | DINOv3, NAF, `shape_slat_flow_model_1024` |
| texture | `get_proj_cond_shape` gets the texture extractor | DINOv3, NAF, `tex_slat_flow_model_1024` |
| decode | `decode_latent` is called | `shape_slat_decoder`, `tex_slat_decoder` |
| export | the port's `image_to_asset` returns | no weights; the port writes the GLB |

The four DINOv3 extractors share one backbone, and the three that upsample share one NAF model.
They stay in memory from `structure` to `texture` and are released when `decode` begins. The shape
decoder is used in `shape_512` and again in `decode`; it is released in between and read again
from disk.

A staged pipeline serves one image: by the end of the run its weights are gone and its hooks stay
installed. The command builds a new one for every image.

## Read a trace

Every run writes a JSONL trace: a sample every 0.5 s (process footprint, swap used,
`kern.memorystatus_level`) and an event for every stage change, load and release.

```bash
stageload summary photo.trace.jsonl
```

prints the peak footprint overall and per stage, swap at the start, peak and end, time per stage,
and the bytes and seconds spent loading models. A model that was loaded outside the stage the plan
gives it is listed under `outside_stage`.

## Check that staged loading changes nothing

The port seeds once at the start of `run()` and draws every stage's noise from the CPU random
generator. Building a model runs its random initialisation, so a model built in the middle of the
run would change the noise of every later stage. `StagedModels` saves and restores the CPU, MPS
and CUDA generator states around each load, so a staged run draws exactly the noise an eager run
draws.

`--fingerprint DIR` hashes the output of every sampler call (sparse structure, the low- and
high-resolution shape latents, the texture latent) and saves a copy, and
`bench/compare_fingerprints.py A B` compares two runs call by call. Together with `--stop-at`, this
also works when eager loading does not fit in memory: stop the eager runs where the texture stage
would begin and compare the stages before it.

## Compare eager and staged

```bash
.venv/bin/python bench/pixal3d_ab.py --python ~/local-llm/Pixal3D-mac/.venv/bin/python \
  --image photo.png --out bench/results/my-run --runs 2 --fingerprints \
  --eager-stop-at texture          # only where eager loading does not fit
.venv/bin/python bench/compare_fingerprints.py bench/results/my-run/fp-eager-1 bench/results/my-run/fp-staged-1
.venv/bin/python bench/compare_glb.py bench/results/my-run/eager-1.glb bench/results/my-run/staged-1.glb
.venv/bin/python bench/summarize_runs.py bench/results/my-run
.venv/bin/python bench/plot_trace.py bench/results/my-run/eager-1.trace.jsonl \
  bench/results/my-run/staged-1.trace.jsonl --out bench/results/my-run/memory
```

Each generation runs in its own process under `stageload guard`, one after another, eager and
staged in turn. The driver stops at the first run that does not finish, so a run the guard had to
stop is never followed by another one.
