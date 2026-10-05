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
   cd path/to/Pixal3D-mac
   .venv/bin/pip install --no-deps git+https://github.com/nefayran/stageload
   ```

3. Run the command from the port's environment, which has torch and the port's other packages.
   It looks for the port in `--port DIR`, then `$PIXAL3D_MAC_DIR`, then the checkout whose
   `.venv` runs it, then the current directory.

## Run

```bash
.venv/bin/stageload-pixal3d photo.png -o photo.glb
.venv/bin/stageload-pixal3d photo.png -o photo.glb --load eager   # the port's own loading
```

| Flag | Default | Meaning |
|---|---|---|
| `--load staged\|eager` | `staged` | staged loading, or the port's `load_pipeline` untouched |
| `--pipeline` | `1024_cascade` | passed to the port as `--pipeline-type`; staged loading has a plan for `1024_cascade` only |
| `--seed` | `7` | passed to the port |
| `--texture-size` | `2048` | passed to the port |
| `--trace FILE` | next to the output | the JSONL trace of this run |
| `--port DIR` | see Install | the port checkout |
| `--stop-at STAGE` | none | end the run where that stage would begin (any stage after `setup`); no GLB is written |
| `--fingerprint DIR` | none | save a hash and a copy of every sampler's output |

Any other flag goes to the port's own argument parser, so its defaults and checks apply.

Refused in both modes, because they select other branches of the port's `main()` that this
command does not repeat: `--flash-sdpa`, `--load-mesh`, `--load-fixture-07`, and
`--free-spent-models` where a local copy of the port has it. They are checked after the port has
parsed them, so abbreviations such as `--flash` are refused too. Run `generate_mps.py` for those.

Refused in staged mode only: the port's diagnostic switches `PIXAL3D_FP32_MODELS`,
`PIXAL3D_CPU_MODELS`, `PIXAL3D_NAF_ANE_REPLACE`, `PIXAL3D_NAF_ANE_WHOLE`, `PIXAL3D_NAF_METAL` and
`PIXAL3D_DUMP_FIXTURES`. They change how models are built or moved, which staged loading replaces.

## Stages

| Stage | Begins when | In memory |
|---|---|---|
| setup | the command starts | RMBG-2, one DINOv3 backbone and one NAF model; none of the pipeline's seven models is read yet |
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

## Check what staged loading changes

The port seeds once at the start of `run()` and draws every stage's noise from the CPU random
generator. Building a model runs its random initialisation, so a model built in the middle of the
run would change the noise of every later stage. `StagedModels` saves and restores the random
generators around each load (torch's CPU, MPS and CUDA ones, Python's `random` and NumPy's global
one), so a staged run draws the noise an eager run draws.

One thing does differ. The port builds its models on the CPU and moves them to MPS; stageload
builds them on MPS, which avoids a second copy during every load. A tensor a model computes when
it is built, rather than reads from its checkpoint, can then come out slightly different: the
sparse attention's rotary frequencies differ by up to 6e-8. Building on the CPU instead cost 2.6
to 11.7 GB more in four stages and more than tripled the loading time, without bringing the
results measurably closer ([bench/results/2026-10-05](../bench/results/2026-10-05/README.md)).

`--fingerprint DIR` hashes the output of every sampler call (sparse structure, the low- and
high-resolution shape latents, the texture latent) and saves a copy, and
`bench/compare_fingerprints.py A B` compares two runs call by call. Together with `--stop-at`, this
also works when eager loading does not fit in memory: stop the eager runs where the texture stage
would begin and compare the stages before it. On MPS the port does not repeat its sparse stages
with the same seed, so the comparison shows how far apart two runs are, not whether staged loading
changed anything. The texture latent and the GLB have not been compared on this machine, because
the eager runs do not get through the texture stage here.

## Compare eager and staged

In a clone of this repository, with its own `.venv` (`pip install -e ".[test,bench]"`):

```bash
R=bench/results/my-run
.venv/bin/python bench/pixal3d_ab.py --python path/to/Pixal3D-mac/.venv/bin/python \
  --image photo.png --out $R --runs 2 --fingerprints \
  --eager-stop-at texture \
  --busy 'ltx-2-mlx'                  # other heavy jobs of the machine to wait for
.venv/bin/python bench/compare_fingerprints.py $R/fp-eager-1 $R/fp-staged-1
.venv/bin/python bench/summarize_runs.py $R
.venv/bin/python bench/plot_stages.py $R/summary.json --out $R/stages
.venv/bin/python bench/plot_trace.py $R/eager-1.trace.jsonl $R/staged-1.trace.jsonl --out $R/memory
```

Leave out `--eager-stop-at texture` where eager loading fits in memory; the eager runs then write
a GLB too, and `bench/compare_glb.py $R/eager-1.glb $R/staged-1.glb` compares the meshes and
their textures.

Each generation runs in its own process under `stageload guard`, one after another, eager and
staged in turn. Every run waits until no other Pixal3D run is going and, with `--busy`, until
the machine's other heavy jobs have been quiet for three minutes. The driver stops at the first
run that does not finish, so a run the guard had to stop is never followed by another one.
