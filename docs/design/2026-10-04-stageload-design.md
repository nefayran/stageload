# stageload: design

Date: 2026-10-04
Status: draft for review

## Summary

`stageload` runs multi-stage PyTorch pipelines on Apple Silicon with only the current stage's
weights in memory. A model is loaded the first time a stage asks for it and released in place when
the pipeline enters a stage that does not need it. The first and only adapter in v1 is
[Pixal3D-mac](https://github.com/pawel-mazurkiewicz/Pixal3D-mac), the Apple Silicon port of
TencentARC's Pixal3D image-to-3D model. The package also ships a process memory meter and a
`guard` command that keeps one heavy job from taking a shared Mac down.

## Problem

Pixal3D's `1024_cascade` pipeline uses seven models (sparse-structure flow and decoder, two shape
flows at 512 and 1024, the shape decoder, the texture flow and the texture decoder), four DINOv3
ViT-L feature extractors, an NAF upsampler, RMBG-2 for background removal and MoGe-2 for the
camera. The flow models are about 5 GB each on disk.

What the code does today (read from the port at commit `0be9e69`, not yet measured):

- `init_pipeline` loads all seven models up front. `pipeline.json` has no `low_vram` key, so the
  pipeline's default of `low_vram=True` applies, and `pipeline.to("mps")` then leaves every model
  on the CPU. Each stage copies its model to MPS and back with `.to(device)` / `.cpu()`.
- On a Mac the CPU and the GPU share one pool of memory. The "offload to CPU" therefore frees
  nothing: all weights stay resident for the whole run, and the active stage has a second copy.
- The four DINOv3 extractors each call `DINOv3ViTModel.from_pretrained` with the same model, so the
  same frozen backbone is in memory four times. Three of them also load their own NAF upsampler.

What was measured on this machine (Apple M5 Pro, 48 GB) before this project:

- Releasing the weights of finished stages freed 11.3 GB before the texture stage and 5.5 GB before
  the GLB export.
- A `1024_cascade` run with other work on the machine took swap from 18 GB to 36 GB, and the Mac
  rebooted.
- These numbers come from runs with different images and different background load. They show the
  problem but are not a controlled comparison, and the README and the post will not use them.

## Goals

1. Run Pixal3D `1024_cascade` with only the current stage's weights resident, and produce the same
   output as the port's own path.
2. Measure the difference: peak process footprint overall and per stage, swap growth and wall time,
   eager against staged on the same image and seed.
3. Keep the core small and independent of Pixal3D, so the same pieces work for another multi-stage
   pipeline.
4. Give a shared Mac a guard: a heavy job starts only when there is room and is killed before it
   can take the machine down.

## Non-goals (v1)

- Adapters for other models (TRELLIS.2 and others). The core is generic; a second adapter waits for
  a real need.
- Publishing to PyPI. v1 installs from GitHub.
- CUDA, Linux or Windows memory metering. The core runs anywhere PyTorch runs; the meter and the
  guard read macOS counters.
- One process per stage, a warm pipeline reused across many images, ComfyUI nodes.

## Architecture

```
src/stageload/
  release.py     release(module) and ReleasedModuleError
  registry.py    StagedModels: lazy, stage-aware model registry
  share.py       share(): one instance for identical loads
  hooks.py       on_call(): run a callback before or after a method of one object
  events.py      Event and the EventSink protocol
  meter.py       macOS memory readers and MemoryMeter
  guard.py       admission control and a swap budget for one command
  cli.py         `stageload guard` and `stageload summary`
  pixal3d/
    port.py      locate, import and check the Pixal3D-mac port
    staged.py    build the staged pipeline: plan, hooks, shared backbone
    cli.py       `stageload-pixal3d`
bench/
  pixal3d_ab.py  eager against staged, each run in its own guarded process
  compare_glb.py output comparison
  plot_trace.py  memory chart from trace files
tests/
docs/
```

### Core

**`release(module, *, name=None, empty_cache=True) -> int`**
Replaces every parameter and buffer of `module` and its submodules with a tensor on the `meta`
device, runs `gc.collect()` and, if MPS or CUDA is available and `empty_cache` is true, empties
the allocator cache. Returns the number of bytes freed. The module object stays valid, so
references held elsewhere (for example a local variable inside a pipeline's `run()`) do not keep
the memory alive. The module is marked as released, and its `forward` raises
`ReleasedModuleError(name, stage)` instead of the PyTorch error about meta tensors. Tensors whose
storage is shared with a module that is not being released are left alone and counted in the
`release` event.

**`StagedModels(loaders, *, stages, pinned=(), events=None)`**
A `MutableMapping[str, nn.Module]`.

- `loaders` maps a name to a zero-argument callable that builds the module on the target device.
- `stages` maps a stage name to the model names that stage needs.
- `models[name]` loads the model on first access, which emits a `load` event with bytes and seconds.
- `name in models`, `len(models)` and iteration over names never load anything.
- `values()` and `items()` yield only the models that are loaded right now. Iterating the registry
  must not load the whole pipeline.
- `enter(stage)` releases every loaded model that the stage does not list and that is not pinned,
  then records `current_stage`.
- Accessing a model the current stage does not list still loads it, and the event is marked
  `outside_stage`, so a wrong plan shows up in the trace instead of failing.
- A model released earlier in the run is loaded again on access (`reload` event).
- `models[name] = module` registers a module that is already built. It has no loader, so accessing
  it after release raises `ReleasedModuleError`.
- `pin(name)` and `unpin(name)` keep a model across stages. `close()` releases everything, pinned
  models included.

**`share(owner, attr, *, key=None)`**
A context manager. Inside the block, `getattr(owner, attr)` is wrapped so that calls with the same
key (by default the call's arguments) return the instance built by the first call. On exit the
original attribute is restored. It is used while the adapter builds the four DINOv3 extractors.

**`on_call(obj, method, *, before=None, after=None) -> uninstall`**
Wraps one method of one object (an instance attribute, so other instances are not affected).
`before(*args, **kwargs)` runs before the call and `after(result, *args, **kwargs)` after it.
Calling `uninstall()` restores the original method.

**`Event` and `EventSink`**
`Event(t, kind, name, stage, nbytes, seconds, note)`, where `kind` is one of `load`, `reload`,
`release`, `stage` or `mark`. An `EventSink` has one method, `emit(event)`. `MemoryMeter`
implements it.

### Meter

- `process_footprint(pid=None) -> int` reads `ri_phys_footprint` through `proc_pid_rusage` in
  `libproc`. This is the number Activity Monitor shows in its Memory column. It includes Metal
  allocations made by the process and needs no root for the process itself. A probe on this
  machine read 308 MB after a 300 MB allocation.
- `swap_used() -> int` parses `sysctl vm.swapusage`. `memorystatus_level() -> int` reads
  `kern.memorystatus_level` (0–100, percent of memory available).
- `MemoryMeter(path, *, interval=0.5)` is a context manager. A background thread writes one JSON line
  per sample (`t`, `footprint`, `swap_used`, `memorystatus_level`) and one per event, flushed per
  line, so the trace survives a kill. `summary()` returns the peak footprint overall and per stage,
  swap at the start, at its peak and at the end, and per-stage wall time.

### Guard

`stageload guard [options] -- command ...` runs one command under two rules.

Admission. The command starts only when all of these hold at the same moment:
- `memorystatus_level` is at least `--wait-free` (default 40);
- if `--swap-limit` is set, swap used plus `--swap-budget` is within it;
- no running process matches any `--busy REGEX` (repeatable, no default patterns).

If that does not happen within `--start-timeout` (default 30 min), the guard exits with code 4.

Budget. While the command runs, every `--poll` seconds (default 5) the guard reads swap and
`memorystatus_level`. It sends SIGTERM to the command's process group, then SIGKILL after 5 s, and
exits with code 3 when either of these happens:
- swap used goes over the swap at start plus `--swap-budget` (default 8 GB), or over
  `--swap-limit`;
- `memorystatus_level` stays under `--kill-free` (default 10) for two samples in a row.

Otherwise it exits with the command's own code. Every decision is logged with the readings
behind it.

The guard does not replace a system-wide OOM killer such as
[macos-oom-guard](https://github.com/fl4p/macos-oom-guard), which runs as root and kills the
largest process on the machine. This guard is for one job you start yourself: it waits for room
before starting, and the only process it ever kills is that job.

## Pixal3D adapter

### Port

`Port.load(path)` adds the port's directory to `sys.path`, imports `generate_mps` and `pixal3d`,
and checks that the names the adapter uses exist: `parse_args`, `load_runtime_deps`,
`_configure_fdg_environment`, `load_pipeline`, `image_to_asset`, `asset_to_glb`,
`IMAGE_COND_CONFIGS`, `build_image_cond_model`, and on the pipeline class `get_proj_cond_ss`,
`get_proj_cond_shape`, `decode_latent` and `preprocess_image`. A missing name stops the run with a
message naming the tested commit (`0be9e69`). A different commit only prints a warning. The port is
found from `--port DIR`, then `$PIXAL3D_MAC_DIR`, then `~/local-llm/Pixal3D-mac`.

### Building the staged pipeline

1. Run the port's runtime setup: `load_runtime_deps()` and `_configure_fdg_environment(args)`.
2. Build the pipeline with the port's own constructor, `Pixal3DImageTo3DPipeline.from_pretrained`,
   while `pixal3d.models.from_pretrained` is replaced by a recorder. The recorder returns a small
   placeholder module and remembers the checkpoint path for each name, so the samplers, the
   normalization and the rest of the configuration come from the port's code, and none of the
   seven pipeline models is read. The constructor still builds RMBG-2 directly, as it always does;
   it is the first model the run needs.
3. Replace `pipeline.models` with a `StagedModels` whose loaders call the real
   `pixal3d.models.from_pretrained(path)`, move the model to MPS and call `eval()`.
4. Build the four extractors inside `share(DINOv3ViTModel, "from_pretrained")`, so they hold one
   backbone. Load NAF once and give the same instance to the three extractors that use it.
5. Set `pipeline.low_vram = False` and `pipeline._device = mps`. Models are created on MPS and are
   never copied to the CPU and back.
6. Release the RMBG-2 model with `on_call(pipeline, "preprocess_image", after=...)`. MoGe-2 is
   already loaded and freed inside `image_to_asset` by the port.
7. Install the stage hooks below, run the port's `image_to_asset`, call `models.close()` and release
   the extractors, then run the port's `asset_to_glb`.

### Stage plan (1024_cascade)

| Stage | Entered when | Resident |
|---|---|---|
| preprocess | `preprocess_image` is called | RMBG-2, released when it returns |
| camera | inside `image_to_asset` | MoGe-2, loaded and freed by the port |
| structure | `get_proj_cond_ss` is called | DINOv3, `sparse_structure_flow_model`, `sparse_structure_decoder` |
| shape_512 | `get_proj_cond_shape` is called with the 512 extractor | DINOv3, NAF, `shape_slat_flow_model_512`, `shape_slat_decoder` (upsampling) |
| shape_1024 | `get_proj_cond_shape` is called with the 1024 extractor | DINOv3, NAF, `shape_slat_flow_model_1024` |
| texture | `get_proj_cond_shape` is called with the texture extractor | DINOv3, NAF, `tex_slat_flow_model_1024` |
| decode | `decode_latent` is called | `shape_slat_decoder`, `tex_slat_decoder` |
| export | after `image_to_asset` returns | no weights |

DINOv3 and NAF are needed by four consecutive stages, so they stay pinned until `decode`, which
unpins and releases them. The shape decoder is needed in `shape_512` and in `decode`. It is
released in between and loaded again for `decode`, which costs one read of a 0.9 GB file.

The other pipeline types (`512`, `1024`, `1536_cascade`) call the same methods. Their stages
resolve the same way, and a model a stage never asks for is never loaded.

### `stageload-pixal3d`

```
stageload-pixal3d IMAGE -o OUT.glb [--port DIR] [--load staged|eager]
                  [--pipeline 1024_cascade] [--seed 7] [--texture-size 2048]
                  [--trace TRACE.jsonl] [-- extra generate_mps flags]
```

The port's arguments are built with its own `parse_args`, so its defaults apply. `--load eager`
uses the port's `load_pipeline` untouched and is the baseline for comparisons. Both modes write
the same trace format.

### Errors

- Port missing, or a checked name missing: exit before loading anything, naming the commit the
  adapter was tested against.
- The port's diagnostic switches (`PIXAL3D_FP32_MODELS`, `PIXAL3D_CPU_MODELS`,
  `PIXAL3D_NAF_ANE_*`, `PIXAL3D_NAF_METAL`, `PIXAL3D_DUMP_FIXTURES`) in staged mode: refuse before
  loading and suggest `--load eager`.
- A model used outside its stage: loaded on demand and flagged in the trace and the summary.
- A released module called again: `ReleasedModuleError` names the model and the stage that
  released it.
- The process killed by the guard or by the system: the trace up to the last sample is on disk.

## Measurement protocol

- Same input image, `1024_cascade`, seed 7, texture size 2048, the port at `0be9e69`, this
  machine.
- Runs alternate eager, staged, eager, staged. Each run is a fresh process under
  `stageload guard --wait-free 40 --swap-budget 8G` with `--busy` patterns for the other heavy jobs
  on this machine, so every run starts on a quiet machine.
- From the traces: peak footprint overall and per stage, swap at the start and at the peak, wall
  time per stage and in total, model load time and bytes.
- Output comparison: staged against eager, and eager against eager as the noise reference.
  Compared are vertex and face counts, vertex positions (maximum and mean distance) and the base
  color texture (maximum difference and PSNR).
- `bench/results/<date>/` holds the traces, `summary.json` and the chart (memory over time, one
  line per mode, stage bands). The README and the post quote numbers from `summary.json` only.
- The input is one of the port's example images, referenced by path and not copied into this
  repository. Before any image appears in the README or the post, its license is checked; if it is
  unclear, an image of our own is used instead.

## Testing

All tests run on the CPU with small modules, under GitHub Actions on `ubuntu-latest` and
`macos-latest`, with ruff.

- `release`: every parameter and buffer is on `meta` afterwards; a weak reference to an old
  parameter dies; shared storage is kept and reported; `forward` raises `ReleasedModuleError`.
- `StagedModels`: lazy loading, `in` without loading, `values()` without loading, `enter` releases
  what the stage does not need, pinned models stay, reload after release, `outside_stage`
  flagging, `close`.
- `share` and `on_call`: one instance per key, original attribute restored, the predicate sees the
  call's arguments, `uninstall` leaves no trace.
- Meter: `vm.swapusage` parsing; on macOS `process_footprint` grows after an allocation; trace lines
  are valid JSON and are flushed.
- Guard, with fake readers: admission waits and then starts; timeout gives exit code 4; a budget
  breach kills a real child process group (`sleep 60`) and gives exit code 3; `--busy` matching;
  the child's own exit code is passed through.
- Adapter, against a stub port with the same API (`models`, `get_proj_cond_ss`,
  `get_proj_cond_shape`, `decode_latent`, `image_to_asset`, `asset_to_glb`): stage order, what is
  resident in each stage, `low_vram` off, the refusal on diagnostic switches, the error on a
  missing name. No weights are needed.

The real Pixal3D run is checked by hand: one staged smoke run, then the bench.

## Packaging

- Python 3.10 or newer (the port's environment is 3.10.21 with torch 2.12.0). Dependencies:
  `torch>=2.1` and `safetensors`. The `bench` extra adds `numpy`, `trimesh` and `matplotlib`.
- Installed into the port's environment with
  `pip install git+https://github.com/nefayran/stageload`. Console scripts: `stageload` and
  `stageload-pixal3d`.
- MIT license, the same as Pixal3D and the port. No code from the port is copied; the adapter
  imports it at run time.

## Risks

- The MPS allocator may keep released memory cached. `release` empties the cache, and the per-stage
  footprint in the trace shows whether memory really goes back. If it does not, that goes into the
  README as a limitation.
- MPS may not be bit-for-bit deterministic. The eager-against-eager comparison sets the noise
  level that staged results are judged against.
- `low_vram=False` takes the port's other branch of `run()`. That branch only skips device copies,
  and the output comparison checks it.
- Sharing one DINOv3 backbone assumes no extractor changes the backbone's state. The extractors
  freeze it with `requires_grad_(False)`, and the output comparison checks the rest.
- Reading weights from disk at each stage adds time. The trace measures it, and the summary
  reports it next to the total.
- A new port commit can rename or restructure methods. The adapter checks names at start and is
  pinned to a tested commit.

## Order of work

1. Core: `release`, `StagedModels`, `share`, `on_call`, events, with tests.
2. Meter and guard, with tests.
3. Pixal3D adapter with the stub-port tests.
4. One staged smoke run on the real port under the guard.
5. Bench: two eager and two staged runs, comparison and chart.
6. README and docs, with numbers from `summary.json`.
7. Review with the owner, then publish to GitHub.

## Done when

- Tests and ruff pass in CI.
- A staged `1024_cascade` run produces a GLB on this machine.
- The bench results are in `bench/results/`: peak footprint and time for both modes, and an
  output comparison against the eager-against-eager noise.
- The README explains the problem, how stageload works, how to install and run it, and its limits,
  using measured numbers only.
