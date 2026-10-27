# stageload

Run multi-stage PyTorch pipelines on Apple Silicon with only the current stage's weights in memory.

Many PyTorch pipelines run several models one after another: a text-to-image pipeline runs a text
encoder, then a denoiser, then an image decoder. stageload loads each model when its stage first
uses it and frees its weights as soon as the pipeline moves to a stage that does not need it. The
package also has two commands for a Mac that runs heavy jobs: `stageload guard` starts a command
only when enough memory is free and stops it if swap grows past a set budget, and `stageload run`
queues heavy commands so they run one at a time.

Measured on a 48 GB M5 Pro, staged loading (stageload) against each pipeline's own eager loading.
Each run went through `stageload guard`, stageload's memory guard, which would stop it once swap
had grown by 8 GB after it started ([Results](#results) has the tables and traces):

- Pixal3D-mac, the Apple Silicon port of the Pixal3D image-to-3D model (`1024_cascade` pipeline):
  eager loading (the port's default, every model built at the start) reached 44 GB in the texture
  stage and added 12 GB of swap, so the run was stopped there. With staged loading both runs
  finished in about ten minutes and peaked at 30 GB while models were loaded; the port's mesh
  export, after every weight is freed, goes higher.
- Qwen-Image-2.1 through diffusers, 1024 x 1024: eager loading (the whole pipeline moved to the
  GPU) and diffusers' `enable_model_cpu_offload` added 8 and 9.6 GB of swap and were stopped in
  the VAE decode, before either wrote an image. Staged loading peaked at 19 GB, and swap did not
  grow in either of its two runs.

## Quick start

```sh
pip install stageload
```

```python
from stageload import StagedModels

models = StagedModels(loaders, stages={"encode": ["encoder"], "decode": ["decoder"]})
models.enter("encode"); latent = models["encoder"](image)  # the encoder is loaded here
models.enter("decode"); out = models["decoder"](latent)    # and freed here
```

`loaders` maps each model's name to a function that builds the model and returns it. [Use it in
your own pipeline](#use-it-in-your-own-pipeline) has a version you can run, and `on_call` adds the
`enter` calls to a pipeline whose code you do not own.

![Peak footprint per stage of Pixal3D on a 48 GB Mac, eager loading against staged loading](https://raw.githubusercontent.com/nefayran/stageload/main/bench/results/2026-10-05/stages.png)

## Why

On Apple Silicon the CPU and the GPU share one pool of memory. The usual way to fit a big
pipeline, keeping idle models "on the CPU" and moving the active one to the GPU, frees nothing
there: both copies sit in the same RAM. Pixal3D-mac runs in that mode by default (`low_vram` is on
unless its `pipeline.json` turns it off), so a `1024_cascade` run keeps all seven of its models
in memory for the whole run and copies the active one to the GPU (MPS) on top. It also loads the
same DINOv3 backbone four times, once per feature extractor.

stageload loads a model the first time a stage asks for it and releases it in place when the
pipeline moves to a stage that does not need it. Releasing moves the module's tensors to PyTorch's
`meta` device, which frees their storage even while other code still holds a reference to the
module. The design is in [docs/design/2026-10-04-stageload-design.md](https://github.com/nefayran/stageload/blob/main/docs/design/2026-10-04-stageload-design.md).

## Use it in your own pipeline

Nothing in the core knows about Pixal3D. A pipeline needs three things: a function that builds
each model on its own, the list of models each stage uses, and a call where each stage begins.
In code you own, that call is `models.enter("decode")`; in code you don't, `on_call` attaches it
to a method that already runs at that point, and the pipeline's code stays as it is. A
text-to-image pipeline also runs its models in turn (a text encoder, a denoiser and an image
decoder): [`bench/qwen_image.py`](https://github.com/nefayran/stageload/blob/main/bench/qwen_image.py)
runs the diffusers Qwen-Image-2.1 pipeline this way without changing its code
([results](#qwen-image-21-through-diffusers)). The example below shows the core on two models, and
the test suite runs it as written:

```python
import torch
from safetensors.torch import load_file
from stageload import StagedModels

DEVICE = "mps"

def loader(cls, path):             # cls: one of your nn.Module classes
    def load():
        with torch.device(DEVICE):  # built on the GPU, not on the CPU and then moved
            model = cls()
        model.load_state_dict(load_file(path, device=DEVICE))
        return model.eval()
    return load

models = StagedModels(
    {"encoder": loader(Encoder, "encoder.safetensors"),
     "decoder": loader(Decoder, "decoder.safetensors")},
    stages={"encode": ["encoder"], "decode": ["decoder"]},
)

models.enter("encode")
latent = models["encoder"](image)   # loaded here
models.enter("decode")              # the encoder is released here
mesh = models["decoder"](latent)
models.close()
```

stageload ships one ready-made adapter, for Pixal3D-mac ([below](#pixal3d-mac)).

## How it works

- `StagedModels` is a dictionary of models that loads each one on first access.
  `enter(stage)` releases every loaded model the stage does not list. Checking `name in models`,
  `len()` and iterating names never load anything.
- Loading a model leaves the random generators where they were: torch's CPU, MPS and CUDA
  generators, Python's `random` and NumPy's global one. A pipeline that seeds once and draws noise
  stage by stage would otherwise get different noise, and a different result, as soon as a model
  is built in the middle of the run.
- `release(module)` moves every parameter and buffer to `meta`, collects garbage and empties the
  MPS cache. Calling a released module raises `ReleasedModuleError` with its name and the stage
  that released it, instead of PyTorch's error about meta tensors.
- `on_call(obj, method, before=..., after=...)` marks stage boundaries on an existing pipeline
  object, so its own `run()` executes unchanged.
- `share(cls, "from_pretrained")` makes identical loads return one instance for the duration of
  a block; the Pixal3D adapter uses it to give four extractors one DINOv3 backbone.
- `MemoryMeter` writes a JSONL trace: the process footprint (`phys_footprint`, the number
  Activity Monitor shows), swap and `kern.memorystatus_level` every 0.5 s, plus every load,
  release and stage change. `stageload summary TRACE` prints a trace's summary as JSON, with the
  peak footprint of each stage.

## Results

Every run below was its own process under `stageload guard --wait-free 40 --swap-budget 8G`
([the memory guard](#the-memory-guard-stageload-guard)): it started once at least 40 % of memory
was available, and it would be stopped once swap had grown by 8 GB over its level at the start, or
if available memory stayed under 10 %. "Stopped" in the tables below, and "stopped by the guard"
in the charts, mean that happened. Peak footprint is the process's memory as Activity Monitor
shows it, including memory that macOS has compressed.

### Pixal3D

Pixal3D `1024_cascade` on an Apple M5 Pro with 48 GB, one image, seed 7, two runs per mode
([details and traces](https://github.com/nefayran/stageload/blob/main/bench/results/2026-10-05/README.md)).
Eager is the port's own loading; staged is `stageload-pixal3d`.

| stage | eager, peak footprint | staged, peak footprint |
|---|---|---|
| setup | 20.4 GB | 2.9 GB |
| camera | 23.2 GB | 6.7 GB |
| structure | 22.2 GB | 10.9 GB |
| shape_512 | 26.6 GB | 11.4 to 11.5 GB |
| shape_1024 | 30.5 GB | 13.1 GB |
| texture | stopped at 44.2 GB (a separate run on 2026-10-04) | 30.1 GB |
| decode | – | 17.9 to 18.4 GB |
| export | – | 69.6 to 70.2 GB |

- Eager loading did not get through the texture stage on this machine. Within 10 s of entering
  it the footprint rose from 27 to 44 GB, available memory fell to 6 % and swap reached 27 GB,
  12 GB more than when the run started, so the run was stopped. The other eager numbers come
  from two runs set to stop where the texture stage begins (`--stop-at texture`). Both staged runs
  finished, in about ten minutes each, of which 24 s went to loading models.
- Most of the texture stage's 30 GB is working memory: the texture model itself is 2.6 GB. When
  decoding began, releasing it and emptying the MPS allocator's cache brought the footprint down
  to 4.5 GB in one run and 8.0 GB in the other within two seconds.
- The highest staged footprint is the port's own mesh export, after every weight has been
  released; stageload does not change it.
- Output: the first latent (the sparse structure) is the same bit for bit in every run. From the
  next stage on, Pixal3D-mac does not repeat itself on MPS even with the same seed: in the 512
  shape latent, the largest difference was 1.77 between the two eager runs and 1.24 to 1.86
  between a staged and an eager run. That shows any change from staged loading is no larger than
  the port's own variation; it cannot show there is none.

A staged run on a character image, rendered in Blender ([`bench/render_glb.py`](https://github.com/nefayran/stageload/blob/main/bench/render_glb.py)):

![The input image and the generated model](https://raw.githubusercontent.com/nefayran/stageload/main/docs/images/input-output.png)

The footprint over time, for an eager run set to stop where the texture stage begins, the eager
run stopped in the texture stage, and a staged run:

![Process footprint over time: eager loading against staged loading](https://raw.githubusercontent.com/nefayran/stageload/main/bench/results/2026-10-05/memory.png)

### Qwen-Image-2.1 through diffusers

Text to image at 1024 x 1024, 20 steps, seed 7, on the same Mac
([details and traces](https://github.com/nefayran/stageload/blob/main/bench/results/2026-10-06-qwen-image/README.md)). Eager is the usual
`QwenImage21Pipeline.from_pretrained(...).to("mps")`. Staged keeps the VAE and loads the text
encoder (16.3 GB) and the transformer (13.3 GB) one at a time; the pipeline's code is unchanged.

![Peak footprint per stage of Qwen-Image-2.1 on a 48 GB Mac, eager loading against staged loading](https://raw.githubusercontent.com/nefayran/stageload/main/bench/results/2026-10-06-qwen-image/stages.png)

| stage | eager, peak footprint | staged, peak footprint |
|---|---|---|
| load | 31.4 GB | 2.1 GB |
| encode | 31.4 GB | 19.0 GB |
| denoise | 32.5 GB | 16.2 to 18.6 GB |
| decode | stopped at 43.6 GB | 14.4 GB |

- Eager ran all 20 steps. The VAE decode then took its footprint from 32.5 to 43.6 GB and swap
  grew by 8 GB, so the run was stopped before it wrote an image. Staged released the transformer
  before the decode, and its swap did not grow in either of its two runs.
- Model CPU offload did not finish either. The same run with
  `pipe.enable_model_cpu_offload(device="mps")` (one run, 2026-10-08,
  [trace](https://github.com/nefayran/stageload/blob/main/bench/results/2026-10-08-qwen-image-offload/README.md))
  kept the process footprint at 18.5 GB, but swap grew by 9.6 GB and the run was stopped during
  the VAE decode, at the same point as eager. Footprint does not count pages already in swap, so
  a footprint-only comparison would make offload look like staged.
- Counted from the moment the transformer was in memory to the start of the decode, the 20 steps
  took 70 s and 99 s in the two staged runs and 77.5 s for eager; the traces do not show why the
  staged runs differ. Staged also spent about 13 s per run loading the two models and 4 to 5 s
  releasing the transformer.
- The two staged runs produced the same image, bit for bit. Eager produced none, so these runs do
  not show that both modes give the same image.

## Pixal3D-mac

Install [Pixal3D-mac](https://github.com/pawel-mazurkiewicz/Pixal3D-mac) with its own README, then
add stageload to the port's environment without touching its packages, and run it from there:

```bash
cd path/to/Pixal3D-mac
.venv/bin/pip install --no-deps git+https://github.com/nefayran/stageload
.venv/bin/stageload-pixal3d photo.png -o photo.glb
```

`--load eager` runs the port's own loading for comparison. The flags, the stages and the
benchmark are described in [docs/pixal3d.md](https://github.com/nefayran/stageload/blob/main/docs/pixal3d.md).

## The memory guard: `stageload guard`

`stageload guard` runs one heavy command: it starts the command only when there is room for it,
and stops it before swap runs away. It needs no root, and the only processes it ever stops are the
command and its process group.

```bash
stageload guard --wait-free 40 --swap-budget 8G --busy 'llama-server' -- python train.py
```

It waits until macOS reports at least `--wait-free` percent of memory available
(`kern.memorystatus_level`) and no process has matched a `--busy` pattern for `--quiet-for`, then
runs the command; the pattern above makes it wait while a llama.cpp server is running.
`--quiet-for` is a minute by default, so the short gap between two jobs of a script that runs
them one after another does not count as quiet. It stops the command if swap grows more
than `--swap-budget` over what was used at the start, or if available memory stays under
`--kill-free` (10 %) for two checks in a row. Stopping means SIGTERM to the command's process
group and, five seconds later, SIGKILL to whatever of the group is left.

Exit code 3 means the guard stopped the command, 4 that it never found room to start it, and 127
that the command could not be started. If the guard itself gets Ctrl-C, SIGTERM or SIGHUP, it
stops the command first and exits with 128 plus the signal number; a second signal while the
command is shutting down sends SIGKILL at once. A signal the guard was started to ignore, such as
SIGHUP under `nohup`, stays ignored. Otherwise you get the command's own exit code, or 128 plus
the signal number if a signal ended it.

The `--busy` patterns match whole command lines, like `pgrep -f`. The guard leaves itself and the
shells that started it out, but any other process whose command line contains a pattern counts,
a shell running a script that mentions it included.

For a safety net across the whole machine, see
[macos-oom-guard](https://github.com/fl4p/macos-oom-guard), a root daemon that kills the largest
process before the kernel panics. The two do different jobs and can run together.

## One queue for the machine: `stageload run`

Each `stageload guard` decides on its own, and that goes wrong in two ways. Two guards that find
free memory at the same moment both start their commands. And a script that runs heavy jobs one
after another leaves short pauses between them: a guard that looks during such a pause starts its
command, and the script's next job then needs the same memory. In one case that next job pushed
swap past the budget of the command that had started in the pause within 20 s. When several
people, scripts or agents share one Mac, start the heavy commands with `stageload run` instead.
It takes the guard's flags:

```bash
stageload run --who ana-train -- python train.py
```

- The command waits in a lane: a named line of commands that run one at a time. The lane is `ram`
  unless `--lane` names another, and the queue lives under `STAGELOAD_HOME` (`~/.stageload` by
  default). The command starts once the commands ahead of it in its lane have finished and it
  holds the lane's lock; then the guard's memory checks run as usual. Lanes do not wait for each
  other, so jobs held back by something other than memory, such as a remote API's quota, can have
  a lane of their own (`--lane api`) and not wait behind the ones that need memory.
- Turns are fair between owners. A run's owner is `--owner` (or `$STAGELOAD_OWNER`), else the part
  of `--who` before its first dash, so `ana-train` belongs to `ana`. The owner who started a
  command least recently goes next, and each owner's own commands keep the order they joined in.
  An owner who queues a batch takes turns with the others instead of holding the lane until the
  batch is done.
- The command inherits the lane's lock, so the lane stays taken while it runs even if the process
  that queued it is killed with SIGKILL.
- A waiting run does not show the command in its own command line: `stageload run` writes it to
  a file and hands over, with the same pid, to `python -m stageload.queue <file>`. Otherwise a
  guard whose `--busy` pattern matches the command would take the waiting run for the running
  program, and the two would wait for each other.
- Patterns in `~/.stageload/busy.txt`, one per line, are added to the `--busy` of every run in the
  `ram` lane, so all of them wait for the same heavy programs started outside the queue. Other
  lanes skip the list; a lane that needs no memory should also pass `--wait-free 0`.
- Each run writes a memory trace (by default under `~/.stageload/traces`; `--trace PATH`,
  `--no-trace`) with the footprint of the command and of every process under it, and appends one
  line to `~/.stageload/history.jsonl`: who, exit code, seconds waited and run, peak footprint and
  swap.
- `--who` defaults to `$STAGELOAD_WHO`, `--start-timeout` to 24 hours and `--quiet-for` to 20 s,
  since waiting in the queue is the normal case. Exit codes are the guard's.

`stageload queue` shows what runs and what waits; `--history 10` adds the last ten finished runs,
`--json` prints JSON.

```
$ stageload queue
ram     0 running  ana-train            pid 4242    since 2026-10-07T12:01:00+00:00  python train.py
ram     1 waiting  ben-render           pid 4310    since 2026-10-07T12:03:12+00:00  blender -b scene.blend -a
```

## Limits

- The memory meter, the guard and the queue read macOS counters (`libproc`, `sysctl`).
  `StagedModels` and `release` work wherever PyTorch runs.
- One adapter so far: Pixal3D-mac, tested at commit `0be9e69`, and only its `1024_cascade`
  pipeline has a staged plan (the models each stage needs).
- A staged Pixal3D pipeline serves one image; the command builds a new one per image.
- The Qwen-Image setup is a benchmark script for one pipeline, run with diffusers from main
  (0.41.0.dev0); there is no packaged adapter for it.
- stageload builds Pixal3D's models directly on MPS, where the port builds them on the CPU. The
  sparse attention's rotary frequencies, computed when a model is built, then come out slightly
  different (up to 6e-8 apart). Because the port does not repeat its sparse stages on MPS with the
  same seed, a staged result can only be checked against eager within that run-to-run spread,
  and the texture latent and the final GLB were not compared at all.
- The meter samples from a thread inside the process. While native code holds the interpreter
  lock it takes no samples, so a short peak inside such a call can be missed.

## License

MIT. Pixal3D and Pixal3D-mac are MIT-licensed as well and are not included here; the adapter
imports the port at run time.
