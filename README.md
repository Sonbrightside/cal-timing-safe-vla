# Timing-Safe Execution of Stale VLA Plans Under Moving Hazards

CSE 598 Intelligent & Safe CPS (Prof. Hokeun Kim), Fall 2026 — course project.

**Team:** Hongju Son, Phil Hogan, Sunyoung Kim

## What this project is about

A VLA policy (SmolVLA) controls a robot arm in simulation (LIBERO / robosuite / MuJoCo).
VLA inference is slow, so by the time an action chunk is executed it was planned from
sensor data that is already *stale*. We add a **moving hazard** to the scene and study
how long that staleness can be tolerated before the robot gets dangerously close to it.

The quantity we derive is a **state-dependent staleness bound** `B(d, v)`: given the
current distance `d` to the hazard and its speed `v`, how old may the data behind the
currently executing plan be before we must stop or re-plan. The bound comes from the
measured timing of the whole pipeline (sensing → inference → control), in the spirit of
speed-and-separation monitoring (ISO/TS 15066), and is enforced with a Lingua Franca
style deadline check rather than "always stop when a hazard appears".

Planned comparison (not all implemented yet):

1. fixed staleness bound
2. state-dependent bound `B(d, v)`
3. (2) + a second SmolVLA instance that takes over on hazard
4. (2) + MPPI fallback on hazard

## Repository layout

Scripts you run:

| Path | What it is |
|---|---|
| `run_smolva.py` | **Main entry point.** Loads a LIBERO task via `OffScreenRenderEnv`, injects the moving hazard into the scene XML (reads `hazard_body.xml`), runs SmolVLA on it, and writes `log.csv` + two mp4 files |
| `profile_smolvla.py` | **Timing profile, no hazard.** Same policy / env / obs-conversion code as `run_smolva.py`, but no hazard, no local sensor, no video. Records wall-clock time of every pipeline stage per step and the inference-latency distribution. This is where the injected delivery delay `D` comes from. See "Timing profile" under Running |
| `job_profile.sh` | SLURM batch script that runs `profile_smolvla.py` on all 10 `libero_goal` tasks inside **one** job (same node, same GPU) |
| `diag_render.py` | Rendering check. Builds the same hazard scene and renders one frame each through MuJoCo and robosuite. Run this first on a new machine — if it fails, the problem is EGL/GPU, not our code |
| `hazard_body.xml` | MuJoCo `<body>` definition of the moving hazard. `run_smolva.py` and `diag_render.py` splice this into LIBERO's scene XML right before `</worldbody>` |

Everything else is results or reference material:

| Path | What it is |
|---|---|
| `test_examples/` | Example output of `run_smolva.py` so you can check your own run against it: `log.csv` (per-step log), `vla_view.mp4` (what the policy saw), `debug_view.mp4` (third-person view with the hazard drawn) |
| `diag_mujoco.png`, `diag_robosuite.png` | Example output of `diag_render.py` — what a working render looks like |
| `scene_original.xml`, `scene_with_hazard.xml` | **Reference only.** LIBERO's scene XML dumped before/after hazard injection, for inspecting the scene structure. Not used at runtime |
| `env.sh.example` | Template for `env.sh` (see Setup step 4) |

Python deps beyond lerobot's extras: `scipy`, `imageio` — both already pulled in by lerobot/LIBERO, nothing extra to install.

## Setup

Everyone installs their own environment. Nothing in this repo depends on anyone else's
home or scratch directory. Requirements: **Linux** (LIBERO does not support macOS/Windows),
a GPU for SmolVLA inference, ~10 GB of disk for the env + checkpoint.

The steps below were tested on ASU Sol with lerobot at commit `e0d50211` (main, 110 commits after v0.6.1;
reports itself as 0.6.2). Other versions of lerobot change the LIBERO/SmolVLA APIs, so pin that commit.

### 0. (Sol) Get a GPU compute node first

```bash
# Request an interactive shell on a GPU node. Flags:
#   -p public            partition (node group) open to all users
#   -q htc               QoS: "high-throughput" queue, meant for shorter jobs — usually starts faster
#   --gres=gpu:a100:1    1x NVIDIA A100
#   -c 8                 8 CPU cores
#   --mem=32G            32 GB RAM
#   -t 0-04:00           time limit, days-hours:minutes (4 hours here; the shell is killed when it runs out)
#   -C a100_80           constraint: the 80 GB A100 variant (not the 40 GB one)
interactive -p public -q htc --gres=gpu:a100:1 -c 8 --mem=32G -t 0-04:00 -C a100_80
```


### 1. Create a conda environment

```bash
# On Sol, load conda first:
module load mamba/latest        # (or the conda module name Sol currently uses)

# Python version: lerobot 0.6.x requires Python >= 3.12 (0.5.x and older used 3.10 — that will NOT work here).
# This project is tested only on 3.12 (3.12.14 on Sol). 3.13+ is untested and some binary deps
# (torch, mujoco, robosuite) may lag behind, so stick to 3.12.
#
# Location: `-p <path>` creates the env at that exact path. (`-n <name>` would put it under ~/.conda/envs,
# i.e. in your home directory, regardless of your current directory — and Sol home quotas are small.
# The env is several GB, so keep it on /scratch.)
conda create -p /scratch/$USER/envs/lerobot-libero python=3.12 -y

# An env made with -p is activated by its path, not by a name.
conda activate /scratch/$USER/envs/lerobot-libero

# ffmpeg is a standalone video encoder (not a Python package, so pip can't install it).
# lerobot calls it to write the rollout videos (*.mp4). Without it the run works but video saving fails.
conda install -c conda-forge ffmpeg -y
```


### 2. Install lerobot from source with LIBERO + SmolVLA extras

```bash
cd /scratch/$USER            # or wherever you keep source checkouts
git clone https://github.com/huggingface/lerobot.git
cd lerobot

# Pin lerobot to the exact commit this project was developed on.
# `git clone` gives you today's `main`, which changes weekly and breaks the LIBERO/SmolVLA API.
# This commit is 110 commits after the v0.6.1 tag (`git describe --tags` → v0.6.1-110-ge0d50211);
# `pip show lerobot` reports it as 0.6.2 (the in-development version number).
git checkout e0d50211

# What this one line does:
#   .                   install the package in the current folder; pip reads lerobot's pyproject.toml
#   -e                  "editable": link to this source folder instead of copying into site-packages,
#                       so edits under lerobot/src/ take effect without reinstalling
#   [libero,smolvla]    extras = optional dependency groups defined in pyproject.toml
#                       ([project.optional-dependencies]). Core deps (torch, numpy, ...) are always
#                       installed; these two groups add LIBERO/robosuite/MuJoCo and the SmolVLA deps.
#   pip then resolves the chain automatically: lerobot -> hf-libero -> robosuite -> mujoco.
#   Everything lands in the conda env's site-packages, NOT in this lerobot/ folder.
pip install -e ".[libero,smolvla]"
```

In more detail, this one line does all of the following (it takes a while — torch alone is several GB):

- pip then resolves the whole dependency chain automatically. For our stack that is:

  ```
  lerobot  (policy code: SmolVLA, eval loop, env adapters)      ← Hugging Face
    └─ [libero] extra
         └─ hf-libero   (LIBERO benchmark: tasks, scenes, Franka Panda)   ← UT Austin
              └─ robosuite   (robot/table/camera assembly on top of MuJoCo)  ← ARISE
                   └─ mujoco   (physics engine)                          ← DeepMind
    └─ [smolvla] extra
         └─ transformers, num2words, ...   (SmolVLM backbone)
  ```

  You only ask for the top; pip fetches everything below it.
- Everything lands in the conda env (`.../lerobot-libero/lib/python3.12/site-packages/`), not in
  the `lerobot/` folder. The `lerobot/` folder stays pure source code.

**Where things come from, for orientation:**

| Thing | What it is | Where it comes from |
|---|---|---|
| lerobot | Framework: policy models (SmolVLA, ...), datasets, train/eval scripts | GitHub `huggingface/lerobot`, pinned to commit `e0d50211` |
| LIBERO | Benchmark: 130 manipulation tasks in 4 suites (`libero_spatial/object/goal/10`), Franka Panda in sim | PyPI `hf-libero`, pulled in by the `[libero]` extra |
| robosuite / MuJoCo | Simulation framework + physics engine under LIBERO. Our `hazard_body.xml` / `scene_with_hazard.xml` are MuJoCo XML | pulled in by `hf-libero` |
| SmolVLA **code** | Model architecture, preprocessing, inference loop | inside lerobot |
| SmolVLA **weights** | ~1 GB `.safetensors` checkpoint (`lerobot/smolvla_libero`, fine-tuned on LIBERO) | Hugging Face Hub, downloaded once into `$HF_HOME` (on /scratch) — never committed to this repo |

Quick check that both extras are present:

```bash
python -c "import lerobot, robosuite, mujoco, transformers, num2words; print('ok', lerobot.__file__)"
```

Expected: `ok /scratch/<you>/lerobot/src/lerobot/__init__.py`. A `[robosuite WARNING] It is recommended to use
a private macro file` line before it is normal — it is a suggestion, not an error; ignore it.

### 3. Hugging Face login

This step does **not** download anything. It stores your Hub token on this machine so that
the one-time checkpoint download (see "Running") is authenticated.

```bash
# huggingface_hub is already installed by step 2 (it is a core lerobot dependency);
# this only makes sure it is current. It also provides the `hf` command.
pip install -U huggingface_hub

# Saves your token to ~/.cache/huggingface/token. Every Hub request from this machine uses it from now on.
# Get a token (read access is enough) from https://huggingface.co/settings/tokens
hf auth login
```

Why log in at all: `lerobot/smolvla_libero` is public, but unauthenticated downloads are
rate-limited, some models/datasets are gated, and pushing your own fine-tuned checkpoint
later requires it.

The checkpoint itself is **not** fetched automatically: `run_smolva.py` runs in Hub-offline
mode (Sol compute nodes have no internet), so you download it once from the **login node**
with `hf download lerobot/smolvla_libero` — the exact steps are under "Running" below. It
lands in `$HF_HOME`, which `env.sh` puts on `/scratch`.

### 4. Clone this repo and set the environment variables

```bash
cd /scratch/$USER
git clone https://github.com/Sonbrightside/cal-timing-safe-vla.git
cd cal-timing-safe-vla
cp env.sh.example env.sh      # then edit env.sh: env path, lerobot path
source env.sh
```

`env.sh` loads conda, activates the env, sets `MUJOCO_GL=egl` (headless GPU rendering — use `osmesa` if you have no GPU),
and points `HF_HOME` at `/scratch`.
### 5. Check that rendering works

```bash
python diag_render.py
```

It should write `diag_mujoco.png` and `diag_robosuite.png` without errors. If you see
EGL / OpenGL errors on Sol, you are probably on the login node — go back to step 0 and get a GPU node.


## Running

### One-time: download the checkpoint on the login node

Sol **compute nodes have no outbound internet**, and `run_smolva.py` runs in Hub-offline mode
(`HF_HUB_OFFLINE=1`) so it never tries. So the checkpoint has to be fetched once from the
**login node** (which does have internet):

```bash
# on the login node, after `source env.sh`
hf download lerobot/smolvla_libero          # ~1 GB, goes into $HF_HOME
```

### Hazard run

```bash
# on a GPU node (README step 0), after `source env.sh`
python run_smolva.py            # writes to ./test_examples/
python run_smolva.py my_run     # writes to ./my_run/  (pass a folder name so you don't overwrite test_examples)
```

There are no other flags. Experiment settings are constants near the top of the script:

| Constant | Meaning | Current value |
|---|---|---|
| `SUITE, TASK_ID` | LIBERO suite and task index | `libero_goal`, 1 |
| `CKPT` | policy checkpoint (HF Hub id) | `lerobot/smolvla_libero` — SmolVLA fine-tuned on LIBERO |
| `start`, `end` | hazard path (crosses 10 cm in front of the end-effector, right to left) | x = +0.45 → −0.45 m |
| `V` | hazard speed | 0.3 m/s |
| `dt` | control period | 1/20 s (20 Hz) |
| `T` | total steps | max(crossing time, 200) |

What a run does: the VLA plans in chunks of 50 actions and **only observes the world when a chunk
is used up** (that is the staleness floor); the hazard is hidden from the cameras (`geom_group=3`)
and moved every step; a "local sensor" reads EE–hazard distance and contact every step.

Outputs in the chosen folder:

| File | Contents |
|---|---|
| `log.csv` | one row per step: `t` [s], `dist` [m] (EE–hazard), `contact` (0/1) |
| `vla_view.mp4` | the front camera **as the policy saw it** (hazard invisible) |
| `debug_view.mp4` | same camera with the hazard drawn (for humans) |

The console also prints when each new chunk is inferred and the min distance / contact count at the end.

### Timing profile (no hazard): `profile_smolvla.py`

Measures the wall-clock time of each SmolVLA inference call while the policy runs plain
LIBERO episodes in sync mode (sim frozen during inference, exactly like `run_smolva.py`).
No hazard, no local sensor, no video — video encoding would pollute the timing.

```bash
# on a GPU node, after `source env.sh`
python profile_smolvla.py <out_dir> <task_id>
python profile_smolvla.py profile_sg048/task3 3      # e.g. task 3, output under ./profile_sg048/task3/
```

- `<out_dir>` is created under the script directory; nested paths are fine.
- `<task_id>` is a `libero_goal` task index (0–9). Defaults to 1 if omitted.
- Other settings are constants at the top of the file: `N_EPISODES = 20`, `MAX_STEPS = 300`,
  `WARMUP_CHUNKS = 2` (the first 2 inference calls are excluded from the summary — CUDA warm-up).

Outputs in `<out_dir>`:

| File | Contents |
|---|---|
| `chunks.csv` | one row per inference call: `t_pre_ms` (obs → batch, incl. move to GPU), `t_inf_ms` (pure `predict_action_chunk`, GPU-synchronized) |
| `steps.csv` | one row per sim step: all stage timings, index in chunk, plan age in steps |
| `summary.json` | `node`, `gpu`, `gpu_sms` (SM count — full A100 = 108, MIG slice fewer), torch version, task, `t_inf_ms` p50/p95/p99/max, `t_inf_in_steps_p99` = ceil(p99 / 50 ms) (the value used as `D`), sync success rate |

**All tasks on one node (Sol).** `job_profile.sh` loops over tasks 0–9 inside one SLURM job, so
every task is measured on the same node and GPU and task-to-task differences are not confounded
with node differences:

```bash
sbatch job_profile.sh        # submit 2–3 times to sample different nodes
squeue -u $USER              # NODELIST: each submission should land on a different node
```

Output goes to `profile_<jobid>/task<N>/`, console log to `logs/prof_<jobid>.out`.
**One folder = one node, one subfolder = one task.**

**Jetson (Orin Nano / Orin AGX).** The script needs no changes. Setup steps 1–4 apply as
written, with these differences:

1. Use the JetPack build of PyTorch (ARM + CUDA); the pip wheel will not work. Check
   `torch.cuda.is_available()` before installing lerobot.
2. The Jetson has internet, so download the checkpoint directly on it (`hf download lerobot/smolvla_libero`
   after `source env.sh`). The script still runs with `HF_HUB_OFFLINE=1`, so this must happen first.
3. `env.sh`: point `HF_HOME` at a disk with ≥ 2 GB free (not `/scratch` — that is a Sol path); keep `MUJOCO_GL=egl`,
   fall back to `osmesa` if EGL fails. Run `python diag_render.py` to check rendering.
4. Smoke test first: set `N_EPISODES = 2` in the file, run `python profile_smolvla.py jetson_test 1`,
   and check `summary.json` has `gpu_sms` = 16 (Orin AGX) or 8 (Orin Nano).
5. Full run: `N_EPISODES = 20`, loop tasks 0–9 (same `for` loop as in `job_profile.sh`, without the `#SBATCH` lines).
   Expect it to be much slower than on Sol — inference is 5–10× longer and MuJoCo renders on the CPU.
6. **Record the power mode and clock state next to the results**: `sudo nvpmodel -q` and whether
   `jetson_clocks` was on. Inference time on Jetson depends strongly on both.

**Reading the results.** Compare `t_inf_ms` across tasks on one node first (SmolVLA pads the
instruction to a fixed length and uses fixed image sizes, so task-to-task variation is expected
to be small), then across nodes/devices — that is the variability `D` has to absorb. Do not merge
runs with different `gpu_sms` or power modes into one distribution.

## Status / next steps

- [x] Step 0: SmolVLA runs on LIBERO on Sol, baseline success rates recorded (results not in repo)
- [x] Moving hazard inserted into the LIBERO scene
- [x] Hazard-scene rollouts with per-step logging (`test_examples/`)
- [ ] Measure pipeline timing (sensing / inference / control) distribution — `profile_smolvla.py` done; Sol multi-node and Jetson runs in progress
- [ ] Derive and implement `B(d, v)` + deadline check
- [ ] Four-way comparison (fixed vs state-dependent vs handoff vs MPPI fallback)
- [ ] Test generation / falsification with psy-taliro
- [ ] Final report & presentation (Dec 10)

## Notes

- Large artifacts (model checkpoints, LIBERO datasets, big video sets) are **not** committed. See `.gitignore`.
  This includes `profile_*/` and `logs/` — `steps.csv` is ~6,000 rows per task.
- Do not commit tokens. `env.sh` should reference `HF_TOKEN` etc. from your shell, not contain them.
