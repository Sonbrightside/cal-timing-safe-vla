"""
Step 4 (baseline): profile the SmolVLA pipeline on LIBERO with NO hazard.
  - Same policy / env / obs-conversion code as run_smolva.py, but no hazard body,
    no local sensor, no video (video rendering would pollute the timing).
  - sync mode, exactly like run_smolva.py: the sim is frozen while the model runs,
    so wall-clock inference time never enters sim time. We measure it anyway, because
    later we will inject it back into sim time as an action-delivery delay.
  - Every step gets one row with wall-clock timings of each pipeline stage and the
    staleness (age) of the plan the robot is following, in sim steps.

Output (in OUT/):
  steps.csv    one row per sim step
  chunks.csv   one row per inference call
  summary.json percentiles of T_inf, wall vs sim time, GPU name, config

Usage:
    python profile_smolvla.py              # writes to ./profile_baseline/
    python profile_smolvla.py my_run       # writes to ./my_run/
    python profile_smolvla.py out_dir 3     # task 3, writes to ./out_dir/
"""
import os, sys, time, json, platform

os.environ.setdefault("HF_HOME", f"/scratch/{os.environ.get('USER', 'unknown')}/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, sys.argv[1] if len(sys.argv) > 1 else "profile_baseline")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------
# Experiment settings (constants, not flags — same style as run_smolva.py)
# ---------------------------------------------------------------------
SUITE = "libero_goal"
TASK_ID = int(sys.argv[2]) if len(sys.argv) > 2 else 1
N_EPISODES     = 20          # episodes to run
MAX_STEPS      = 300         # steps per episode (300 steps = 6 chunks of 50 = 15 s sim time)
STOP_ON_DONE   = False       # True: stop the episode when LIBERO reports success
DT             = 1 / 20      # control period T_c [s]  (LIBERO runs at 20 Hz)
WARMUP_CHUNKS  = 2           # first N inference calls of the whole run are excluded from the summary
SEED           = 0

# =====================================================================
# 1. Policy
# =====================================================================
CKPT   = "lerobot/smolvla_libero"
device = "cuda" if torch.cuda.is_available() else "cpu"
policy = SmolVLAPolicy.from_pretrained(CKPT).to(device).eval()
pre, post = make_pre_post_processors(
    policy.config, pretrained_path=CKPT,
    preprocessor_overrides={"device_processor": {"device": device}},
)
CHUNK = policy.config.chunk_size
print("chunk_size =", CHUNK, " n_action_steps =", policy.config.n_action_steps, "(n_action_steps unused: we run the full chunk)")

def sync():
    """Wait for the GPU to finish everything queued so far. Needed for correct wall-clock timing."""
    if device == "cuda":
        torch.cuda.synchronize()

# =====================================================================
# 2. Environment (no hazard)
# =====================================================================
suite = benchmark.get_benchmark_dict()[SUITE]()
task  = suite.get_task(TASK_ID)
bddl  = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
env   = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
init_states = suite.get_task_init_states(TASK_ID)
print("task:", task.language, "| init states available:", len(init_states))

# =====================================================================
# 3. Observation -> model input (identical to run_smolva.py)
# =====================================================================
def quat_to_axisangle(q):
    return R.from_quat(q).as_rotvec().astype(np.float32)

def to_batch(obs):
    def img(k):
        t = torch.from_numpy(obs[k][::-1].copy()).permute(2, 0, 1).float() / 255
        return t.unsqueeze(0).to(device)
    state = np.concatenate([
        obs["robot0_eef_pos"],
        quat_to_axisangle(obs["robot0_eef_quat"]),
        obs["robot0_gripper_qpos"],
    ]).astype(np.float32)
    return {
        "observation.images.camera1": img("agentview_image"),
        "observation.images.camera2": img("robot0_eye_in_hand_image"),
        "observation.state": torch.from_numpy(state).unsqueeze(0).to(device),
        "task": [task.language],
    }

# =====================================================================
# 4. Main loop
# =====================================================================
step_rows, chunk_rows = [], []
chunk_id_global = 0                 # counts inference calls across the whole run (for warmup exclusion)
wall_run0 = time.perf_counter()

for ep in range(N_EPISODES):
    env.seed(SEED + ep)
    env.reset()
    env.set_init_state(init_states[ep % len(init_states)])
    obs, _, _, _ = env.step(np.zeros(7))          # one zero-action step to get a first obs

    chunk, i, t_obs, chunk_id = None, 0, -1, -1
    success = False
    wall_ep0 = time.perf_counter()

    for t in range(MAX_STEPS):
        wall_step0 = time.perf_counter()
        ran_inference = False
        t_pre = t_inf = t_post = t_env = 0.0

        # ---- (a) chunk used up -> infer a new one from the current obs ----
        if chunk is None or i >= CHUNK:
            ran_inference = True
            t_obs = t                              # the obs below describes the world at the start of step t

            sync(); w0 = time.perf_counter()
            batch = pre(to_batch(obs))             # format conversion + normalization + move to GPU
            sync(); w1 = time.perf_counter()
            with torch.no_grad():
                chunk = policy.predict_action_chunk(batch)[0]   # (CHUNK, 7), normalized
            sync(); w2 = time.perf_counter()

            t_pre, t_inf = w1 - w0, w2 - w1
            i = 0
            chunk_id += 1
            chunk_id_global += 1
            chunk_rows.append(dict(episode=ep, chunk_id=chunk_id, chunk_id_global=chunk_id_global,
                                   t_obs=t_obs, t_pre_ms=t_pre * 1e3, t_inf_ms=t_inf * 1e3))
            print(f"ep={ep:2d} t={t:3d} chunk={chunk_id:2d}  T_inf={t_inf*1e3:7.1f} ms  (pre {t_pre*1e3:5.1f} ms)")

        # ---- (b) i-th action of the chunk -> de-normalize -> env.step ----
        sync(); w3 = time.perf_counter()
        a = post(chunk[i:i+1]).squeeze(0).cpu().numpy()
        sync(); w4 = time.perf_counter()
        obs, r, done, info = env.step(a)           # sim advances by one T_c; renders camera obs
        w5 = time.perf_counter()
        t_post, t_env = w4 - w3, w5 - w4

        age = t - t_obs                            # sim steps since the obs this action is based on (0 .. CHUNK-1)
        step_rows.append(dict(
            episode=ep, step=t, sim_time_s=t * DT,
            chunk_id=chunk_id, idx_in_chunk=i, t_obs=t_obs, age_steps=age, age_s=age * DT,
            ran_inference=int(ran_inference),
            t_pre_ms=t_pre * 1e3, t_inf_ms=t_inf * 1e3, t_post_ms=t_post * 1e3, t_env_ms=t_env * 1e3,
            t_step_wall_ms=(time.perf_counter() - wall_step0) * 1e3,
        ))
        i += 1
        if done:
            success = True
            if STOP_ON_DONE:
                break

    wall_ep = time.perf_counter() - wall_ep0
    n_steps = step_rows[-1]["step"] + 1
    print(f"--- ep={ep} success={success} steps={n_steps} sim={n_steps*DT:.1f}s wall={wall_ep:.1f}s")
    for row in step_rows:
        if row["episode"] == ep:
            row["success"] = int(success)

wall_run = time.perf_counter() - wall_run0

# =====================================================================
# 5. Save + summary
# =====================================================================
def write_csv(path, rows):
    keys = list(rows[0].keys())
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")

write_csv(f"{OUT}/steps.csv", step_rows)
write_csv(f"{OUT}/chunks.csv", chunk_rows)

inf = np.array([c["t_inf_ms"] for c in chunk_rows if c["chunk_id_global"] > WARMUP_CHUNKS])
warm = np.array([c["t_inf_ms"] for c in chunk_rows if c["chunk_id_global"] <= WARMUP_CHUNKS])
env_ms = np.array([s["t_env_ms"] for s in step_rows])
sim_total = len(step_rows) * DT

def pct(x, p): return float(np.percentile(x, p)) if len(x) else None

summary = dict(
    node=platform.node(),
    gpu_sms=torch.cuda.get_device_properties(0).multi_processor_count if device == "cuda" else None,
    gpu=torch.cuda.get_device_name(0) if device == "cuda" else platform.processor(),
    torch=torch.__version__, device=device, ckpt=CKPT, suite=SUITE, task_id=TASK_ID, task=task.language,
    chunk_size=CHUNK, dt_s=DT, n_episodes=N_EPISODES, max_steps=MAX_STEPS,
    n_inference_calls=len(chunk_rows), warmup_excluded=WARMUP_CHUNKS,
    warmup_t_inf_ms=warm.tolist(),
    t_inf_ms=dict(p50=pct(inf, 50), p95=pct(inf, 95), p99=pct(inf, 99), max=float(inf.max()), min=float(inf.min())),
    t_inf_in_steps_p99=int(np.ceil(pct(inf, 99) / (DT * 1e3))),
    t_env_ms=dict(p50=pct(env_ms, 50), p99=pct(env_ms, 99), max=float(env_ms.max())),
    sim_time_total_s=sim_total, wall_time_total_s=wall_run,
    age_steps_max=int(max(s["age_steps"] for s in step_rows)),
    success_rate=float(np.mean([s["success"] for s in step_rows if s["idx_in_chunk"] == 0 and s["chunk_id"] == 0])),
)
with open(f"{OUT}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== summary ===")
print(json.dumps({k: v for k, v in summary.items() if k != "warmup_t_inf_ms"}, indent=2))
print(f"\nS_vla (worst case, sync + injected T_inf) = T_c + ceil(T_inf_p99/T_c)*T_c + CHUNK*T_c "
      f"= {DT*1e3:.0f} + {summary['t_inf_in_steps_p99']*DT*1e3:.0f} + {CHUNK*DT*1e3:.0f} ms "
      f"= {(1 + summary['t_inf_in_steps_p99'] + CHUNK)*DT*1e3:.0f} ms")