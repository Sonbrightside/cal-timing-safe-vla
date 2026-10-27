"""
Step 3: while SmolVLA (lerobot/smolvla_libero) performs a LIBERO task, move a hazard
        (a mocap body) every simulation step.
  - The VLA cannot see the hazard (geom_group=3 -> hidden from the cameras).
  - A "local sensor" path (EE-hazard distance, contact) is read every step.
  - The model only looks at the world when it has used up its action chunk (50 steps)
    -> that chunk length is the floor of the staleness.
Output (in OUT/): vla_view.mp4, debug_view.mp4, log.csv

Usage:
    python run_smolva.py            # writes to ./test_examples/
    python run_smolva.py my_run     # writes to ./my_run/
Experiment settings (suite/task, hazard speed V, path, chunk handling) are constants below, not flags.
"""
import os, sys

# These two must be set BEFORE huggingface_hub is imported (it reads them at import time).
# setdefault = keep the value from env.sh if it is already exported; otherwise use this default.
os.environ.setdefault("HF_HOME", f"/scratch/{os.environ.get('USER', 'unknown')}/hf_cache")
# Offline mode: never touch the network, load the checkpoint from HF_HOME only.
# Sol compute nodes have no outbound internet, so the checkpoint must be downloaded once beforehand
# (see README, "Running"). Set HF_HUB_OFFLINE=0 in env.sh to allow downloads.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np, imageio
import torch
from scipy.spatial.transform import Rotation as R

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors

HERE = os.path.dirname(os.path.abspath(__file__))              # folder this script lives in
OUT  = os.path.join(HERE, sys.argv[1] if len(sys.argv) > 1 else "test_examples")
os.makedirs(OUT, exist_ok=True)
SUITE, TASK_ID = "libero_goal", 1

# =====================================================================
# 1. Load the policy: model weights + pre/post-processors (normalization stats)
# =====================================================================
CKPT   = "lerobot/smolvla_libero"          # SmolVLA fine-tuned on LIBERO (HF Hub repo id)
device = "cuda"
policy = SmolVLAPolicy.from_pretrained(CKPT).to(device).eval()
pre, post = make_pre_post_processors(
    policy.config, pretrained_path=CKPT,
    preprocessor_overrides={"device_processor": {"device": device}},
)
for k, v in policy.config.input_features.items():
    print("IN ", k, v.shape)
for k, v in policy.config.output_features.items():
    print("OUT", k, v.shape)
print("chunk_size =", policy.config.chunk_size, " n_action_steps =", policy.config.n_action_steps)
print(pre)   # if there is a rename step, the image -> camera1 mapping shows up here

# =====================================================================
# 2. Environment + hazard injection (same as diag_render.py)
# =====================================================================
suite = benchmark.get_benchmark_dict()[SUITE]()
task  = suite.get_task(TASK_ID)
bddl  = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
env.seed(0); env.reset()
# LIBERO builds the scene as MuJoCo XML; splice hazard_body.xml in before </worldbody> and reload.
xml = env.sim.model.get_xml().replace(
    "</worldbody>", open(f"{HERE}/hazard_body.xml").read() + "\n</worldbody>")
env.env.reset_from_xml_string(xml)
env.env.sim.reset()
env.set_init_state(suite.get_task_init_states(TASK_ID)[0])

sim  = env.sim
hid  = sim.model.body_name2id("hazard")
mid  = sim.model.body_mocapid[hid]            # hazard is a mocap body: moved by writing mocap_pos
gid  = sim.model.geom_name2id("hazard_geom")
sid  = sim.model.site_name2id("gripper0_grip_site")
robot_geoms = {i for i in range(sim.model.ngeom)
               if (sim.model.geom_id2name(i) or "").startswith(("robot0", "gripper0"))}

def ee_pos():      return sim.data.site_xpos[sid].copy()     # end-effector (gripper) position
def hazard_pos():  return sim.data.body_xpos[hid].copy()
def set_hazard(p): sim.data.mocap_pos[mid] = p

def hazard_contact():
    """True if any robot/gripper geom is currently in contact with the hazard geom."""
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        if gid in (c.geom1, c.geom2) and ({c.geom1, c.geom2} & robot_geoms):
            return True
    return False

def render_debug(cam="agentview"):
    """Third-person view WITH the hazard visible (for humans, never shown to the VLA)."""
    sim.model.geom_group[gid] = 1                                        # add to the drawn groups
    img = sim.render(width=256, height=256, camera_name=cam)[::-1, ::-1] # rotate 180 deg
    sim.model.geom_group[gid] = 3                                        # hide again
    return img

# =====================================================================
# 3. Observation -> model input (format conversion only; normalization is done by `pre`)
# =====================================================================
def quat_to_axisangle(q):                     # robosuite quaternions are (x, y, z, w)
    return R.from_quat(q).as_rotvec().astype(np.float32)

def to_batch(obs):
    def img(k):
        t = torch.from_numpy(obs[k][::-1].copy()).permute(2, 0, 1).float() / 255   # HWC uint8 -> CHW float
        return t.unsqueeze(0).to(device)
    state = np.concatenate([
        obs["robot0_eef_pos"],                     # 3  end-effector position
        quat_to_axisangle(obs["robot0_eef_quat"]), # 3  end-effector orientation (axis-angle)
        obs["robot0_gripper_qpos"],                # 2  gripper
    ]).astype(np.float32)                           # = 8
    return {
        "observation.images.camera1": img("agentview_image"),          # front camera
        "observation.images.camera2": img("robot0_eye_in_hand_image"), # wrist camera
        "observation.state": torch.from_numpy(state).unsqueeze(0).to(device),
        "task": [task.language],
    }

# =====================================================================
# 4. Hazard trajectory: at EE height, crossing in front of the EE from right to left
# =====================================================================
ee0 = ee_pos()
print("EE at reset:", ee0)
print("task:", task.language)
start = np.array([ 0.45, ee0[1] + 0.10, ee0[2]])   # 10 cm in front of the EE (y), same height
end   = np.array([-0.45, ee0[1] + 0.10, ee0[2]])
V, dt = 0.3, 1/20                                   # hazard speed [m/s], control period [s] (20 Hz)
n = int(np.linalg.norm(end - start) / V / dt)       # steps for the hazard to cross
T = max(n, 200)                                     # total steps (long enough for ~4 chunks)

# =====================================================================
# 5. Main loop: re-infer when the chunk is used up; read the local sensor every step
# =====================================================================
obs, _, _, _ = env.step(np.zeros(7))                # get a first obs (one zero-action step)

vla_frames, dbg_frames, log = [], [], []
chunk, i = None, 0

for t in range(T):
    set_hazard(start + min(t / n, 1.0) * (end - start))   # stays at `end` after reaching it

    # ---- No chunk, or chunk used up: infer a new one from the CURRENT obs (the moment the model sees the world) ----
    if chunk is None or i >= chunk.shape[0]:
        with torch.no_grad():
            chunk = policy.predict_action_chunk(pre(to_batch(obs)))[0]   # (50, 7), normalized space
        i = 0
        print(f"t={t:3d}  new chunk  d={np.linalg.norm(hazard_pos() - ee_pos()):.3f} m")

    # ---- i-th action of the chunk: de-normalize -> numpy -> env.step ----
    a = post(chunk[i:i+1]).squeeze(0).cpu().numpy()   # (7,)
    obs, r, done, info = env.step(a)
    i += 1

    # ---- Local sensor path (every step) ----
    d = np.linalg.norm(hazard_pos() - ee_pos())
    log.append((t * dt, d, float(hazard_contact())))
    vla_frames.append(obs["agentview_image"][::-1])
    dbg_frames.append(render_debug())

# =====================================================================
# 6. Save
# =====================================================================
imageio.mimsave(f"{OUT}/vla_view.mp4", vla_frames, fps=20)
imageio.mimsave(f"{OUT}/debug_view.mp4", dbg_frames, fps=20)
# log.csv columns: t [s], dist [m] (EE-hazard), contact (0/1)
np.savetxt(f"{OUT}/log.csv", np.array(log), delimiter=",", header="t,dist,contact", comments="")
log = np.array(log)
print(f"steps={T}  min dist={log[:,1].min():.3f} m at t={log[log[:,1].argmin(),0]:.2f}s  "
      f"contact steps={int(log[:,2].sum())}")
print("hazard start/end:", start, end)