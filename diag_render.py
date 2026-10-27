"""
diag_render.py — rendering sanity check for the hazard scene.

Run this first on any new machine. It answers three questions:
  1. Does the hazard body actually get injected into the LIBERO scene?
  2. Does the robosuite renderer show it (geom group 0)?
  3. Does the raw MuJoCo renderer show it (all geom groups)?

Expected output: two images, diag_robosuite.png and diag_mujoco.png, each with a
visible red hazard in front of the gripper, and a non-zero "red pixels" count for each.
If this script fails with EGL / OpenGL errors, you are not on a GPU node (see README step 0).
"""
import os, numpy as np, mujoco, imageio
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

# Folder this script lives in — hazard_body.xml is read from here, images are written here.
HERE = os.path.dirname(os.path.abspath(__file__))

# --- Build a stock LIBERO task (libero_goal, task 1) ---------------------------------
suite = benchmark.get_benchmark_dict()["libero_goal"]()
task  = suite.get_task(1)
bddl  = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
env.seed(0); env.reset()

# --- Inject the hazard -----------------------------------------------------------------
# LIBERO builds the scene as MuJoCo XML. We take that XML, splice hazard_body.xml in
# right before </worldbody>, and reload the sim from the modified XML.
# run_smolva.py does exactly the same thing.
xml = env.sim.model.get_xml().replace("</worldbody>", open(f"{HERE}/hazard_body.xml").read() + "\n</worldbody>")
env.env.reset_from_xml_string(xml); env.env.sim.reset()
env.set_init_state(suite.get_task_init_states(1)[0])

sim = env.sim
gid = sim.model.geom_name2id("hazard_geom"); hid = sim.model.body_name2id("hazard")
mid = sim.model.body_mocapid[hid]        # hazard is a mocap body: we move it by setting mocap_pos

# Park the hazard 10 cm in front of the gripper so it is guaranteed to be in view.
sid = sim.model.site_name2id("gripper0_grip_site")
sim.data.mocap_pos[mid] = sim.data.site_xpos[sid] + np.array([0, 0.10, 0])
env.step(np.zeros(7))
print("hazard xpos:", sim.data.body_xpos[hid])

# --- (1) Does a write through robosuite's model wrapper reach the raw MuJoCo model? ----
# robosuite wraps mujoco's MjModel. If the wrapper copies arrays instead of sharing them,
# setting geom_group here would not affect what the renderer sees.
print("wrapper type:", type(sim.model), "| has _model:", hasattr(sim.model, "_model"))
sim.model.geom_group[gid] = 0
raw_model = getattr(sim.model, "_model", None)
print("geom_group via wrapper:", sim.model.geom_group[gid],
      "| via raw:", raw_model.geom_group[gid] if raw_model is not None else "n/a")
print("geom_rgba :", sim.model.geom_rgba[gid])

# --- (2) Is the hazard inside the camera's field of view? Print camera info. -----------
cam_id = sim.model.camera_name2id("agentview")
print("agentview cam pos:", sim.model.cam_pos[cam_id], " fovy:", sim.model.cam_fovy[cam_id])
print("cam world pos (data):", sim.data.cam_xpos[cam_id])

def count_red(img):
    """Number of strongly red pixels — a crude 'is the hazard visible' metric."""
    f = img.astype(int); return int(((f[...,0]>150)&(f[...,1]<80)&(f[...,2]<80)).sum())

# --- (3) robosuite renderer, hazard in geom group 0 -------------------------------------
# [::-1, ::-1] flips the image: robosuite returns it upside-down and mirrored.
img_a = sim.render(width=256, height=256, camera_name="agentview")[::-1, ::-1]
print("robosuite render, group=0 -> red pixels:", count_red(img_a))
imageio.imwrite(f"{HERE}/diag_robosuite.png", img_a)

# --- (4) Raw MuJoCo renderer, bypassing robosuite, with all geom groups enabled ---------
# robosuite's renderer only draws some geom groups. Rendering directly with mujoco.Renderer
# and geomgroup[:] = 1 shows the hazard regardless of its group — this tells apart
# "the hazard is not in the scene" from "the hazard is there but robosuite hides it".
sim.model.geom_group[gid] = 3
raw_data = getattr(sim.data, "_data", None)
if raw_model is not None and raw_data is not None:
    r = mujoco.Renderer(raw_model, height=256, width=256)
    opt = mujoco.MjvOption(); opt.geomgroup[:] = 1          # show every geom group
    r.update_scene(raw_data, camera="agentview", scene_option=opt)
    img_b = r.render()
    print("mujoco.Renderer, all groups -> red pixels:", count_red(img_b))
    imageio.imwrite(f"{HERE}/diag_mujoco.png", img_b)
else:
    # The wrapper's private attribute is not named _model/_data in this robosuite version.
    # Print what is available so the names can be fixed.
    print("raw model/data attribute is not named _model/_data ->", [a for a in dir(sim.model) if not a.startswith('__')][:30])