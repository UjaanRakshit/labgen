"""Inspect the Newton model Isaac Lab built for the bench task: are the pads colliders?"""
import sys
sys.path.insert(0, "/mnt/c/Ujaan Docx/Research/labgen")
import argparse
from isaaclab.app import add_launcher_args, launch_simulation
p = argparse.ArgumentParser(); add_launcher_args(p); args = p.parse_args()
import gymnasium as gym, numpy as np, torch
import labgen_tasks  # noqa
from labgen_tasks.bench_yam import BenchYamIkRelEnvCfg
cfg = BenchYamIkRelEnvCfg()
with launch_simulation(cfg, args):
    env = gym.make("Isaac-LabBench-YAM-IK-Rel-v0", cfg=cfg)
    env.reset()
    from isaaclab_newton.physics.newton_manager import NewtonManager as NM
    m = NM._model
    labels = list(m.shape_label) if hasattr(m, "shape_label") else [str(i) for i in range(m.shape_count)]
    flags = m.shape_flags.numpy(); grp = m.shape_collision_group.numpy(); body = m.shape_body.numpy()
    typ = m.shape_type.numpy(); scale = m.shape_scale.numpy()
    ke = m.shape_material_ke.numpy(); kd = m.shape_material_kd.numpy(); mu = m.shape_material_mu.numpy()
    for i, l in enumerate(labels):
        if any(k in l for k in ("box", "beaker", "bench", "tip", "geom")):
            print(f"{i:3d} {l[-60:]:60s} body {body[i]:3d} type {typ[i]} flags {flags[i]} grp {grp[i]} "
                  f"scale {np.round(scale[i],4)} ke {ke[i]:.0f} kd {kd[i]:.1f} mu {mu[i]:.2f}")
    idx = m._shape_sdf_index.numpy() if hasattr(m._shape_sdf_index, "numpy") else m._shape_sdf_index
    for i, l in enumerate(labels):
        if "Scene" in l:
            print(f"   sdf index {l.split('/')[-2]:10s} {idx[i]}")
    src = m.shape_source[8] if hasattr(m, "shape_source") else None
    print("   beaker mesh source:", type(src).__name__, "sdf" if getattr(src, "sdf", None) is not None else "NO sdf attached",
          "verts", getattr(src, "vertices", np.zeros((0,))).shape if src is not None else None)
    print("shape_contact_pairs:", getattr(m, "shape_contact_pair_count", None))
    env.close()
