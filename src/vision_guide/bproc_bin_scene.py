import blenderproc as bproc  # noqa: I001  (BlenderProc 규칙: 첫 줄이어야 한다)
"""PLAN_v5 S1 하우징 bin 장면 렌더링 — **Blender 내부에서만 실행** (``blenderproc run``). 합성 데이터.

호스트(``vision_guide.bin_v5``)가 만든 작업 파일(JSON)을 받아 장면마다 npz 하나를 쓴다.
numpy·blenderproc·bpy 외 의존 없음 (vision_guide를 import하지 않는다). pytest가 import하지 않는다.

좌표: Blender world = 로봇 base 좌표 (m). 빈 바닥 윗면 = z 0.
출력 ``{raw_path}`` (임시 파일 → 이름 바꾸기, 있으면 건너뜀):
    rgb uint8 (V,H,W,3), depth float32 (V,H,W; m, z-깊이, 없음 0), inst int16 (V,H,W; 하우징 번호, 그 밖 −1),
    T_base_obj float64 (S,4,4), cam2world_cv float64 (V,4,4; OpenCV 카메라 규약 T_base_cam), meta (JSON)
"""

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import bpy  # noqa: E402
import numpy as np  # noqa: E402

FLIP = np.diag([1.0, -1.0, -1.0, 1.0])   # OpenCV 카메라 ↔ Blender 카메라 (자기 역행렬)
MAX_HOUSINGS = 20


def _u(rng, lohi):
    return float(rng.uniform(float(lohi[0]), float(lohi[1])))


def make_static(job):
    """바닥(큰 판) + 빈(바닥 + 벽 4면). 빈 바닥 윗면이 z = 0."""
    b = job["bin"]
    cx, cy, _ = b["center_m"]
    sx, sy = b["inner_size_xy_m"]
    h, t = float(b["wall_height_m"]), float(b["wall_thickness_m"])
    parts = []
    ground = bproc.object.create_primitive("PLANE", scale=[2.0, 2.0, 1.0], location=[cx, cy, -0.02])
    ground.set_name("ground")
    floor = bproc.object.create_primitive("CUBE", scale=[sx / 2 + t, sy / 2 + t, 0.01], location=[cx, cy, -0.01])
    floor.set_name("bin_floor")
    parts.append(floor)
    walls = [([sx / 2 + t, t / 2, h / 2], [cx, cy - sy / 2 - t / 2, h / 2]),
             ([sx / 2 + t, t / 2, h / 2], [cx, cy + sy / 2 + t / 2, h / 2]),
             ([t / 2, sy / 2, h / 2], [cx - sx / 2 - t / 2, cy, h / 2]),
             ([t / 2, sy / 2, h / 2], [cx + sx / 2 + t / 2, cy, h / 2])]
    for i, (sc, loc) in enumerate(walls):
        w = bproc.object.create_primitive("CUBE", scale=sc, location=loc)
        w.set_name(f"bin_wall_{i}")
        parts.append(w)
    return ground, parts


def make_lights():
    plane = bproc.object.create_primitive("PLANE", scale=[0.6, 0.6, 1], location=[0.6, 0.0, 2.2])
    plane.set_name("light_plane")
    mat = bproc.material.create("light_material")
    plane.replace_materials(mat)
    point = bproc.types.Light()
    point.set_type("POINT")
    return plane, mat, point


def load_housings(job):
    hs = []
    src = bproc.loader.load_obj(job["housing_mesh"])[0]
    src.set_name("h_0")
    hs.append(src)
    for k in range(1, MAX_HOUSINGS):
        d = src.duplicate()
        d.set_name(f"h_{k}")
        hs.append(d)
    for k, o in enumerate(hs):
        o.set_location([10.0 + k, 10.0, 0.0])
        o.hide(True)
    return hs


def _material(o, name, color, roughness, metallic=0.0):
    mats = o.get_materials()
    if not mats:
        m = bproc.material.create(name)
        o.replace_materials(m)
        mats = [m]
    for m in mats:
        m.set_principled_shader_value("Base Color", [*color, 1.0])
        m.set_principled_shader_value("Roughness", float(roughness))
        m.set_principled_shader_value("Metallic", float(metallic))


def _rigid(o, active, shape, **kw):
    if o.has_rigidbody_enabled():
        o.disable_rigidbody()
    o.enable_rigidbody(active, collision_shape=shape, **kw)


def _unrigid(o):
    if o.has_rigidbody_enabled():
        o.disable_rigidbody()


def render_scene(job, sc, housings, static, lights, out_path):
    t0 = time.time()
    seed = int(sc["seed"])
    rng = np.random.default_rng(seed)
    np.random.seed(seed % (2 ** 32))
    random.seed(seed)
    ground, bin_parts = static
    plane, light_mat, point = lights
    n = int(sc["n_housings"])
    used = housings[:n]
    for k, o in enumerate(housings):
        _unrigid(o)
        if k >= n:
            o.hide(True)
            o.set_location([10.0 + k, 10.0, 0.0])
    hm = job["housing_material"]
    for o in used:
        o.hide(False)
        g = _u(rng, hm["base_gray"])
        _material(o, f"housing_mat_{o.get_name()}", [g, g, g * float(rng.uniform(0.95, 1.05))],
                  _u(rng, hm["roughness"]), _u(rng, hm["metallic"]))
    # 빈·바닥 재질 (무작위 색)
    bc = rng.uniform(0.1, 0.6, 3).tolist()
    for p in bin_parts:
        _material(p, f"bin_mat_{p.get_name()}", bc, float(rng.uniform(0.5, 0.9)))
    gc = float(rng.uniform(0.2, 0.6))
    _material(ground, "ground_mat", [gc, gc, gc], 0.9)

    b = job["bin"]
    cx, cy, _ = b["center_m"]
    dr = b["drop"]

    def sample_pose(o):
        o.set_location([cx + _u(rng, (-dr["xy_half_m"][0], dr["xy_half_m"][0])),
                        cy + _u(rng, (-dr["xy_half_m"][1], dr["xy_half_m"][1])), _u(rng, dr["z_range_m"])])
        o.set_rotation_euler(rng.uniform(0, 2 * np.pi, 3).tolist())

    bproc.object.sample_poses(used, sample_pose_func=sample_pose, objects_to_check_collisions=used, max_tries=1000)
    for p in bin_parts:
        _rigid(p, False, "BOX", friction=0.8)
    ph = b["physics"]
    for o in used:
        _rigid(o, True, ph["collision_shape"], mass=0.4, friction=0.6, linear_damping=0.5, angular_damping=0.5)
    bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=float(ph["min_simulation_time"]),
                                                      max_simulation_time=float(ph["max_simulation_time"]),
                                                      check_object_interval=float(ph["check_object_interval"]))
    for o in used + bin_parts:
        _unrigid(o)
    t_phys = time.time() - t0

    # 조명
    li = job["lighting"][sc["lighting"]]
    plane.set_location([cx, cy, float(li["ceiling"]["z_m"])])
    s = float(li["ceiling"]["size_m"])
    plane.set_scale([s / 2, s / 2, 1.0])
    light_mat.make_emissive(emission_strength=_u(rng, li["ceiling"]["strength"]),
                            emission_color=[*rng.uniform(0.85, 1.0, 3).tolist(), 1.0])
    pr, pz = _u(rng, li["point"]["radius_m"]), _u(rng, li["point"]["z_m"])
    pa = rng.uniform(0, 2 * np.pi)
    point.set_location([cx + pr * np.cos(pa), cy + pr * np.sin(pa), pz])
    point.set_energy(_u(rng, li["point"]["energy_w"]))
    point.set_color(rng.uniform(0.85, 1.0, 3).tolist())
    bproc.renderer.set_world_background(rng.uniform(0.6, 1.0, 3).tolist(), strength=_u(rng, li["world_strength"]))

    # 카메라 (호스트가 OpenCV 규약 T_base_cam 목록을 준다)
    bproc.utility.reset_keyframes()
    cams = [np.asarray(M, float) for M in sc["cam2world_cv"]]
    for M in cams:
        bproc.camera.add_camera_pose(M @ FLIP)
    t_setup = time.time() - t0
    data = bproc.renderer.render()
    t_render = time.time() - t0 - t_setup

    names = {o.get_name(): k for k, o in enumerate(used)}
    rgb = np.stack([np.asarray(c, dtype=np.uint8)[..., :3] for c in data["colors"]])
    depth = np.stack([np.asarray(d, dtype=np.float32) for d in data["depth"]])
    depth[~np.isfinite(depth) | (depth > float(job["render"]["max_depth_m"]))] = 0.0
    inst = np.full(depth.shape, -1, dtype=np.int16)
    for f in range(len(cams)):
        seg = np.asarray(data["instance_segmaps"][f])
        for item in data["instance_attribute_maps"][f]:
            k = names.get(str(item.get("name")))
            if k is not None:
                inst[f][seg == int(item["idx"])] = k
    T_base_obj = np.stack([np.asarray(o.get_local2world_mat(), dtype=float) for o in used])
    meta = {"scene_id": int(sc["scene_id"]), "split": sc["split"], "render_set": sc.get("render_set"),
            "seed": seed, "n_housings": n, "lighting": sc["lighting"], "t_phys_s": t_phys, "t_setup_s": t_setup,
            "t_render_s": t_render, "t_total_s": time.time() - t0, "label": "합성 (BlenderProc, 합성 하우징 CAD)"}
    tmp = out_path + ".tmp.npz"
    np.savez_compressed(tmp, rgb=rgb, depth=depth, inst=inst, T_base_obj=T_base_obj,
                        cam2world_cv=np.stack(cams), meta=json.dumps(meta))
    os.replace(tmp, out_path)
    print(f"[bproc_bin] {sc['split']} {sc.get('render_set')} scene {sc['scene_id']} n={n} -> {out_path} "
          f"(phys {t_phys:.1f}s, render {t_render:.1f}s)", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--job", required=True)
    args = p.parse_args([a for a in sys.argv[1:] if a != "--"])
    with open(args.job, "r", encoding="utf-8") as fh:
        job = json.load(fh)
    todo = [sc for sc in job["scenes"] if not os.path.exists(sc["raw_path"])]
    if not todo:
        print("[bproc_bin] nothing to do", flush=True)
        return
    bproc.init()
    r = job["render"]
    bproc.renderer.set_render_devices(desired_gpu_device_type=[r.get("device", "CUDA")])
    bproc.renderer.set_max_amount_of_samples(int(r["max_samples"]))
    bproc.renderer.set_noise_threshold(float(r["noise_threshold"]))
    bproc.renderer.set_denoiser(r.get("denoiser", "INTEL"))
    bproc.renderer.set_output_format(enable_transparency=False)
    K = np.asarray(job["K"], float).copy()
    K[0, 2] -= 0.5   # 이 프로젝트: 픽셀 중심 = (u+0.5) → BlenderProc(OpenCV, 중심 = 정수)
    K[1, 2] -= 0.5
    bproc.camera.set_intrinsics_from_K_matrix(K, int(job["width"]), int(job["height"]))
    housings = load_housings(job)
    static = make_static(job)
    lights = make_lights()
    bproc.renderer.enable_depth_output(activate_antialiasing=bool(r.get("depth_antialiasing", False)))
    bproc.renderer.enable_segmentation_output(map_by=["instance", "name"])
    for sc in todo:
        render_scene(job, sc, housings, static, lights, sc["raw_path"])


main()
