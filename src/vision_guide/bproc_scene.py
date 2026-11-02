import blenderproc as bproc  # noqa: I001  (BlenderProc 규칙: 첫 줄이어야 한다)
"""v4.2-A 합성 장면 렌더링 — **Blender 내부에서만 실행** (``blenderproc run``). 합성 데이터.

호스트(``vision_guide.synth``)가 만든 작업 파일(JSON: 설정 + 장면 목록)을 받아 장면마다 npz 하나를 쓴다.
numpy·blenderproc·bpy 외 의존 없음 (vision_guide를 import하지 않는다). 이 파일은 pytest가 import하지 않는다.

출력 ``raw/{split}_{scene:06d}.npz`` (임시 파일 → 이름 바꾸기, 있으면 건너뜀 = 장면 단위 재개):
    rgb uint8 (N,H,W,3), depth float32 (N,H,W; m, z-깊이, 하늘 0), inst int16 (N,H,W; 슬롯 번호, 배경 −1),
    slot_obj_ids int32 (S,), slot_is_target bool (S,), T_cam_obj float64 (N,S,4,4; OpenCV 카메라 규약, m),
    cam2world float64 (N,4,4; Blender 규약), K float64 (3,3; 원래 K = datasets.py 규약), meta (JSON 문자열)

좌표 규약
- Blender 카메라는 −Z를 보고 +Y가 위다. OpenCV 규약(+Z 앞, +Y 아래)으로 바꾸려면 오른쪽에 diag(1,−1,−1) 곱:
  T_cam_obj = diag(1,−1,−1,1) · inv(cam2world) · obj2world.
- BlenderProc ``set_intrinsics_from_K_matrix``는 픽셀 중심 = 정수 좌표(OpenCV) 규약이다. 이 프로젝트 로더는
  픽셀 (u, v) 중심 = K 좌표 (u+0.5, v+0.5) 규약이므로 cx−0.5, cy−0.5를 넘긴다. 저장 K는 원래 값.
  호스트 ``synth.check_scene``이 정답 자세 레이캐스팅과 마스크를 비교해 규약 오류를 잡는다.
"""

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import bpy  # noqa: E402
import numpy as np  # noqa: E402

FLIP = np.diag([1.0, -1.0, -1.0, 1.0])


def _u(rng, lohi):
    return float(rng.uniform(float(lohi[0]), float(lohi[1])))


def load_objects(job):
    """대상·방해물 모델을 한 번만 읽는다 (mm → m 스케일을 메시에 반영)."""
    s = float(job["objects"]["unit_scale"])
    slots = []
    for kind in ("targets", "distractors"):
        spec = job["objects"][kind]
        for oid in spec["obj_ids"]:
            path = os.path.join(spec["root_abs"], f"obj_{int(oid):06d}.ply")
            o = bproc.loader.load_obj(path)[0]
            o.set_scale([s, s, s])
            o.persist_transformation_into_mesh(location=False, rotation=False, scale=True)
            name = f"{'tgt' if kind == 'targets' else 'dis'}_{int(oid)}"
            o.set_name(name)
            o.set_location([100.0 + len(slots), 100.0, 0.0])
            o.hide(True)
            slots.append({"name": name, "obj_id": int(oid), "is_target": kind == "targets", "obj": o})
    return slots


def make_room(job):
    sc = job["scene"]
    H, wh = float(sc["floor_half_m"]), float(sc["wall_height_m"])
    hp = np.pi / 2
    planes = [bproc.object.create_primitive("PLANE", scale=[H, H, 1]),
              bproc.object.create_primitive("PLANE", scale=[H, wh / 2, 1], location=[0, -H, wh / 2], rotation=[-hp, 0, 0]),
              bproc.object.create_primitive("PLANE", scale=[H, wh / 2, 1], location=[0, H, wh / 2], rotation=[hp, 0, 0]),
              bproc.object.create_primitive("PLANE", scale=[wh / 2, H, 1], location=[H, 0, wh / 2], rotation=[0, -hp, 0]),
              bproc.object.create_primitive("PLANE", scale=[wh / 2, H, 1], location=[-H, 0, wh / 2], rotation=[0, hp, 0])]
    for i, p in enumerate(planes):
        p.set_name(f"room_{i}")
    lc = sc["lights"]["ceiling_plane"]
    sz = float(lc["size_m"])
    light_plane = bproc.object.create_primitive("PLANE", scale=[sz, sz, 1], location=[0, 0, float(lc["z_m"])])
    light_plane.set_name("light_plane")
    light_mat = bproc.material.create("light_material")
    light_plane.replace_materials(light_mat)
    point = bproc.types.Light()
    point.set_type("POINT")
    return planes, light_plane, light_mat, point


def background_material(rng, sc):
    """절차적 텍스처(noise/voronoi/wave) + 무작위 두 색 (외부 텍스처 없음)."""
    mat = bpy.data.materials.new("bg_material")
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    kind = sc["background"]["kinds"][int(rng.integers(len(sc["background"]["kinds"])))]
    if kind == "noise":
        tex = nodes.new("ShaderNodeTexNoise")
        out = tex.outputs["Fac"]
    elif kind == "voronoi":
        tex = nodes.new("ShaderNodeTexVoronoi")
        out = tex.outputs["Distance"]
    else:
        tex = nodes.new("ShaderNodeTexWave")
        out = tex.outputs["Fac"]
    tex.inputs["Scale"].default_value = _u(rng, sc["background"]["scale"])
    ramp = nodes.new("ShaderNodeValToRGB")
    for el in ramp.color_ramp.elements:
        c = rng.uniform(0.0, 1.0, 3)
        el.color = (float(c[0]), float(c[1]), float(c[2]), 1.0)
    links.new(out, ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = float(rng.uniform(0.4, 1.0))
    return mat, kind


def _set_rigid(o, active, shape, **kw):
    if o.has_rigidbody_enabled():
        o.disable_rigidbody()
    o.enable_rigidbody(active, collision_shape=shape, **kw)


def _drop_rigid(o):
    if o.has_rigidbody_enabled():
        o.disable_rigidbody()


def render_scene(job, sc, slots, room, out_path):
    t0 = time.time()
    rng = np.random.default_rng(int(sc["seed"]))
    np.random.seed(int(sc["seed"]) % (2 ** 32))
    random.seed(int(sc["seed"]))
    ob, cam, sce = job["objects"], job["camera"], job["scene"]
    planes, light_plane, light_mat, point = room

    # 등장 물체
    tg = [s for s in slots if s["is_target"]]
    ds = [s for s in slots if not s["is_target"]]
    present = [s for s in tg if rng.random() < float(ob["targets"]["p_present"])]
    if not present:
        present = [tg[int(rng.integers(len(tg)))]]
    n_d = int(rng.integers(int(ob["distractors"]["n_range"][0]), int(ob["distractors"]["n_range"][1]) + 1))
    dsel = [ds[i] for i in sorted(rng.choice(len(ds), size=min(n_d, len(ds)), replace=False))]
    used = present + dsel
    used_names = {s["name"] for s in used}
    for k, s in enumerate(slots):
        o = s["obj"]
        _drop_rigid(o)
        if s["name"] not in used_names:
            o.hide(True)
            o.set_location([100.0 + k, 100.0, 0.0])
    # 재질 무작위
    mcfg = ob["material"]
    for s in used:
        o = s["obj"]
        o.hide(False)
        for m in o.get_materials():
            m.set_principled_shader_value("Roughness", _u(rng, mcfg["roughness"]))
            m.set_principled_shader_value("Specular IOR Level", _u(rng, mcfg["specular_ior_level"]))
    # 초기 자세 (충돌 검사) → 물리 낙하
    dr = ob["drop"]

    def sample_pose(o):
        o.set_location([_u(rng, (-dr["xy_half"], dr["xy_half"])), _u(rng, (-dr["xy_half"], dr["xy_half"])),
                        _u(rng, dr["z_range"])])
        o.set_rotation_euler(rng.uniform(0, 2 * np.pi, 3).tolist())

    objs = [s["obj"] for s in used]
    bproc.object.sample_poses(objs, sample_pose_func=sample_pose, objects_to_check_collisions=objs, max_tries=500)
    for p in planes:
        _set_rigid(p, False, "BOX", friction=100.0, linear_damping=0.99, angular_damping=0.99)
    for o in objs:
        _set_rigid(o, True, ob["physics"]["collision_shape"], mass=1.0, friction=100.0,
                   linear_damping=0.99, angular_damping=0.99)
    ph = ob["physics"]
    bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=float(ph["min_simulation_time"]),
                                                      max_simulation_time=float(ph["max_simulation_time"]),
                                                      check_object_interval=float(ph["check_object_interval"]))
    for o in objs + planes:
        _drop_rigid(o)
    t_phys = time.time() - t0

    # 배경·조명
    old = [m for m in bpy.data.materials if m.name.startswith("bg_material")]
    mat, bg_kind = background_material(rng, sce)
    for p in planes:
        p.blender_obj.data.materials.clear()
        p.blender_obj.data.materials.append(mat)
    for m in old:
        bpy.data.materials.remove(m)
    li = sce["lights"]
    light_mat.make_emissive(emission_strength=_u(rng, li["ceiling_plane"]["strength"]),
                            emission_color=[*rng.uniform(0.7, 1.0, 3).tolist(), 1.0])
    pr, pz = _u(rng, li["point"]["radius_m"]), _u(rng, li["point"]["z_m"])
    pa = rng.uniform(0, 2 * np.pi)
    point.set_location([pr * np.cos(pa), pr * np.sin(pa), pz])
    point.set_energy(_u(rng, li["point"]["energy_w"]))
    point.set_color(rng.uniform(0.6, 1.0, 3).tolist())
    wc = rng.uniform(0.3, 1.0, 3).tolist()
    bproc.renderer.set_world_background(wc, strength=_u(rng, li["world_strength"]))

    # 카메라
    bproc.utility.reset_keyframes()
    centers = np.array([s["obj"].get_location() for s in present])
    poi0 = centers.mean(axis=0)
    cam2worlds = []
    for _v in range(int(cam["views_per_scene"])):
        poi = poi0 + rng.uniform(-cam["poi_jitter_m"], cam["poi_jitter_m"], 3) * np.array([1.0, 1.0, 0.3])
        d = _u(rng, cam["distance_m"])
        el, az = np.deg2rad(_u(rng, cam["elevation_deg"])), rng.uniform(0, 2 * np.pi)
        loc = poi + d * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        R = bproc.camera.rotation_from_forward_vec(poi - loc, inplane_rot=_u(rng, cam["inplane_rot_rad"]))
        M = bproc.math.build_transformation_mat(loc, R)
        bproc.camera.add_camera_pose(M)
        cam2worlds.append(np.asarray(M, dtype=float))
    t_setup = time.time() - t0

    data = bproc.renderer.render()
    t_render = time.time() - t0 - t_setup

    N = len(cam2worlds)
    S = len(used)
    name2slot = {s["name"]: k for k, s in enumerate(used)}
    rgb = np.stack([np.asarray(c, dtype=np.uint8)[..., :3] for c in data["colors"]])
    depth = np.stack([np.asarray(d, dtype=np.float32) for d in data["depth"]])
    depth[~np.isfinite(depth) | (depth > float(job["render"]["max_depth_m"]))] = 0.0
    inst = np.full(depth.shape, -1, dtype=np.int16)
    for f in range(N):
        seg = np.asarray(data["instance_segmaps"][f])
        for item in data["instance_attribute_maps"][f]:
            k = name2slot.get(str(item.get("name")))
            if k is not None:
                inst[f][seg == int(item["idx"])] = k
    o2w = np.stack([np.asarray(s["obj"].get_local2world_mat(), dtype=float) for s in used])
    T = np.zeros((N, S, 4, 4))
    for f, M in enumerate(cam2worlds):
        T[f] = FLIP @ np.linalg.inv(M) @ o2w
    meta = {"scene_id": int(sc["scene_id"]), "split": sc["split"], "seed": int(sc["seed"]),
            "slots": [s["name"] for s in used], "bg_kind": str(bg_kind),
            "t_phys_s": t_phys, "t_setup_s": t_setup, "t_render_s": t_render, "t_total_s": time.time() - t0,
            "label": "합성 (BlenderProc)"}
    tmp = out_path + ".tmp.npz"
    np.savez_compressed(tmp, rgb=rgb, depth=depth, inst=inst,
                        slot_obj_ids=np.array([s["obj_id"] for s in used], np.int32),
                        slot_is_target=np.array([s["is_target"] for s in used], bool), T_cam_obj=T,
                        cam2world=np.stack(cam2worlds), K=np.asarray(cam["K"], float), obj2world=o2w,
                        meta=json.dumps(meta))
    os.replace(tmp, out_path)
    print(f"[bproc_scene] {sc['split']} scene {sc['scene_id']} -> {out_path} "
          f"(phys {t_phys:.1f}s, render {t_render:.1f}s, total {time.time() - t0:.1f}s)", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--job", required=True)
    args = p.parse_args([a for a in sys.argv[1:] if a != "--"])
    with open(args.job, "r", encoding="utf-8") as fh:
        job = json.load(fh)
    todo = [sc for sc in job["scenes"] if not os.path.exists(sc["raw_path"])]
    if not todo:
        print("[bproc_scene] nothing to do", flush=True)
        return
    bproc.init()
    r = job["render"]
    bproc.renderer.set_render_devices(desired_gpu_device_type=[r.get("device", "CUDA")])
    bproc.renderer.set_max_amount_of_samples(int(r["max_samples"]))
    bproc.renderer.set_noise_threshold(float(r["noise_threshold"]))
    bproc.renderer.set_denoiser(r.get("denoiser", "INTEL"))
    bproc.renderer.set_output_format(enable_transparency=False)
    cam = job["camera"]
    K = np.asarray(cam["K"], float).copy()
    K[0, 2] -= 0.5
    K[1, 2] -= 0.5
    bproc.camera.set_intrinsics_from_K_matrix(K, int(cam["width"]), int(cam["height"]))
    slots = load_objects(job)
    room = make_room(job)
    bproc.renderer.enable_depth_output(activate_antialiasing=bool(r.get("depth_antialiasing", False)))
    bproc.renderer.enable_segmentation_output(map_by=["instance", "name"])
    for sc in todo:
        render_scene(job, sc, slots, room, sc["raw_path"])


main()
