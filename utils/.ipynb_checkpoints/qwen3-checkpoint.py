import os
import json
import math
import argparse
import subprocess
from pathlib import Path
from collections import deque

import torch
import torchvision


# ============================================================
# argparse: register qwen-related args
# ============================================================
def add_qwen3_args(parser):
    group = parser.add_argument_group("Qwen3 ROI")

    # 开关与阶段性刷新
    group.add_argument("--enable_qwen_roi", action="store_true", default=False)
    group.add_argument("--qwen_region_dir", type=str, default="")
    group.add_argument("--qwen_start_iter", type=int, default=3000)
    group.add_argument("--qwen_update_iters", nargs="+", type=int, default=[3000, 5000, 8000])

    # ROI loss 超参数
    group.add_argument("--lambda_qwen_roi", type=float, default=0.40)
    group.add_argument("--qwen_geo_weight", type=float, default=1.00)
    group.add_argument("--qwen_blur_weight", type=float, default=0.60)
    group.add_argument("--qwen_decay_tau", type=float, default=2500.0)
    group.add_argument("--qwen_box_dilate", type=int, default=6)

    # 外部 qwen3vl 环境
    group.add_argument("--qwen_conda_exe", type=str, default="/root/miniconda3/bin/conda")
    group.add_argument("--qwen_conda_env", type=str, default="qwen3vl")
    group.add_argument("--qwen_runner_py", type=str, default="./utils/qwen3.py")

    # qwen3vl 模型与 HF 缓存
    group.add_argument(
        "--qwen_model_path",
        type=str,
        default="/root/autodl-tmp/cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
    )
    group.add_argument("--qwen_hf_home", type=str, default="/root/autodl-tmp/cache/huggingface")
    group.add_argument("--qwen_hf_hub_cache", type=str, default="/root/autodl-tmp/cache/huggingface/hub")
    group.add_argument("--qwen_max_side", type=int, default=896)
    group.add_argument("--qwen_max_new_tokens", type=int, default=192)


def validate_qwen3_args(args):
    if not getattr(args, "enable_qwen_roi", False):
        return

    assert len(args.qwen_update_iters) > 0, "qwen_update_iters must not be empty"
    assert args.lambda_qwen_roi >= 0.0
    assert args.qwen_geo_weight >= 0.0
    assert args.qwen_blur_weight >= 0.0
    assert args.qwen_decay_tau > 0.0
    assert args.qwen_box_dilate >= 0
    assert os.path.isfile(args.qwen_runner_py), f"qwen_runner_py not found: {args.qwen_runner_py}"


# ============================================================
# 通用工具
# ============================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def clip_xyxy_to_hw(box, H, W):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1 = max(0, min(x1, W - 1))
    y1 = max(0, min(y1, H - 1))
    x2 = max(0, min(x2, W - 1))
    y2 = max(0, min(y2, H - 1))
    return [x1, y1, x2, y2]


def valid_xyxy(box):
    x1, y1, x2, y2 = box
    return (x2 > x1) and (y2 > y1)


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def intersection_area(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def iou(box1, box2):
    inter = intersection_area(box1, box2)
    union = box_area(box1) + box_area(box2) - inter
    return inter / union if union > 0 else 0.0


def overlap_ratio_on_smaller(box1, box2):
    inter = intersection_area(box1, box2)
    smaller = min(box_area(box1), box_area(box2))
    return inter / smaller if smaller > 0 else 0.0


def rel1000_box_to_pixel(box, W, H):
    x1, y1, x2, y2 = box
    x1 = int(round(x1 / 1000.0 * W))
    y1 = int(round(y1 / 1000.0 * H))
    x2 = int(round(x2 / 1000.0 * W))
    y2 = int(round(y2 / 1000.0 * H))
    return [x1, y1, x2, y2]


def map_box_between_sizes(box, src_w, src_h, dst_w, dst_h):
    x1, y1, x2, y2 = box
    x1 = int(round(x1 / src_w * dst_w))
    y1 = int(round(y1 / src_h * dst_h))
    x2 = int(round(x2 / src_w * dst_w))
    y2 = int(round(y2 / src_h * dst_h))
    return [x1, y1, x2, y2]


def add_offset_to_box(box, offset_x, offset_y):
    x1, y1, x2, y2 = box
    return [x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y]


def crop_pil(img, box):
    x1, y1, x2, y2 = box
    return img.crop((x1, y1, x2, y2))


def extract_json_text(output_text: str) -> str:
    txt = output_text.strip()

    if txt.startswith("```json"):
        txt = txt[len("```json"):].strip()
    if txt.startswith("```"):
        txt = txt[len("```"):].strip()
    if txt.endswith("```"):
        txt = txt[:-3].strip()

    l = txt.find("{")
    r = txt.rfind("}")
    if l != -1 and r != -1 and r > l:
        txt = txt[l:r + 1]

    return txt


# ============================================================
# 训练侧：memory + ROI loss
# ============================================================
def init_qwen_state(args):
    return {
        "qwen_memory": {},
        "qwen_region_bank": {},
    }


def load_qwen_region_bank(qwen_region_dir: str):
    bank = {}
    if (qwen_region_dir is None) or (qwen_region_dir == ""):
        print("[INFO] qwen_region_dir is empty, skip loading qwen region bank.")
        return bank
    if not os.path.isdir(qwen_region_dir):
        print(f"[WARN] qwen_region_dir not found: {qwen_region_dir}")
        return bank

    for p in sorted(Path(qwen_region_dir).glob("*.json")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            regions = data.get("regions", []) if isinstance(data, dict) else []
            bank[p.stem] = regions
        except Exception as e:
            print(f"[WARN] Failed to load {p}: {e}")

    print(f"[INFO] Loaded {len(bank)} qwen region jsons from {qwen_region_dir}")
    return bank


def maybe_bootstrap_qwen_memory_from_bank(qwen_state, image_name, iteration):
    if image_name in qwen_state["qwen_memory"]:
        return
    if image_name not in qwen_state["qwen_region_bank"]:
        return

    qwen_state["qwen_memory"][image_name] = {
        "regions": qwen_state["qwen_region_bank"][image_name],
        "last_update_iter": iteration,
        "source": "bootstrap_from_bank",
    }


def get_region_box_for_current_view(region: dict):
    for key in ["bbox_xyxy_original", "bbox_xyxy", "bbox_xyxy_resized"]:
        if key in region and region[key] is not None and len(region[key]) == 4:
            return region[key]
    return None


def get_region_weight(error_type: str, args):
    if error_type == "geometry_error":
        return args.qwen_geo_weight
    elif error_type == "under_reconstruction_blur":
        return args.qwen_blur_weight
    elif error_type == "floater_artifact":
        return 0.0
    else:
        return 0.0


def build_qwen_weight_map(regions, H, W, args, device):
    weight_map = torch.zeros((H, W), dtype=torch.float32, device=device)

    for reg in regions:
        box = get_region_box_for_current_view(reg)
        if box is None:
            continue

        x1, y1, x2, y2 = clip_xyxy_to_hw(box, H, W)

        if args.qwen_box_dilate > 0:
            d = int(args.qwen_box_dilate)
            x1 = max(0, x1 - d)
            y1 = max(0, y1 - d)
            x2 = min(W - 1, x2 + d)
            y2 = min(H - 1, y2 + d)

        if not valid_xyxy([x1, y1, x2, y2]):
            continue

        w = get_region_weight(reg.get("error_type", ""), args)
        if w <= 0:
            continue

        weight_map[y1:y2, x1:x2] = torch.maximum(
            weight_map[y1:y2, x1:x2],
            torch.full((y2 - y1, x2 - x1), w, device=device)
        )

    return weight_map


def compute_qwen_roi_loss(image, gt_image, qwen_mem_item, iteration, args):
    if qwen_mem_item is None:
        return image.new_tensor(0.0)

    regions = qwen_mem_item.get("regions", [])
    if len(regions) == 0:
        return image.new_tensor(0.0)

    _, H, W = gt_image.shape
    device = image.device

    weight_map = build_qwen_weight_map(regions, H, W, args, device)
    if float(weight_map.max().item()) <= 0:
        return image.new_tensor(0.0)

    staleness = max(0, iteration - qwen_mem_item.get("last_update_iter", iteration))
    decay = math.exp(-float(staleness) / max(1.0, float(args.qwen_decay_tau)))
    weight_map = weight_map * decay

    per_pixel_l1 = torch.abs(image - gt_image).mean(dim=0)
    roi_loss = (per_pixel_l1 * weight_map).sum() / (weight_map.sum() + 1e-6)

    return args.lambda_qwen_roi * roi_loss


# ============================================================
# 阶段性刷新：render current train views -> call qwen3vl env
# ============================================================
def render_train_views_for_qwen(scene, render_func, render_args, out_dir):
    ensure_dir(out_dir)
    render_dir = os.path.join(out_dir, "renders")
    gt_dir = os.path.join(out_dir, "gts")
    ensure_dir(render_dir)
    ensure_dir(gt_dir)

    cameras = scene.getTrainCameras()
    for viewpoint in cameras:
        render_pkg = render_func(viewpoint, scene.gaussians, *render_args)
        image = torch.clamp(render_pkg["render"], 0.0, 1.0)
        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

        image_name = viewpoint.image_name
        torchvision.utils.save_image(image, os.path.join(render_dir, f"{image_name}.png"))
        torchvision.utils.save_image(gt_image, os.path.join(gt_dir, f"{image_name}.png"))

    return render_dir, gt_dir


def run_qwen_detect_batch_subprocess(args, gt_dir, render_dir, out_dir):
    ensure_dir(out_dir)

    cmd = [
        args.qwen_conda_exe,
        "run",
        "-n",
        args.qwen_conda_env,
        "python",
        args.qwen_runner_py,
        "--mode",
        "detect_batch",
        "--gt_dir",
        gt_dir,
        "--render_dir",
        render_dir,
        "--out_dir",
        out_dir,
        "--model_path",
        args.qwen_model_path,
        "--hf_home",
        args.qwen_hf_home,
        "--hf_hub_cache",
        args.qwen_hf_hub_cache,
        "--max_side",
        str(args.qwen_max_side),
        "--max_new_tokens",
        str(args.qwen_max_new_tokens),
    ]

    print("[INFO] Running qwen detect batch:")
    print(" ".join(cmd))

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Qwen batch detect failed, returncode={result.returncode}")


def load_qwen_json_bank_from_dir(out_dir):
    bank = {}
    if not os.path.isdir(out_dir):
        return bank

    for p in sorted(Path(out_dir).glob("*.json")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                bank[p.stem] = data.get("regions", [])
        except Exception as e:
            print(f"[WARN] Failed to load qwen json {p}: {e}")
    return bank


def refresh_qwen_memory_stage(scene, render_func, render_args, iteration, args, qwen_state):
    stage_root = os.path.join(args.model_path, "qwen_refresh", f"iter_{iteration}")
    render_dir, gt_dir = render_train_views_for_qwen(
        scene=scene,
        render_func=render_func,
        render_args=render_args,
        out_dir=stage_root,
    )

    qwen_json_dir = os.path.join(stage_root, "jsons")
    ensure_dir(qwen_json_dir)

    run_qwen_detect_batch_subprocess(
        args=args,
        gt_dir=gt_dir,
        render_dir=render_dir,
        out_dir=qwen_json_dir,
    )

    new_bank = load_qwen_json_bank_from_dir(qwen_json_dir)
    print(f"[INFO] Loaded {len(new_bank)} refreshed qwen jsons from {qwen_json_dir}")

    for image_name, regions in new_bank.items():
        qwen_state["qwen_region_bank"][image_name] = regions
        qwen_state["qwen_memory"][image_name] = {
            "regions": regions,
            "last_update_iter": iteration,
            "source": f"stage_refresh_iter_{iteration}",
        }

    return qwen_json_dir


# ============================================================
# qwen3vl 环境下执行的高召回批量检测逻辑
# ============================================================

PROMPT_FULL = """
You are a visual difference detector for sparse-view 3D reconstruction.

Task:
Compare Image 1 (ground-truth image) and Image 2 (rendered image).
Detect all clearly visible local regions in Image 2 that differ noticeably from Image 1.

Your goal is to find local structure / texture / detail differences between the two images.
Do NOT require certainty that the issue is a severe reconstruction failure.
If a small but meaningful local structure is present in Image 1 but missing, weakened, blurred, erased, or smoothed away in Image 2, you should still return it.

Focus especially on:
- structural mismatch
- local texture mismatch
- missing local details
- erased or disappeared small structures
- blurred or softened local structures
- duplicated or ghosted structures
- local appearance drift
- local rendering inconsistency

Very important priority:
1. Small-object omission is important.
2. If a small structure exists in Image 1 but is absent or noticeably weakened in Image 2, return a box around it.
3. This includes small dark components, ceiling fixtures, nozzles, lamps, handles, edge parts, corners, and other small semantic structures.
4. Do NOT ignore a region just because it is small.

Important rules:
1. Focus on visible local differences between Image 1 and Image 2.
2. Return up to 10 candidate regions.
3. Small regions are allowed and encouraged if they contain meaningful visible differences.
4. Prefer missing-object regions, erased small structures, and fine-detail loss as valid detections.
5. A region can be returned even if the issue is subtle, as long as it is visibly noticeable.
6. If a region is visibly different but not clearly geometric, still return it.
7. Small overlaps between candidate boxes are allowed at this stage.
8. Do NOT return only the single most obvious region if other visible local differences also exist.
9. Return an empty list only if the two images are almost identical even at local detail level.

Labeling rules:
- Use "geometry_error" for missing local structures, disappeared objects, distorted shapes, duplicated structures, or strong spatial mismatch.
- Use "under_reconstruction_blur" for local blur, softening, missing fine detail, or weakly reconstructed structure.
- Use "floater_artifact" for detached or obviously spurious rendered content.
- Use "appearance_only" for a visible local mismatch that is noticeable but not clearly geometric.
- Use "occlusion_ambiguous" only if the region is genuinely hard to judge.

Important fallback rule:
If a small local structure is present in Image 1 but absent or heavily weakened in Image 2,
prefer "geometry_error" instead of returning nothing.

Coordinate rule:
- All bounding boxes must use relative coordinates on a 0-1000 scale for Image 2.
- [0, 0] is the top-left corner of Image 2.
- [1000, 1000] is the bottom-right corner of Image 2.

Return ONLY valid JSON:
{
  "regions": [
    {
      "bbox_xyxy": [x1, y1, x2, y2],
      "error_type": "geometry_error",
      "confidence": 0.0,
      "brief_reason": "short phrase"
    }
  ]
}

Allowed error_type:
["geometry_error", "under_reconstruction_blur", "floater_artifact", "appearance_only", "occlusion_ambiguous"]
"""

PROMPT_CROP = """
You are a visual difference detector for sparse-view 3D reconstruction.

Task:
Image 1 and Image 2 are cropped local views from a larger scene.
Compare Image 1 (ground-truth crop) and Image 2 (rendered crop).
Detect all clearly visible local regions in Image 2 that differ noticeably from Image 1.

Your goal is to find local structure / texture / detail differences between the two cropped images.
Do NOT require certainty that the issue is a severe reconstruction failure.
If a small but meaningful local structure is present in Image 1 but missing, weakened, blurred, erased, or smoothed away in Image 2, you should still return it.

Focus especially on:
- structural mismatch
- local texture mismatch
- missing local details
- erased or disappeared small structures
- blurred or softened local structures
- duplicated or ghosted structures
- local appearance drift
- local rendering inconsistency

Very important priority:
1. Small-object omission is important.
2. If a small structure exists in Image 1 but is absent or noticeably weakened in Image 2, return a box around it.
3. This includes small dark components, ceiling fixtures, nozzles, lamps, handles, edge parts, corners, and other small semantic structures.
4. Do NOT ignore a region just because it is small.

Important rules:
1. Focus on visible local differences between Image 1 and Image 2 inside this crop.
2. Return up to 8 candidate regions.
3. Small regions are allowed and encouraged if they contain meaningful visible differences.
4. Prefer missing-object regions, erased small structures, and fine-detail loss as valid detections.
5. A region can be returned even if the issue is subtle, as long as it is visibly noticeable.
6. If a region is visibly different but not clearly geometric, still return it.
7. Small overlaps between candidate boxes are allowed at this stage.
8. Do NOT return only the single most obvious region if other visible local differences also exist in this crop.
9. Return an empty list only if the two cropped images are almost identical even at local detail level.

Labeling rules:
- Use "geometry_error" for missing local structures, disappeared objects, distorted shapes, duplicated structures, or strong spatial mismatch.
- Use "under_reconstruction_blur" for local blur, softening, missing fine detail, or weakly reconstructed structure.
- Use "floater_artifact" for detached or obviously spurious rendered content.
- Use "appearance_only" for a visible local mismatch that is noticeable but not clearly geometric.
- Use "occlusion_ambiguous" only if the region is genuinely hard to judge.

Important fallback rule:
If a small local structure is present in Image 1 but absent or heavily weakened in Image 2,
prefer "geometry_error" instead of returning nothing.

Coordinate rule:
- All bounding boxes must use relative coordinates on a 0-1000 scale for the current cropped Image 2.
- [0, 0] is the top-left corner of the cropped Image 2.
- [1000, 1000] is the bottom-right corner of the cropped Image 2.

Return ONLY valid JSON:
{
  "regions": [
    {
      "bbox_xyxy": [x1, y1, x2, y2],
      "error_type": "geometry_error",
      "confidence": 0.0,
      "brief_reason": "short phrase"
    }
  ]
}

Allowed error_type:
["geometry_error", "under_reconstruction_blur", "floater_artifact", "appearance_only", "occlusion_ambiguous"]
"""


def default_confidence(error_type):
    if error_type in ["geometry_error", "under_reconstruction_blur", "floater_artifact"]:
        return 0.65
    return 0.40


def run_detection_once(model, processor, process_vision_info, gt_img, render_img, prompt, max_new_tokens=256):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": gt_img},
                {"type": "image", "image": render_img},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    if torch.cuda.is_available():
        inputs = {k: v.to("cuda:0") if torch.is_tensor(v) else v for k, v in inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    try:
        json_text = extract_json_text(output_text)
        raw_result = json.loads(json_text)
    except Exception as e:
        raw_result = {
            "regions": [],
            "_parse_error": str(e),
            "_raw_output": output_text,
        }

    return output_text, raw_result


def build_quadrant_views(gt_img, render_img, overlap_ratio_x=0.10, overlap_ratio_y=0.10):
    w, h = render_img.size
    mx = w // 2
    my = h // 2
    ox = int(w * overlap_ratio_x)
    oy = int(h * overlap_ratio_y)

    boxes = {
        "quad_tl": [0, 0, min(w, mx + ox), min(h, my + oy)],
        "quad_tr": [max(0, mx - ox), 0, w, min(h, my + oy)],
        "quad_bl": [0, max(0, my - oy), min(w, mx + ox), h],
        "quad_br": [max(0, mx - ox), max(0, my - oy), w, h],
    }

    views = []
    for name, box in boxes.items():
        views.append({
            "view_name": name,
            "crop_box_in_full": box,
            "gt_img": crop_pil(gt_img, box),
            "render_img": crop_pil(render_img, box),
        })
    return views


def raw_result_to_candidates(raw_result,
                             source_view,
                             view_box_in_full,
                             view_w, view_h,
                             full_resized_w, full_resized_h,
                             full_orig_w, full_orig_h):
    candidates = []
    regions = raw_result.get("regions", []) if isinstance(raw_result, dict) else []

    offset_x = 0
    offset_y = 0
    if view_box_in_full is not None:
        offset_x, offset_y = view_box_in_full[0], view_box_in_full[1]

    for r in regions:
        bbox = r.get("bbox_xyxy", None)
        error_type = r.get("error_type", "unknown")
        brief_reason = r.get("brief_reason", "")

        raw_conf = r.get("confidence", None)
        try:
            confidence = float(raw_conf)
        except Exception:
            confidence = -1.0

        if confidence <= 0.0:
            confidence = default_confidence(error_type)

        if bbox is None or len(bbox) != 4:
            continue

        box_in_view = rel1000_box_to_pixel(bbox, view_w, view_h)
        box_in_view = clip_xyxy_to_hw(box_in_view, view_h, view_w)
        if not valid_xyxy(box_in_view):
            continue

        box_in_full_resized = add_offset_to_box(box_in_view, offset_x, offset_y)
        box_in_full_resized = clip_xyxy_to_hw(box_in_full_resized, full_resized_h, full_resized_w)
        if not valid_xyxy(box_in_full_resized):
            continue

        area_ratio_resized = box_area(box_in_full_resized) / float(full_resized_w * full_resized_h)

        box_in_full_original = map_box_between_sizes(
            box_in_full_resized,
            full_resized_w, full_resized_h,
            full_orig_w, full_orig_h
        )
        box_in_full_original = clip_xyxy_to_hw(box_in_full_original, full_orig_h, full_orig_w)

        candidates.append({
            "source_view": source_view,
            "bbox_xyxy_resized": box_in_full_resized,
            "bbox_xyxy_original": box_in_full_original,
            "error_type": error_type,
            "confidence": confidence,
            "brief_reason": brief_reason,
            "area_ratio_resized": round(area_ratio_resized, 4),
        })

    return candidates


def same_region_det(box1, box2):
    return (iou(box1, box2) > 0.35) or (overlap_ratio_on_smaller(box1, box2) > 0.65)


def build_overlap_components_det(candidates):
    n = len(candidates)
    if n == 0:
        return []

    adj = [[] for _ in range(n)]
    for i in range(n):
        b1 = candidates[i]["bbox_xyxy_resized"]
        for j in range(i + 1, n):
            b2 = candidates[j]["bbox_xyxy_resized"]
            if same_region_det(b1, b2):
                adj[i].append(j)
                adj[j].append(i)

    visited = [False] * n
    comps = []

    for i in range(n):
        if visited[i]:
            continue
        q = deque([i])
        visited[i] = True
        comp_idx = []

        while q:
            u = q.popleft()
            comp_idx.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    q.append(v)

        comps.append([candidates[k] for k in comp_idx])

    return comps


def choose_representative_region_det(component):
    if len(component) == 1:
        c = component[0]
        return {
            "source_view": c["source_view"],
            "bbox_xyxy_resized": c["bbox_xyxy_resized"],
            "bbox_xyxy_original": c["bbox_xyxy_original"],
            "error_type": c["error_type"],
            "confidence": c["confidence"],
            "brief_reason": c["brief_reason"],
            "area_ratio_resized": c["area_ratio_resized"],
            "member_count": 1,
        }

    top_conf = max(c["confidence"] for c in component)
    near_top = [c for c in component if c["confidence"] >= top_conf - 0.03]

    areas = sorted(c["area_ratio_resized"] for c in near_top)
    median_area = areas[len(areas) // 2]

    def key_fn(c):
        source_penalty = 1 if c["source_view"] == "global" else 0
        return (
            abs(c["area_ratio_resized"] - median_area),
            source_penalty,
            -c["confidence"],
        )

    best = sorted(near_top, key=key_fn)[0]
    return {
        "source_view": best["source_view"],
        "bbox_xyxy_resized": best["bbox_xyxy_resized"],
        "bbox_xyxy_original": best["bbox_xyxy_original"],
        "error_type": best["error_type"],
        "confidence": best["confidence"],
        "brief_reason": best["brief_reason"],
        "area_ratio_resized": best["area_ratio_resized"],
        "member_count": len(component),
    }


def final_priority_det(region):
    conf = float(region.get("confidence", 0.0))
    area_ratio = float(region.get("area_ratio_resized", 0.0))
    source_bonus = 0.02 if region.get("source_view", "global") != "global" else 0.0
    return conf + source_bonus - 0.20 * area_ratio


def strict_non_overlap_regions_det(regions):
    ordered = sorted(regions, key=final_priority_det, reverse=True)
    keep = []

    for r in ordered:
        b1 = r["bbox_xyxy_resized"]
        overlapped = False
        for k in keep:
            b2 = k["bbox_xyxy_resized"]
            if intersection_area(b1, b2) > 0:
                overlapped = True
                break
        if not overlapped:
            keep.append(r)

    return keep


def postprocess_candidates_det(candidates, conf_th=0.30, max_ratio=0.35, keep_weak_types=True):
    valid_error_types = {
        "geometry_error",
        "under_reconstruction_blur",
        "floater_artifact",
        "appearance_only",
        "occlusion_ambiguous",
    }
    strong_types = {
        "geometry_error",
        "under_reconstruction_blur",
        "floater_artifact",
    }

    filtered = []
    for c in candidates:
        if c["error_type"] not in valid_error_types:
            continue
        if c["confidence"] < conf_th:
            continue
        if (not keep_weak_types) and (c["error_type"] not in strong_types):
            continue
        if not valid_xyxy(c["bbox_xyxy_resized"]):
            continue
        filtered.append(c)

    components = build_overlap_components_det(filtered)
    representatives = [choose_representative_region_det(comp) for comp in components]
    non_overlap_regions = strict_non_overlap_regions_det(representatives)

    final_regions = []
    large_regions_for_refine = []

    for r in non_overlap_regions:
        out_error_type = r["error_type"]

        # 为了训练侧更容易消费，把弱视觉差异并到 under_reconstruction_blur
        if out_error_type == "appearance_only":
            out_error_type = "under_reconstruction_blur"

        out = {
            "bbox_xyxy_original": r["bbox_xyxy_original"],
            "error_type": out_error_type,
            "confidence": r["confidence"],
            "brief_reason": r["brief_reason"],
        }

        if r["area_ratio_resized"] > max_ratio:
            large_regions_for_refine.append(out)
        else:
            final_regions.append(out)

    return {
        "regions": final_regions,
        "large_regions_for_refine": large_regions_for_refine,
    }


def detect_batch_main(cli_args):
    os.environ["HF_HOME"] = cli_args.hf_home
    os.environ["HF_HUB_CACHE"] = cli_args.hf_hub_cache
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from qwen_vl_utils import process_vision_info

    def resize_keep_ratio(img_path, max_side=1024):
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        scale = min(max_side / max(w, h), 1.0)
        new_w = int(w * scale)
        new_h = int(h * scale)
        if scale < 1.0:
            img = img.resize((new_w, new_h))
        return img, (w, h), (new_w, new_h)

    device_map = {"": "cuda:0"} if torch.cuda.is_available() else "cpu"

    model = AutoModelForImageTextToText.from_pretrained(
        cli_args.model_path,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map,
        local_files_only=True,
    )
    processor = AutoProcessor.from_pretrained(
        cli_args.model_path,
        local_files_only=True,
    )

    gt_files = {p.stem: p for p in sorted(Path(cli_args.gt_dir).glob("*.png"))}
    render_files = {p.stem: p for p in sorted(Path(cli_args.render_dir).glob("*.png"))}
    keys = sorted(set(gt_files.keys()) & set(render_files.keys()))

    ensure_dir(cli_args.out_dir)

    debug_root = os.path.join(cli_args.out_dir, "_debug")
    ensure_dir(debug_root)

    for stem in keys:
        gt_img, gt_orig_hw, gt_resized_hw = resize_keep_ratio(str(gt_files[stem]), max_side=cli_args.max_side)
        render_img, render_orig_hw, render_resized_hw = resize_keep_ratio(str(render_files[stem]), max_side=cli_args.max_side)

        orig_w, orig_h = render_orig_hw
        resized_w, resized_h = render_resized_hw

        debug_dir = os.path.join(debug_root, stem)
        ensure_dir(debug_dir)

        all_candidates = []

        # ---------- pass 1: global ----------
        global_output_text, global_raw_result = run_detection_once(
            model, processor, process_vision_info,
            gt_img, render_img,
            PROMPT_FULL,
            max_new_tokens=max(cli_args.max_new_tokens, 192)
        )

        with open(os.path.join(debug_dir, "global_output.txt"), "w", encoding="utf-8") as f:
            f.write(global_output_text)

        global_candidates = raw_result_to_candidates(
            global_raw_result,
            source_view="global",
            view_box_in_full=None,
            view_w=resized_w,
            view_h=resized_h,
            full_resized_w=resized_w,
            full_resized_h=resized_h,
            full_orig_w=orig_w,
            full_orig_h=orig_h,
        )
        all_candidates.extend(global_candidates)

        # ---------- pass 2-5: quadrants ----------
        quadrant_views = build_quadrant_views(
            gt_img, render_img,
            overlap_ratio_x=0.10,
            overlap_ratio_y=0.10
        )

        per_view_raw = {"global": global_raw_result}
        per_view_raw_text = {"global": global_output_text}

        for view in quadrant_views:
            raw_text, raw_result = run_detection_once(
                model, processor, process_vision_info,
                view["gt_img"], view["render_img"],
                PROMPT_CROP,
                max_new_tokens=max(cli_args.max_new_tokens, 192)
            )

            per_view_raw[view["view_name"]] = raw_result
            per_view_raw_text[view["view_name"]] = raw_text

            crop_w, crop_h = view["render_img"].size
            candidates = raw_result_to_candidates(
                raw_result,
                source_view=view["view_name"],
                view_box_in_full=view["crop_box_in_full"],
                view_w=crop_w,
                view_h=crop_h,
                full_resized_w=resized_w,
                full_resized_h=resized_h,
                full_orig_w=orig_w,
                full_orig_h=orig_h,
            )
            all_candidates.extend(candidates)

        processed = postprocess_candidates_det(
            all_candidates,
            conf_th=0.30,
            max_ratio=0.35,
            keep_weak_types=True,
        )

        out_regions = processed["regions"] + processed["large_regions_for_refine"]
        out_json = {"regions": out_regions}

        out_path = os.path.join(cli_args.out_dir, f"{stem}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_json, f, ensure_ascii=False, indent=2)

        with open(os.path.join(debug_dir, "global_raw.json"), "w", encoding="utf-8") as f:
            json.dump(global_raw_result, f, ensure_ascii=False, indent=2)

        with open(os.path.join(debug_dir, "all_candidates.json"), "w", encoding="utf-8") as f:
            json.dump(all_candidates, f, ensure_ascii=False, indent=2)

        with open(os.path.join(debug_dir, "processed_full.json"), "w", encoding="utf-8") as f:
            json.dump(processed, f, ensure_ascii=False, indent=2)

        with open(os.path.join(debug_dir, "per_view_raw.json"), "w", encoding="utf-8") as f:
            json.dump(per_view_raw, f, ensure_ascii=False, indent=2)

        with open(os.path.join(debug_dir, "per_view_raw_text.json"), "w", encoding="utf-8") as f:
            json.dump(per_view_raw_text, f, ensure_ascii=False, indent=2)


# ============================================================
# CLI
# ============================================================
def build_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="")
    parser.add_argument("--gt_dir", type=str, default="")
    parser.add_argument("--render_dir", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--hf_home", type=str, default="/root/autodl-tmp/cache/huggingface")
    parser.add_argument("--hf_hub_cache", type=str, default="/root/autodl-tmp/cache/huggingface/hub")
    parser.add_argument("--max_side", type=int, default=896)
    parser.add_argument("--max_new_tokens", type=int, default=192)
    return parser


if __name__ == "__main__":
    parser = build_cli()
    cli_args = parser.parse_args()

    if cli_args.mode == "detect_batch":
        detect_batch_main(cli_args)
    else:
        raise ValueError(f"Unknown mode: {cli_args.mode}")