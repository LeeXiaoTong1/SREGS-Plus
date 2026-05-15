import os

# =========================================
# Environment variables: must be set before transformers import
# =========================================
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/cache/huggingface")
os.environ.setdefault("HF_HUB_CACHE", "/root/autodl-tmp/cache/huggingface/hub")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import json
import re
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info


# =========================================
# Default config
# =========================================

DEFAULT_MODEL_PATH = (
    "/root/autodl-tmp/cache/huggingface/hub/"
    "models--Qwen--Qwen3-VL-8B-Instruct/"
    "snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 128
MIN_CONFIDENCE = 0.50


# =========================================
# Prompt
# =========================================

PROMPT = """
你是 sparse-view 3D Gaussian Splatting 的局部重建退化诊断器。

你会看到一张 2x3 diagnostic montage。它可能来自 train view，也可能来自 pseudo view。

========================
一、面板含义
========================

A = 当前高斯渲染 crop。
    - 如果标题是 render，表示 train-view render。
    - 如果标题是 pseudo-render，表示 pseudo-view render。

B = reference crop。
    - 对 train view，B 是真实 GT crop。
    - 对 pseudo view，B 是由真实训练图像重投影得到的 proxy reference。
    - 注意：pseudo view 的 B 不是 GT，可能包含重投影噪声、遮挡误差、边界错配或模糊。

C = residual / disagreement cue。
    - 对 train view，通常是 render 与 GT 的 photo residual。
    - 对 pseudo view，通常是 pseudo-render 与 proxy reference 的 disagreement。
    - 注意：pseudo view 的 C 可能受 proxy reference 错配影响，不能单独作为错误证据。

D = depth / blur cue。
    - 对 train view，通常是 depth residual。
    - 对 pseudo view，可能是 blur cue 或 depth-related cue。
    - 注意：pseudo view 下 D 不能单独作为 blur 判据。

E = xview residual。
    - 表示 cross-view reprojection residual。
    - 这是判断几何错误、深度不一致、未知视角伪影的重要证据。

F = alpha / valid mask。
    - 表示有效区域、透明区域、hole 或 alpha 覆盖。
    - 如果 F 显示该 patch 大面积无效，应倾向 uncertain。

========================
二、你的任务
========================

只判断当前 patch 属于下面三类之一：

1. structure_error
表示几何结构、边界、深度、形状或多视角一致性明显错误。

典型证据：
- A 中物体边界、位置、形状、深度关系明显异常；
- 出现结构断裂、重复、扭曲、塌陷、漂浮伪影；
- E xview residual 在 patch 内明显偏高；
- D 中深度/几何 cue 明显异常；
- A 与 B 的主要差异不是简单纹理差异，而是结构关系错误。

2. blur
表示局部模糊、细节不足、纹理欠拟合或密集化不足。

重要限制：
- blur 类主要用于 train view。
- train view 中，如果 A 相比 B 更糊，但物体大结构、边界和位置大致正确，可以输出 blur。
- pseudo view 中原则上不要输出 blur。
- 对 pseudo view，如果只是 A 与 B 的纹理不一致、proxy reference 更清晰、C/D 较亮，但没有明确 xview/几何错误，应输出 uncertain，而不是 blur。

3. uncertain
表示证据不足、参考不可靠、遮挡歧义、纯颜色差异、proxy reference 错配、无效区域过多，或者无法可靠判断。

========================
三、train view 判断规则
========================

如果这是 train view montage：
- B 是真实 GT，可以作为可靠参考。
- 如果几何结构、边界、深度、形状明显错，输出 structure_error。
- 如果结构大体正确，但 A 比 B 模糊、细节不足、纹理不清，输出 blur。
- 如果只是轻微颜色、亮度、曝光差异，输出 uncertain。
- 如果缺陷很弱、肉眼不稳定，输出 uncertain。

========================
四、pseudo view 判断规则：非常重要
========================

如果这是 pseudo view montage：
- B 是 proxy reference，不是真实 GT。
- B 可能存在重投影错误、遮挡错配、边界错位或局部噪声。
- 不要把 A 与 B 的普通纹理差异当成 blur。
- 不要因为 C disagreement 或 D blur-cue 较亮就直接输出 blur。
- pseudo view 下，原则上只在几何证据明确时输出 structure_error。
- 几何证据主要来自：
  1. E xview residual 明显；
  2. A 中出现结构断裂、漂浮伪影、形状塌陷、边界严重错位；
  3. D 中呈现明显深度/几何异常；
  4. A 的结构关系在视觉上明显不合理。

pseudo view 下：
- 如果只是 A 比 B 模糊，但 E xview 不明显，输出 uncertain。
- 如果只是 proxy reference 与 pseudo-render 不匹配，输出 uncertain。
- 如果 B proxy reference 本身不清楚、错位、遮挡或无效，输出 uncertain。
- 如果 F valid/alpha 显示大面积无效或黑区，输出 uncertain。
- pseudo view 下不要输出 blur，除非它看起来同时伴随明确结构错误；这种情况下也应输出 structure_error，而不是 blur。

========================
五、严格输出规则
========================

只输出合法 JSON。
不要输出解释性文字。
不要输出 markdown。
不要输出代码块。
不要输出思考过程。

JSON 格式必须是：
{
  "label": "structure_error | blur | uncertain",
  "confidence": 0.0,
  "reason": "不超过20字"
}

label 只能是：
- structure_error
- blur
- uncertain

confidence 范围是 0.0 到 1.0。

对于 pseudo view：
- 如果判断为 structure_error，confidence 应该主要依据 E xview residual 和几何异常强度。
- 如果没有明确几何错误，label 必须是 uncertain。
- pseudo view 下不要因为模糊感、纹理差异或 proxy reference 更清晰而输出 blur。

示例输出：
{
  "label": "structure_error",
  "confidence": 0.78,
  "reason": "xview高且结构错"
}
"""


VALID_LABELS = {"structure_error", "blur", "uncertain"}


# =========================================
# Utility
# =========================================

def resolve_image_path(path_str, patch_json_path):
    """
    diagnostic_path may be absolute or relative.
    """
    p = Path(path_str)

    if p.exists():
        return p

    p2 = patch_json_path.parent / p
    if p2.exists():
        return p2

    p3 = Path.cwd() / p
    if p3.exists():
        return p3

    raise FileNotFoundError(f"diagnostic image not found: {path_str}")


def extract_json_text(output_text):
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


def parse_model_json(output_text):
    json_text = extract_json_text(output_text)

    try:
        obj = json.loads(json_text)
    except Exception as e:
        raise ValueError(f"failed to parse JSON: {output_text[:300]}") from e

    return obj


def normalize_label(label):
    label = str(label).strip().lower()

    structure_aliases = {
        "structure_error",
        "structural_error",
        "geometry_error",
        "geometric_error",
        "geometry",
        "structure",
    }

    blur_aliases = {
        "blur",
        "blurry",
        "under_reconstruction_blur",
        "under-reconstruction_blur",
        "under_reconstructed_blur",
        "texture_blur",
        "detail_blur",
    }

    uncertain_aliases = {
        "uncertain",
        "unknown",
        "ambiguous",
        "occlusion_ambiguous",
        "appearance_only",
        "none",
        "no_clear_defect",
    }

    if label in structure_aliases:
        return "structure_error"

    if label in blur_aliases:
        return "blur"

    if label in uncertain_aliases:
        return "uncertain"

    return "uncertain"


def normalize_result(obj):
    label = normalize_label(obj.get("label", "uncertain"))

    try:
        confidence = float(obj.get("confidence", 0.0))
    except Exception:
        confidence = 0.0

    confidence = max(0.0, min(1.0, confidence))

    reason = str(obj.get("reason", "")).strip()
    if len(reason) > 40:
        reason = reason[:40]

    if label != "uncertain" and confidence < MIN_CONFIDENCE:
        label = "uncertain"

    return {
        "label": label,
        "confidence": confidence,
        "reason": reason,
    }


# =========================================
# Qwen3-VL
# =========================================

def load_qwen3vl(model_path):
    print(f"[qwen] loading model from: {model_path}")

    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map={"": DEVICE} if DEVICE.startswith("cuda") else None,
            local_files_only=True,
        )
    except TypeError:
        # For older transformers versions.
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": DEVICE} if DEVICE.startswith("cuda") else None,
            local_files_only=True,
        )

    model.eval()

    print("[qwen] loading processor...")
    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True,
    )

    return model, processor


def run_qwen_once(model, processor, image_path, prompt):
    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
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

    inputs = {
        k: v.to(DEVICE) if torch.is_tensor(v) else v
        for k, v in inputs.items()
    }

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    obj = parse_model_json(output_text)
    result = normalize_result(obj)

    return output_text, result


# =========================================
# Main labeling logic
# =========================================

def label_patches(patch_json, out_json, model_path):
    patch_json = Path(patch_json)
    out_json = Path(out_json)

    with open(patch_json, "r", encoding="utf-8") as f:
        db = json.load(f)

    raw_dir = out_json.parent / f"{out_json.stem}_raw_texts"
    raw_dir.mkdir(parents=True, exist_ok=True)

    model, processor = load_qwen3vl(model_path)

    views = db.get("views", {})

    total = 0
    counts = {
        "structure_error": 0,
        "blur": 0,
        "uncertain": 0,
        "error": 0,
    }

    for uid, patches in views.items():
        for idx, patch in enumerate(tqdm(patches, desc=f"view {uid}")):
            total += 1

            diag_path_str = patch.get("diagnostic_path", None)

            if diag_path_str is None:
                patch["heuristic_label"] = patch.get("label", None)
                patch["heuristic_confidence"] = patch.get("confidence", None)
                patch["label"] = "uncertain"
                patch["confidence"] = 0.0
                patch["label_source"] = "qwen3vl_error"
                patch["qwen_error"] = "missing diagnostic_path"
                counts["error"] += 1
                continue

            try:
                image_path = resolve_image_path(diag_path_str, patch_json)

                patch["heuristic_label"] = patch.get("label", None)
                patch["heuristic_confidence"] = patch.get("confidence", None)

                raw_text, result = run_qwen_once(
                    model=model,
                    processor=processor,
                    image_path=image_path,
                    prompt=PROMPT,
                )

                raw_text_path = raw_dir / f"view_{uid}_patch_{idx}.txt"
                with open(raw_text_path, "w", encoding="utf-8") as f:
                    f.write(raw_text)

                patch["label"] = result["label"]
                patch["confidence"] = result["confidence"]
                patch["qwen_reason"] = result["reason"]
                patch["qwen_raw_text_path"] = str(raw_text_path)
                patch["label_source"] = "qwen3vl"

                counts[result["label"]] += 1

                print(
                    f"[qwen] view={uid} patch={idx} "
                    f"label={result['label']} "
                    f"conf={result['confidence']:.3f} "
                    f"reason={result['reason']}"
                )

            except Exception as e:
                patch["heuristic_label"] = patch.get("label", None)
                patch["heuristic_confidence"] = patch.get("confidence", None)

                patch["label"] = "uncertain"
                patch["confidence"] = 0.0
                patch["label_source"] = "qwen3vl_error"
                patch["qwen_error"] = str(e)

                counts["error"] += 1

                print(f"[qwen_error] view={uid} patch={idx}: {e}")

            if DEVICE.startswith("cuda"):
                torch.cuda.empty_cache()

    db["qwen_labeling"] = {
        "model_path": str(model_path),
        "total_patches": total,
        "counts": counts,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)

    print(f"\n[qwen] saved: {out_json}")
    print(f"[qwen] total={total} counts={counts}")


def main():
    parser = argparse.ArgumentParser(
        description="Label mined 3DGS patches with local Qwen3-VL."
    )

    parser.add_argument(
        "--patch_json",
        type=str,
        required=True,
        help="Input patches.json from patch mining.",
    )

    parser.add_argument(
        "--out_json",
        type=str,
        required=True,
        help="Output patches_qwen.json.",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Local Qwen3-VL model snapshot path.",
    )

    args = parser.parse_args()

    label_patches(
        patch_json=args.patch_json,
        out_json=args.out_json,
        model_path=args.model_path,
    )


if __name__ == "__main__":
    main()