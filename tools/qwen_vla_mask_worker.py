#!/usr/bin/env python3
"""Qwen worker for VLA-SREGS mask mining.

Run from the SREGS environment through:
  conda run -n qwen3vl python tools/qwen_vla_mask_worker.py \
      --model /path/to/Qwen3-VL --image pseudo_rgb.png --out qwen_regions.json

The worker returns strict JSON:
{
  "regions": [
    {"category": "geometry_ambiguity" | "appearance_collapse",
     "box": [x1, y1, x2, y2], "confidence": 0.0-1.0}
  ]
}
"""

import argparse
import json
import os
import re
from typing import Any, Dict

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

try:
    from qwen_vl_utils import process_vision_info
except Exception:
    process_vision_info = None


PROMPT = """
You are a sparse-view 3D Gaussian Splatting overfitting diagnostician.

You will see a pseudo novel-view rendering. There is no ground-truth image.
Your task is NOT to judge aesthetics. Find only regions that likely reveal
sparse-view overfitting in an unseen view.

Allowed categories:
1. geometry_ambiguity: shape drift, unstable object boundary, stretched geometry,
   floaters, broken structure, depth/occlusion inconsistency.
2. appearance_collapse: color collapse, background collapse, view-dependent color
   artifact, severe texture blur caused by training-view memorization.

Return strict JSON only:
{
  "regions": [
    {"category": "geometry_ambiguity" or "appearance_collapse",
     "box": [x1, y1, x2, y2],
     "confidence": 0.0-1.0,
     "reason": "short reason"}
  ]
}

Rules:
- Coordinates must be pixel coordinates in the input image.
- Use at most 6 boxes.
- Prefer high-confidence regions. If no clear overfitting artifact is visible,
  return {"regions": []}.
- Do not include markdown or extra text.
"""


def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return {"regions": []}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {"regions": []}


def sanitize(data: Dict[str, Any], image_path: str) -> Dict[str, Any]:
    img = Image.open(image_path)
    W, H = img.size
    regions = data.get("regions", [])
    if not isinstance(regions, list):
        regions = []
    clean = []
    for r in regions[:6]:
        if not isinstance(r, dict):
            continue
        cat = str(r.get("category", "")).strip()
        if cat not in {"geometry_ambiguity", "appearance_collapse"}:
            continue
        box = r.get("box", r.get("bbox", None))
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        try:
            x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        except Exception:
            continue
        x1 = max(0, min(W - 1, x1)); x2 = max(0, min(W - 1, x2))
        y1 = max(0, min(H - 1, y1)); y2 = max(0, min(H - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        try:
            conf = float(r.get("confidence", 1.0))
        except Exception:
            conf = 1.0
        clean.append({
            "category": cat,
            "box": [x1, y1, x2, y2],
            "confidence": max(0.0, min(1.0, conf)),
            "reason": str(r.get("reason", ""))[:160],
        })
    return {"regions": clean}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=os.environ.get("QWEN3VL_MODEL", ""))
    ap.add_argument("--image", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    if not args.model:
        raise RuntimeError("Set --model or QWEN3VL_MODEL to a local Qwen3-VL model path/name.")

    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model)

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": args.image},
            {"type": "text", "text": PROMPT},
        ],
    }]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if process_vision_info is None:
        inputs = processor(text=[text], images=[Image.open(args.image).convert("RGB")], padding=True, return_tensors="pt")
    else:
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)

    gen_kwargs = {"max_new_tokens": args.max_new_tokens}
    if args.temperature > 0:
        gen_kwargs.update({"do_sample": True, "temperature": args.temperature})
    else:
        gen_kwargs.update({"do_sample": False})

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **gen_kwargs)
    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    out_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    data = sanitize(extract_json(out_text), args.image)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
