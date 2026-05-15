import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw

from gaussian_renderer import render
from utils.consist_view import (
    xview_reproj_depth_loss,
    get_K_Kinv,
    get_W2C_C2W,
)
from utils.pseudo_pose import (
    sample_valid_pseudo_cameras,
    pseudo_camera_to_json,
    pseudo_camera_from_json,
)


# ============================================================
# Basic helpers
# ============================================================

def robust01(x, eps=1e-6):
    x = x.float()
    valid = torch.isfinite(x)

    if valid.sum() < 32:
        return torch.zeros_like(x)

    xv = x[valid]
    lo = torch.quantile(xv, 0.05)
    hi = torch.quantile(xv, 0.95)

    return ((x - lo) / (hi - lo + eps)).clamp(0.0, 1.0)


def to_2d(x):
    if x is None:
        return None
    if x.dim() == 4:
        return x[0, 0]
    if x.dim() == 3:
        return x[0]
    if x.dim() == 2:
        return x
    return x.squeeze()


def is_valid_loss(x):
    return (
        x is not None
        and torch.is_tensor(x)
        and bool(torch.isfinite(x.detach()).all().item())
        and x.detach().item() > 1e-8
    )


def boxes_to_mask(patches, H, W, device):
    mask = torch.zeros((H, W), device=device, dtype=torch.float32)

    for p in patches:
        x1, y1, x2, y2 = p["bbox"]
        x1 = max(0, min(W, int(x1)))
        x2 = max(0, min(W, int(x2)))
        y1 = max(0, min(H, int(y1)))
        y2 = max(0, min(H, int(y2)))

        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1.0

    return mask


def masked_l1(pred, gt, mask):
    if mask is None or mask.sum() < 1:
        return pred.new_tensor(0.0)

    m = mask[None]
    return (torch.abs(pred - gt) * m).sum() / (m.sum() * pred.shape[0] + 1e-6)


def image_grad(x):
    dx = x[:, :, 1:] - x[:, :, :-1]
    dy = x[:, 1:, :] - x[:, :-1, :]
    return dx, dy


def masked_grad_l1(pred, gt, mask):
    if mask is None or mask.sum() < 1:
        return pred.new_tensor(0.0)

    pred_dx, pred_dy = image_grad(pred)
    gt_dx, gt_dy = image_grad(gt)

    mx = mask[:, 1:]
    my = mask[1:, :]

    loss_x = (torch.abs(pred_dx - gt_dx) * mx[None]).sum() / (
        mx.sum() * pred.shape[0] + 1e-6
    )

    loss_y = (torch.abs(pred_dy - gt_dy) * my[None]).sum() / (
        my.sum() * pred.shape[0] + 1e-6
    )

    return loss_x + loss_y


def grad_mag(x):
    """
    x: [3,H,W]
    """
    dx = torch.zeros_like(x[:, :1, :])
    dy = torch.zeros_like(x[:, :, :1])

    gx = x[:, :, 1:] - x[:, :, :-1]
    gy = x[:, 1:, :] - x[:, :-1, :]

    gx = F.pad(gx, (0, 1, 0, 0))
    gy = F.pad(gy, (0, 0, 0, 1))

    return torch.sqrt((gx * gx + gy * gy).mean(dim=0) + 1e-8)


def box_mean(x, box):
    H, W = x.shape
    x1, y1, x2, y2 = box

    x1 = max(0, min(W, int(x1)))
    x2 = max(0, min(W, int(x2)))
    y1 = max(0, min(H, int(y1)))
    y2 = max(0, min(H, int(y2)))

    if x2 <= x1 or y2 <= y1:
        return 0.0

    return float(x[y1:y2, x1:x2].mean().detach().cpu().item())


# ============================================================
# Proxy reference from train image
# ============================================================

def build_proxy_reference(
    pseudo_cam,
    pseudo_pkg,
    ref_cam,
    detach_depth=True,
):
    """
    Build pseudo-view proxy reference by warping a real training image.

    pseudo pixel + pseudo rendered depth
        -> world point
        -> project to ref train camera
        -> sample ref_cam.original_image

    Returns:
        ref_rgb: [3,H,W]
        ref_valid: [H,W]
    """
    depth = to_2d(pseudo_pkg["depth"])
    rgb_device = depth.device

    if detach_depth:
        depth = depth.detach()

    H, W = depth.shape

    Kp, Kp_inv = get_K_Kinv(pseudo_cam, device=rgb_device, dtype=torch.float32)
    Kr, _ = get_K_Kinv(ref_cam, device=rgb_device, dtype=torch.float32)

    W2C_p, C2W_p = get_W2C_C2W(pseudo_cam, device=rgb_device, dtype=torch.float32)
    W2C_r, _ = get_W2C_C2W(ref_cam, device=rgb_device, dtype=torch.float32)

    ys, xs = torch.meshgrid(
        torch.arange(H, device=rgb_device),
        torch.arange(W, device=rgb_device),
        indexing="ij",
    )

    x = xs.reshape(-1).float()
    y = ys.reshape(-1).float()
    z = depth.reshape(-1)

    valid_depth = torch.isfinite(z) & (z > 1e-6)

    pix = torch.stack([x, y, torch.ones_like(x)], dim=0)
    Xc_p = (Kp_inv @ pix) * z[None, :]

    Xw = (C2W_p[:3, :3] @ Xc_p) + C2W_p[:3, 3:4]
    Xc_r = (W2C_r[:3, :3] @ Xw) + W2C_r[:3, 3:4]

    z_r = Xc_r[2, :]
    z_safe = z_r.clamp_min(1e-6)

    uv = Kr @ Xc_r
    u = uv[0, :] / z_safe
    v = uv[1, :] / z_safe

    ref_img = ref_cam.original_image.to(rgb_device).float().clamp(0, 1)
    Hr, Wr = ref_img.shape[-2], ref_img.shape[-1]

    inb = (
        valid_depth
        & (z_r > 1e-6)
        & (u >= 0)
        & (u <= Wr - 1)
        & (v >= 0)
        & (v <= Hr - 1)
    )

    gx = 2.0 * u / max(Wr - 1, 1) - 1.0
    gy = 2.0 * v / max(Hr - 1, 1) - 1.0

    grid = torch.stack([gx, gy], dim=-1).view(1, H, W, 2)

    ref_rgb = F.grid_sample(
        ref_img[None],
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[0]

    ref_valid = inb.view(H, W).float()

    return ref_rgb.detach(), ref_valid.detach()


# ============================================================
# Patch proposal
# ============================================================

def propose_pseudo_patches(score, valid_mask):
    """
    Internal fixed proposal rule.
    """
    patch_size = 96
    stride = 32
    topk = 1
    min_score = 0.20
    min_valid_ratio = 0.35
    nms_iou = 0.30

    H, W = score.shape

    if H < patch_size or W < patch_size:
        patch_size = min(H, W)

    score = score * valid_mask.float()

    pooled = F.avg_pool2d(
        score[None, None],
        kernel_size=patch_size,
        stride=stride,
        ceil_mode=False,
    )[0, 0]

    valid_pooled = F.avg_pool2d(
        valid_mask.float()[None, None],
        kernel_size=patch_size,
        stride=stride,
        ceil_mode=False,
    )[0, 0]

    if pooled.numel() == 0:
        return []

    flat = pooled.flatten()
    k = min(topk * 10, flat.numel())
    vals, ids = torch.topk(flat, k=k)

    def iou(b1, b2):
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
        a2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])

        return inter / (a1 + a2 - inter + 1e-6)

    patches = []

    for val, idx in zip(vals, ids):
        val = float(val.detach().cpu().item())

        yy_pool = int(idx // pooled.shape[1])
        xx_pool = int(idx % pooled.shape[1])

        valid_ratio = float(valid_pooled[yy_pool, xx_pool].detach().cpu().item())

        if val < min_score:
            continue

        if valid_ratio < min_valid_ratio:
            continue

        yy = yy_pool * stride
        xx = xx_pool * stride

        box = [
            max(0, xx),
            max(0, yy),
            min(W, xx + patch_size),
            min(H, yy + patch_size),
        ]

        keep = True
        for old in patches:
            if iou(box, old["bbox"]) > nms_iou:
                keep = False
                break

        if keep:
            patches.append(
                {
                    "bbox": box,
                    "score": val,
                    "valid_ratio": valid_ratio,
                }
            )

        if len(patches) >= topk:
            break

    return patches


def heuristic_pseudo_label(xview_map, disagree_map, blur_map, box):
    """
    Pseudo-view branch is geometry-first.

    The heuristic label before Qwen should not produce blur.
    Qwen can still inspect the montage, but helper will later filter
    pseudo non-structure labels.
    """
    sx = box_mean(robust01(xview_map), box)
    sd = box_mean(robust01(disagree_map), box)

    structure_score = 0.80 * sx + 0.20 * sd

    if structure_score > 0.30:
        label = "structure_error"
        conf = min(0.95, max(0.55, structure_score))
    else:
        label = "uncertain"
        conf = 0.50

    cues = {
        "xview": float(sx),
        "disagreement": float(sd),
        "structure_score": float(structure_score),
        "blur": float(box_mean(robust01(blur_map), box)),
    }

    return label, float(conf), cues


# ============================================================
# Visualization
# ============================================================

def _tensor_chw_to_pil_rgb(x):
    x = x.detach().float().clamp(0, 1).cpu()
    arr = (x.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def _heat_to_pil_rgb(x):
    x = robust01(x.detach().float()).cpu().numpy()
    arr = (x * 255.0).astype(np.uint8)
    rgb = np.stack([arr, arr, arr], axis=-1)
    return Image.fromarray(rgb)


def _crop_pil(img, box, pad=4):
    W, H = img.size
    x1, y1, x2, y2 = box

    x1 = max(0, int(x1) - pad)
    y1 = max(0, int(y1) - pad)
    x2 = min(W, int(x2) + pad)
    y2 = min(H, int(y2) + pad)

    return img.crop((x1, y1, x2, y2))


def save_pseudo_diagnostic_montage(
    pseudo_rgb,
    proxy_ref,
    disagreement_map,
    blur_map,
    xview_map,
    valid_map,
    box,
    out_path,
):
    panel_size = 192

    panels = [
        ("A pseudo-render", _crop_pil(_tensor_chw_to_pil_rgb(pseudo_rgb), box)),
        ("B proxy-ref", _crop_pil(_tensor_chw_to_pil_rgb(proxy_ref), box)),
        ("C disagreement", _crop_pil(_heat_to_pil_rgb(disagreement_map), box)),
        ("D blur-cue", _crop_pil(_heat_to_pil_rgb(blur_map), box)),
        ("E xview", _crop_pil(_heat_to_pil_rgb(xview_map), box)),
        ("F valid-alpha", _crop_pil(_heat_to_pil_rgb(valid_map), box)),
    ]

    canvas = Image.new("RGB", (panel_size * 3, panel_size * 2), (0, 0, 0))

    for i, (title, im) in enumerate(panels):
        im = im.resize((panel_size, panel_size), Image.BILINEAR)
        draw = ImageDraw.Draw(im)
        draw.rectangle([0, 0, panel_size, 18], fill=(0, 0, 0))
        draw.text((4, 2), title, fill=(255, 255, 255))

        x = (i % 3) * panel_size
        y = (i // 3) * panel_size
        canvas.paste(im, (x, y))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path)


# ============================================================
# Mining pseudo patches
# ============================================================

@torch.no_grad()
def mine_pseudo_patches(
    scene,
    gaussians,
    pipe,
    background,
    iteration,
    out_dir,
):
    """
    Mine pseudo-view generalization-risk patches.

    Returns:
        dict:
            {
                "pseudo_xxx": [patch_entries]
            }
    """
    train_cams = scene.getTrainCameras()

    pseudo_infos = sample_valid_pseudo_cameras(
        train_cams=train_cams,
        gaussians=gaussians,
        pipe=pipe,
        background=background,
        max_total=1,
    )

    if len(pseudo_infos) == 0:
        print(f"[pseudo_mining@{iteration}] no valid pseudo cameras")
        return {}

    pseudo_views = {}

    for pidx, info in enumerate(pseudo_infos):
        pseudo_cam = info["camera"]
        pseudo_pkg = info["pkg"]
        ref_cam = info["ref_cam"]

        pseudo_id = f"pseudo_{iteration}_{pidx:04d}"

        pseudo_rgb = pseudo_pkg["render"].detach().clamp(0, 1)
        pseudo_depth = to_2d(pseudo_pkg["depth"]).detach()
        alpha = to_2d(pseudo_pkg.get("rend_alpha", None))

        H, W = pseudo_rgb.shape[-2], pseudo_rgb.shape[-1]

        if alpha is None:
            alpha = torch.ones((H, W), device=pseudo_rgb.device)

        proxy_ref, ref_valid = build_proxy_reference(
            pseudo_cam=pseudo_cam,
            pseudo_pkg=pseudo_pkg,
            ref_cam=ref_cam,
            detach_depth=True,
        )

        valid_map = ((alpha > 0.2).float() * ref_valid).detach()

        disagreement_map = (
            torch.mean(torch.abs(pseudo_rgb - proxy_ref), dim=0) * valid_map
        ).detach()

        blur_map = (
            torch.relu(grad_mag(proxy_ref) - grad_mag(pseudo_rgb)) * valid_map
        ).detach()

        # Xview residual map: pseudo -> train rendered depth consistency.
        pkg_ref_render = render(ref_cam, gaussians, pipe, background)

        xview_map = torch.zeros((H, W), device=pseudo_rgb.device)

        xout = xview_reproj_depth_loss(
            pseudo_cam,
            pseudo_pkg,
            ref_cam,
            pkg_ref_render,
            detach_j=True,
            n_samples=0,
            return_stats=True,
            return_map=True,
        )

        if isinstance(xout, tuple):
            _, extra = xout
            if isinstance(extra, dict) and "err_map" in extra:
                xview_map = extra["err_map"].detach()

        score = valid_map * (
            0.80 * robust01(xview_map)
            + 0.20 * robust01(disagreement_map)
        )

        proposals = propose_pseudo_patches(score, valid_map)

        entries = []

        for k, proposal in enumerate(proposals):
            box = proposal["bbox"]

            label, conf, cues = heuristic_pseudo_label(
                xview_map=xview_map,
                disagree_map=disagreement_map,
                blur_map=blur_map,
                box=box,
            )

            montage_name = f"{pseudo_id}_patch_{k}.png"
            montage_path = os.path.join(out_dir, montage_name)

            save_pseudo_diagnostic_montage(
                pseudo_rgb=pseudo_rgb,
                proxy_ref=proxy_ref,
                disagreement_map=disagreement_map,
                blur_map=blur_map,
                xview_map=xview_map,
                valid_map=valid_map,
                box=box,
                out_path=montage_path,
            )

            entries.append(
                {
                    "view_type": "pseudo",
                    "pseudo_id": pseudo_id,
                    "anchor_uid": int(info["anchor_uid"]),
                    "ref_uid": int(info["ref_uid"]),
                    "bbox": [int(v) for v in box],
                    "score": float(proposal["score"]),
                    "valid_ratio": float(proposal["valid_ratio"]),
                    "label": label,
                    "confidence": float(conf),
                    "label_source": "heuristic_pending_qwen",
                    "diagnostic_path": montage_path,
                    "pseudo_camera": pseudo_camera_to_json(pseudo_cam),
                    "cues": cues,
                }
            )

        if len(entries) > 0:
            pseudo_views[pseudo_id] = entries

    total = sum(len(v) for v in pseudo_views.values())
    print(
        f"[pseudo_mining@{iteration}] "
        f"pseudo_views={len(pseudo_views)} patches={total}"
    )

    return pseudo_views


# ============================================================
# Pseudo patch repair
# ============================================================

def _collect_pseudo_structure_patches(patch_db, conf_th=0.55):
    if patch_db is None:
        return []

    out = []

    for _, entries in patch_db.get("views", {}).items():
        for p in entries:
            if p.get("view_type", "train") != "pseudo":
                continue

            label = p.get("label", "uncertain")
            conf = float(p.get("confidence", 0.0))

            if label == "geometry_error":
                label = "structure_error"

            if label != "structure_error":
                continue

            if conf < conf_th:
                continue

            if "pseudo_camera" not in p:
                continue

            out.append(p)

    return out


def _find_train_cam_by_uid(train_cams, uid):
    for cam in train_cams:
        if int(cam.uid) == int(uid):
            return cam
    return None


def apply_pseudo_patch_repair_loss(
    patch_db,
    iteration,
    loss,
    train_cams,
    gaussians,
    pipe,
    background,
):
    """
    Geometry-only pseudo patch repair.

    Pseudo branch does NOT use RGB / gradient proxy supervision now.
    It only uses masked Xview depth consistency for pseudo structure_error patches.
    """
    # 保留你原来的内部频率逻辑。
    # 如果你原来 interval=4，就继续 interval=4。
    interval = 4
    if iteration % interval != 0:
        return loss

    pseudo_patches = _collect_pseudo_structure_patches(patch_db)

    if len(pseudo_patches) == 0:
        return loss

    patch = pseudo_patches[iteration % len(pseudo_patches)]

    device = background.device

    pseudo_cam = pseudo_camera_from_json(
        patch["pseudo_camera"],
        device=device,
    )

    ref_cam = _find_train_cam_by_uid(train_cams, patch.get("ref_uid", -1))
    if ref_cam is None:
        return loss

    pseudo_pkg = render(pseudo_cam, gaussians, pipe, background)

    pseudo_rgb = pseudo_pkg["render"]
    H, W = pseudo_rgb.shape[-2], pseudo_rgb.shape[-1]

    alpha = to_2d(pseudo_pkg.get("rend_alpha", None))

    mask = boxes_to_mask([patch], H, W, device=device)

    if alpha is not None:
        mask = mask * (alpha > 0.2).float()

    if mask.sum() < 16:
        return loss

    pkg_ref_render = render(ref_cam, gaussians, pipe, background)

    loss_x = xview_reproj_depth_loss(
        pseudo_cam,
        pseudo_pkg,
        ref_cam,
        pkg_ref_render,
        detach_j=True,
        mask_i=mask,
        tau_rel=0.12,
        w_err_min=0.12,
        n_samples=4096,
        return_stats=False,
        return_map=False,
    )

    if not is_valid_loss(loss_x):
        return loss

    # 动态权重：让 pseudo depth consistency 作为局部几何项，
    # 不要无上限地压过主损失。
    target_ratio = 0.03
    max_w = 0.35

    loss_before = loss.detach()

    w_dyn = (
        target_ratio
        * loss_before
        / (loss_x.detach() + 1e-6)
    ).clamp(0.0, max_w)

    loss = loss + w_dyn * loss_x

    if iteration % 400 == 0:
        ratio = ((w_dyn * loss_x).detach() / (loss_before + 1e-6)).item()
        print(
            f"[pseudo_patch@{iteration}] "
            f"id={patch.get('pseudo_id', 'unknown')} "
            f"label=structure_error "
            f"loss_x={loss_x.detach().item():.6f} "
            f"ratio={ratio:.6f} "
            f"valid={float(mask.mean().detach().cpu().item()):.3f}"
        )

    return loss