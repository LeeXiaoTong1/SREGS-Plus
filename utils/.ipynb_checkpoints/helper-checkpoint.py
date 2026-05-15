import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import subprocess
from PIL import Image, ImageDraw

from utils.pseudo_patch import (
    mine_pseudo_patches,
    apply_pseudo_patch_repair_loss,
)
from gaussian_renderer import render
from utils.consist_view import (
    xview_reproj_depth_loss,
    quick_inb_ratio,
    clear_consist_view_cache,
)


# ============================================================
# Small tensor helpers
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


def masked_depth_l1(depth_pred, depth_prior, mask):
    if mask is None or mask.sum() < 1:
        return depth_pred.new_tensor(0.0)

    dp = robust01(depth_pred)
    dt = robust01(depth_prior)

    return (torch.abs(dp - dt) * mask).sum() / (mask.sum() + 1e-6)


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
# Patch proposal
# ============================================================

def propose_patches(score):
    """
    Minimal fixed proposal rule.

    Internal defaults:
        patch_size = 96
        stride = 32
        topk = 2
    """
    patch_size = 96
    stride = 32
    topk = 2
    min_score = 0.25
    nms_iou = 0.30

    H, W = score.shape

    if H < patch_size or W < patch_size:
        patch_size = min(H, W)

    s = score[None, None]

    pooled = F.avg_pool2d(
        s,
        kernel_size=patch_size,
        stride=stride,
        ceil_mode=False,
    )[0, 0]

    if pooled.numel() == 0:
        return []

    flat = pooled.flatten()
    k = min(topk * 8, flat.numel())
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

        if val < min_score:
            continue

        yy = int(idx // pooled.shape[1]) * stride
        xx = int(idx % pooled.shape[1]) * stride

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
            patches.append({
                "bbox": box,
                "score": val,
            })

        if len(patches) >= topk:
            break

    return patches


def heuristic_patch_label(photo_map, xview_map, depth_map, alpha_map, box):
    """
    Temporary label before Qwen3-VL.

    Qwen3-VL 后续会覆盖：
        label
        confidence
        label_source
    """
    photo_n = robust01(photo_map)
    xview_n = robust01(xview_map)
    depth_n = robust01(depth_map)
    alpha_n = robust01(alpha_map)

    sp = box_mean(photo_n, box)
    sx = box_mean(xview_n, box)
    sd = box_mean(depth_n, box)
    sa = box_mean(alpha_n, box)

    structure_score = 0.60 * sx + 0.40 * sd
    blur_score = sp * (1.0 - 0.5 * min(1.0, structure_score))

    if structure_score > 0.35:
        label = "structure_error"
        conf = min(0.95, max(0.55, structure_score))
    elif blur_score > 0.20:
        label = "blur"
        conf = min(0.90, max(0.55, blur_score + 0.30))
    else:
        label = "uncertain"
        conf = 0.50

    cues = {
        "photo": float(sp),
        "xview": float(sx),
        "depth": float(sd),
        "alpha": float(sa),
        "structure_score": float(structure_score),
        "blur_score": float(blur_score),
    }

    return label, float(conf), cues


# ============================================================
# Diagnostic montage
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


def save_diagnostic_montage(
    image,
    gt_image,
    photo_map,
    depth_map,
    xview_map,
    alpha_map,
    box,
    out_path,
):
    panel_size = 192

    render_pil = _tensor_chw_to_pil_rgb(image)
    gt_pil = _tensor_chw_to_pil_rgb(gt_image)

    photo_pil = _heat_to_pil_rgb(photo_map)
    depth_pil = _heat_to_pil_rgb(depth_map)
    xview_pil = _heat_to_pil_rgb(xview_map)
    alpha_pil = _heat_to_pil_rgb(alpha_map)

    panels = [
        ("A render", _crop_pil(render_pil, box)),
        ("B gt", _crop_pil(gt_pil, box)),
        ("C photo", _crop_pil(photo_pil, box)),
        ("D depth", _crop_pil(depth_pil, box)),
        ("E xview", _crop_pil(xview_pil, box)),
        ("F alpha", _crop_pil(alpha_pil, box)),
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
# Training auxiliary helper
# ============================================================

class TrainingAuxHelper:
    """
    Keeps train_llff.py simple.

    Important:
        Global Xview is independent from patch repair.
        Patch repair never replaces global Xview.
    """

    def __init__(self, scene, pipe, background, args):
        self.scene = scene
        self.pipe = pipe
        self.background = background
        self.args = args

        # -----------------------------
        # Global Xview defaults
        # -----------------------------
        self.xview_enabled = True
        self.xview_start_iter = 4000
        self.xview_target_ratio = 0.02
        self.xview_interval = 1
        self.xview_detach_target = True

        # -----------------------------
        # Patch defaults
        # -----------------------------
        self.patch_iter = int(getattr(args, "patch_iter", 5000))
        self.mine_patches = bool(getattr(args, "mine_patches", False))
        self.patch_repair = bool(getattr(args, "patch_repair", False))
        self.patch_json = getattr(args, "patch_json", None)
        self.patch_conf_th = 0.55

        # Local repair weights.
        # They are intentionally kept internal to avoid making train.py parameter-heavy.
        self.struct_photo_w = 0.05
        self.struct_depth_w = 0.03
        self.struct_xview_w = 0.08

        self.blur_photo_w = 0.10
        self.blur_grad_w = 0.05
        # Qwen online labeling inside training
        self.qwen_online = bool(getattr(args, "qwen_online", False))

        # 这里写死，避免 train.py 参数太多
        self.qwen_iters = [3000, 4000, 5000, 6000, 7000, 8000]
        self.pseudo_iters = [4000, 5000, 6000, 7000, 8000]

        # pseudo repair 开始轮次：
        # 只有 5000 以后才允许 pseudo patch 参与训练。
        self.pseudo_start_iter = 3000
        self.pseudo_only_structure = True
        self.pseudo_min_xview_cue = 0.30
        self.qwen_env = "qwen3vl"
        self.qwen_script = "tools/qwen3vl.py"
        self.qwen_model_path = (
            "/root/autodl-tmp/cache/huggingface/hub/"
            "models--Qwen--Qwen3-VL-8B-Instruct/"
            "snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
)

        try:
            clear_consist_view_cache()
        except Exception:
            pass

        self.train_cams = scene.getTrainCameras()

        if len(self.train_cams) >= 2:
            self.centers_cpu = torch.stack(
                [c.camera_center.detach().cpu() for c in self.train_cams],
                dim=0,
            )
            self.uid2idx = {c.uid: i for i, c in enumerate(self.train_cams)}
        else:
            self.centers_cpu = None
            self.uid2idx = {}

        self.patch_db = self._load_patch_db(self.patch_json) if self.patch_repair else None
        if self.patch_db is not None:
            self.repair_start_iter = int(self.patch_db.get("source_iter", self.patch_iter))
        else:
            self.repair_start_iter = self.patch_iter

        # Cache one target render per iteration so local structure Xview can reuse it.
        self._last_xview_iter = None
        self._last_xview_cam = None
        self._last_xview_pkg = None
        self._last_xview_inb = -1.0
        self._last_xview_baseline = 0.0

        # helper.__init__
        self.patch_repair_start_iter = int(
            getattr(args, "patch_repair_start_iter", 3000)
        )
        self.patch_repair_interval = max(
            1,
            int(getattr(args, "patch_repair_interval", 4))
        )
        self.pseudo_repair = bool(
            getattr(args, "pseudo_repair", False)
        )
        self.pseudo_start_iter = int(
            getattr(args, "pseudo_start_iter", 3000)
        )
        
        if self.patch_db is not None:
            db_source_iter = int(self.patch_db.get("source_iter", self.patch_iter))
            self.repair_start_iter = max(db_source_iter, self.patch_repair_start_iter)
        else:
            self.repair_start_iter = max(self.patch_iter, self.patch_repair_start_iter)

    # ------------------------------------------------------------
    # patch db
    # ------------------------------------------------------------

    def run_qwen_if_needed(self, iteration, gaussians):
        """
        在线触发 Qwen3-VL。
    
        3000:
            train-view patches only
    
        5000 / 7000:
            train-view patches + pseudo-view patches
    
        关键：
            pseudo patch 只保留 structure_error；
            pseudo blur / uncertain 不进入训练。
        """
        if not self.qwen_online:
            return
    
        if iteration not in self.qwen_iters:
            return
    
        include_pseudo = (
            iteration >= self.pseudo_start_iter
            and iteration in self.pseudo_iters
        )
    
        print(
            f"\n[qwen_online] Trigger Qwen3-VL at iteration {iteration} "
            f"| include_pseudo={include_pseudo}"
        )
    
        patch_json = self._mine_patches(
            iteration=iteration,
            gaussians=gaussians,
            include_pseudo=include_pseudo,
        )
    
        if patch_json is None or not os.path.exists(patch_json):
            print(f"[qwen_online] failed: patches.json not found at iter {iteration}")
            return
    
        patch_dir = os.path.dirname(patch_json)
        qwen_json = os.path.join(patch_dir, "patches_qwen.json")
    
        cmd = (
            "source /root/miniconda3/etc/profile.d/conda.sh && "
            f"conda activate {self.qwen_env} && "
            f"python {self.qwen_script} "
            f"--patch_json {patch_json} "
            f"--out_json {qwen_json} "
            f"--model_path {self.qwen_model_path}"
        )
    
        print("[qwen_online] running:")
        print(cmd)
    
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
        ret = subprocess.run(
            ["bash", "-lc", cmd],
            stdout=None,
            stderr=None,
        )
    
        if ret.returncode != 0:
            print(f"[qwen_online] Qwen process failed at iter {iteration}")
            return
    
        if not os.path.exists(qwen_json):
            print(f"[qwen_online] failed: qwen json not found: {qwen_json}")
            return
    
        self.patch_json = qwen_json
        self.patch_repair = True
        self.patch_db = self._load_patch_db(qwen_json)
    
        if self.patch_db is not None:
            self.repair_start_iter = int(self.patch_db.get("source_iter", iteration))
    
            # 关键：过滤 pseudo patch，只保留 structure_error。
            self._sanitize_pseudo_patch_labels()
    
            print(
                f"[qwen_online] loaded Qwen patches from {qwen_json}, "
                f"repair_start_iter={self.repair_start_iter}"
            )


    def _load_patch_db(self, path):
        if path is None or len(path) == 0:
            print("[patch_repair] enabled, but patch_json is None.")
            return None

        if not os.path.exists(path):
            print(f"[patch_repair] patch_json not found: {path}")
            return None

        with open(path, "r", encoding="utf-8") as f:
            patch_db = json.load(f)

        n = 0
        for _, patches in patch_db.get("views", {}).items():
            n += len(patches)

        print(f"[patch_repair] loaded {n} patches from {path}")
        return patch_db

    def _patch_db_has_pseudo_structure(self):
        """
        Return True if current patch_db contains at least one valid pseudo structure patch.
        """
        if self.patch_db is None:
            return False
    
        for _, patches in self.patch_db.get("views", {}).items():
            for p in patches:
                if p.get("view_type", "train") != "pseudo":
                    continue
    
                if p.get("label", "uncertain") == "structure_error" and float(p.get("confidence", 0.0)) >= self.patch_conf_th:
                    return True
    
        return False
    
    
    def _sanitize_pseudo_patch_labels(self):
        """
        Pseudo branch is geometry-only in the first stable version.
    
        Rules:
            1. geometry_error -> structure_error
            2. pseudo blur / uncertain -> uncertain, skipped by repair
            3. pseudo structure_error also needs enough xview cue
        """
        if self.patch_db is None:
            return
    
        n_keep = 0
        n_skip_label = 0
        n_skip_xview = 0
    
        for _, patches in self.patch_db.get("views", {}).items():
            for p in patches:
                if p.get("view_type", "train") != "pseudo":
                    continue
    
                label = p.get("label", "uncertain")
    
                # Compatible with possible Qwen outputs.
                if label == "geometry_error":
                    p["label"] = "structure_error"
                    label = "structure_error"
    
                if self.pseudo_only_structure and label != "structure_error":
                    p["pseudo_original_label"] = label
                    p["label"] = "uncertain"
                    p["confidence"] = 0.0
                    p["label_source"] = str(p.get("label_source", "qwen3vl")) + "_pseudo_non_structure_filtered"
                    n_skip_label += 1
                    continue
    
                cues = p.get("cues", {})
                xview_cue = float(cues.get("xview", cues.get("structure_score", 0.0)))
    
                if xview_cue < self.pseudo_min_xview_cue:
                    p["pseudo_original_label"] = label
                    p["label"] = "uncertain"
                    p["confidence"] = 0.0
                    p["label_source"] = str(p.get("label_source", "qwen3vl")) + "_pseudo_low_xview_filtered"
                    n_skip_xview += 1
                    continue
    
                n_keep += 1
    
        print(
            f"[pseudo_filter] keep_structure={n_keep} "
            f"skip_non_structure={n_skip_label} "
            f"skip_low_xview={n_skip_xview}"
        )

    def _get_view_patches(self, uid):
        if self.patch_db is None:
            return []

        patches = self.patch_db.get("views", {}).get(str(uid), [])

        out = []
        for p in patches:
            label = p.get("label", "uncertain")
            conf = float(p.get("confidence", 0.0))

            if label not in ["structure_error", "blur"]:
                continue

            if conf < self.patch_conf_th:
                continue

            out.append(p)

        return out

    # ------------------------------------------------------------
    # xview neighbor
    # ------------------------------------------------------------

    @torch.no_grad()
    def pick_xview_neighbor(self, viewpoint_cam, render_pkg):
        if self.train_cams is None or len(self.train_cams) < 2 or self.centers_cpu is None:
            return None, -1.0, 0.0

        ci = viewpoint_cam.camera_center.detach().cpu()
        d2 = ((self.centers_cpu - ci[None, :]) ** 2).sum(dim=1)

        cur_idx = self.uid2idx.get(viewpoint_cam.uid, None)
        if cur_idx is not None:
            d2[cur_idx] = float("inf")

        k_eff = min(6, len(self.train_cams) - 1)

        if k_eff <= 0:
            return None, -1.0, 0.0

        nn_idx = torch.topk(d2, k=k_eff, largest=False).indices

        candidates = []

        for j_idx_tensor in nn_idx:
            j_idx = int(j_idx_tensor.item())
            cand = self.train_cams[j_idx]

            r = quick_inb_ratio(
                viewpoint_cam,
                render_pkg,
                cand,
                n=1024,
                alpha_th=0.2,
            )

            r = float(r.detach().cpu().item()) if torch.is_tensor(r) else float(r)
            baseline = float(torch.sqrt(d2[j_idx].clamp_min(0.0)).item())

            candidates.append({
                "cam": cand,
                "r": r,
                "baseline": baseline,
            })

        if len(candidates) == 0:
            return None, -1.0, 0.0

        max_baseline = max([c["baseline"] for c in candidates])
        max_baseline = max(max_baseline, 1e-6)

        best_cam = None
        best_r = -1.0
        best_baseline = 0.0
        best_score = -1.0

        for c in candidates:
            r = c["r"]
            baseline = c["baseline"]
            b_norm = baseline / max_baseline

            if r >= 0.75:
                score = r * (1.0 + 0.25 * b_norm)
            else:
                score = r

            if score > best_score:
                best_score = score
                best_cam = c["cam"]
                best_r = r
                best_baseline = baseline

        return best_cam, best_r, best_baseline

    # ------------------------------------------------------------
    # 1. global xview loss
    # ------------------------------------------------------------

    def apply_xview_loss(
        self,
        iteration,
        loss,
        viewpoint_cam,
        render_pkg,
        gaussians,
    ):
        """
        This is your original independent global Xview loss.

        It is not controlled by patch_repair.
        """
        self._last_xview_iter = None
        self._last_xview_cam = None
        self._last_xview_pkg = None
        self._last_xview_inb = -1.0
        self._last_xview_baseline = 0.0

        if not self.xview_enabled:
            return loss

        if iteration < self.xview_start_iter:
            return loss

        if (iteration - self.xview_start_iter) % self.xview_interval != 0:
            return loss

        if len(self.train_cams) < 2:
            return loss

        cam_j, inb, baseline = self.pick_xview_neighbor(viewpoint_cam, render_pkg)

        if cam_j is None or inb < 0.10:
            return loss

        if self.xview_detach_target:
            with torch.no_grad():
                pkg_j = render(cam_j, gaussians, self.pipe, self.background)
        else:
            pkg_j = render(cam_j, gaussians, self.pipe, self.background)

        self._last_xview_iter = iteration
        self._last_xview_cam = cam_j
        self._last_xview_pkg = pkg_j
        self._last_xview_inb = inb
        self._last_xview_baseline = baseline

        xout = xview_reproj_depth_loss(
            viewpoint_cam,
            render_pkg,
            cam_j,
            pkg_j,
            detach_j=self.xview_detach_target,
            return_stats=True,
            return_map=False,
        )

        if isinstance(xout, tuple):
            loss_xview, stats = xout
        else:
            loss_xview, stats = xout, None

        if not is_valid_loss(loss_xview):
            return loss

        warmup = min(
            1.0,
            max(0.0, float(iteration - self.xview_start_iter) / 1000.0),
        )

        loss_before = loss.detach()

        w_dyn = (
            self.xview_target_ratio
            * loss_before
            / (loss_xview.detach() + 1e-6)
        ).clamp(0.0, 0.25)

        w_eff = warmup * w_dyn

        loss = loss + w_eff * loss_xview

        if iteration % 400 == 0:
            ratio = ((w_eff * loss_xview).detach() / (loss_before + 1e-6)).item()

            rel_mean = -1.0
            effn = -1.0
            if isinstance(stats, dict):
                rel_mean = stats.get("rel_mean", -1.0)
                effn = stats.get("w_eff_num", -1.0)

            print(
                f"[xview@{iteration}] "
                f"uid={cam_j.uid} "
                f"inb={inb:.3f} "
                f"baseline={baseline:.3f} "
                f"loss={loss_xview.detach().item():.6f} "
                f"ratio={ratio:.6f} "
                f"rel_mean={rel_mean:.6f} "
                f"effN={effn:.1f}"
            )

        return loss

    # ------------------------------------------------------------
    # 2. patch-conditioned local repair
    # ------------------------------------------------------------

    def apply_patch_repair_loss(
        self,
        iteration,
        loss,
        viewpoint_cam,
        render_pkg,
        image,
        gt_image,
        rendered_depth_2d,
        midas_depth_resized,
        gaussians,
    ):
        """
        Patch repair is an extra term on top of global Xview.
    
        这里分成两个互不阻断的部分：
            1. train-view patch repair
            2. pseudo-view patch repair
    
        注意：
            当前 train view 没有 patch 时，不能直接 return；
            因为 pseudo patch repair 仍然可能需要执行。
        """
        if not self.patch_repair or self.patch_db is None:
            return loss
        
        if (iteration - self.repair_start_iter) % self.patch_repair_interval != 0:
            return loss
    
        repair_warmup = min(
            1.0,
            max(0.0, float(iteration - self.repair_start_iter) / 500.0),
        )
    
        # ============================================================
        # 1. Train-view patch repair
        # ============================================================
        patches = self._get_view_patches(viewpoint_cam.uid)
    
        if len(patches) > 0:
            H, W = image.shape[-2], image.shape[-1]
    
            structure_patches = [
                p for p in patches
                if p.get("label") == "structure_error"
            ]
    
            blur_patches = [
                p for p in patches
                if p.get("label") == "blur"
            ]
    
            # -----------------------------
            # structure_error patch repair
            # -----------------------------
            if len(structure_patches) > 0:
                mask_s = boxes_to_mask(structure_patches, H, W, image.device)
    
                if mask_s.sum() > 16:
                    depth_prior = 255.0 - midas_depth_resized
    
                    loss_s = (
                        self.struct_photo_w * masked_l1(image, gt_image, mask_s)
                        + self.struct_depth_w * masked_depth_l1(
                            rendered_depth_2d,
                            depth_prior,
                            mask_s,
                        )
                    )
    
                    # Extra local masked Xview, on top of global Xview.
                    cam_j = None
                    pkg_j = None
                    inb = -1.0
    
                    if self._last_xview_iter == iteration:
                        cam_j = self._last_xview_cam
                        pkg_j = self._last_xview_pkg
                        inb = self._last_xview_inb
                    else:
                        cam_j, inb, _ = self.pick_xview_neighbor(
                            viewpoint_cam,
                            render_pkg,
                        )
    
                        if cam_j is not None and inb >= 0.10:
                            with torch.no_grad():
                                pkg_j = render(
                                    cam_j,
                                    gaussians,
                                    self.pipe,
                                    self.background,
                                )
    
                    if cam_j is not None and pkg_j is not None and inb >= 0.10:
                        loss_struct_xview = xview_reproj_depth_loss(
                            viewpoint_cam,
                            render_pkg,
                            cam_j,
                            pkg_j,
                            detach_j=True,
                            mask_i=mask_s,
                            tau_rel=0.10,
                            w_err_min=0.10,
                            n_samples=4096,
                            return_stats=False,
                            return_map=False,
                        )
    
                        if is_valid_loss(loss_struct_xview):
                            loss_s = loss_s + self.struct_xview_w * loss_struct_xview
    
                    loss = loss + repair_warmup * loss_s
    
            # -----------------------------
            # blur patch repair
            # -----------------------------
            if len(blur_patches) > 0:
                mask_b = boxes_to_mask(blur_patches, H, W, image.device)
    
                if mask_b.sum() > 16:
                    loss_b = (
                        self.blur_photo_w * masked_l1(image, gt_image, mask_b)
                        + self.blur_grad_w * masked_grad_l1(image, gt_image, mask_b)
                    )
    
                    loss = loss + repair_warmup * loss_b
    
            if iteration % 400 == 0:
                print(
                    f"[patch@{iteration}] "
                    f"uid={viewpoint_cam.uid} "
                    f"n={len(patches)} "
                    f"struct={len(structure_patches)} "
                    f"blur={len(blur_patches)} "
                    f"warmup={repair_warmup:.3f}"
                )
    
        # ============================================================
        # 2. Pseudo-view patch repair
        # ============================================================
        # 不在 helper 里额外加 200 轮频率控制。
        # 频率沿用 apply_pseudo_patch_repair_loss 内部自己的逻辑。
        if iteration >= self.pseudo_start_iter and self._patch_db_has_pseudo_structure():
            loss = apply_pseudo_patch_repair_loss(
                patch_db=self.patch_db,
                iteration=iteration,
                loss=loss,
                train_cams=self.train_cams,
                gaussians=gaussians,
                pipe=self.pipe,
                background=self.background,
            )
        
        return loss

    # ------------------------------------------------------------
    # 3. patch mining
    # ------------------------------------------------------------

    @torch.no_grad()
    def mine_patches_if_needed(self, iteration, gaussians):
        if not self.mine_patches:
            return None
    
        if iteration != self.patch_iter:
            return None
    
        include_pseudo = iteration in self.pseudo_iters
    
        return self._mine_patches(
            iteration=iteration,
            gaussians=gaussians,
            include_pseudo=include_pseudo,
        )

    @torch.no_grad()
    def _mine_patches(self, iteration, gaussians, include_pseudo=False):
        out_dir = os.path.join(self.scene.model_path, "patches", f"iter_{iteration}")
        os.makedirs(out_dir, exist_ok=True)

        patch_db = {
            "version": 1,
            "source_iter": int(iteration),
            "views": {},
        }

        for cam in self.train_cams:
            pkg_i = render(cam, gaussians, self.pipe, self.background)

            image = pkg_i["render"].detach().clamp(0, 1)
            gt = cam.original_image.to(image.device).detach().clamp(0, 1)

            H, W = image.shape[-2], image.shape[-1]

            # 1. photometric residual
            photo_map = torch.mean(torch.abs(image - gt), dim=0)

            # 2. depth residual
            rendered_depth = to_2d(pkg_i["depth"]).detach()

            midas = torch.tensor(cam.depth_image, device=image.device).float().squeeze()
            midas = F.interpolate(
                midas[None, None],
                size=(H, W),
                mode="bicubic",
                align_corners=False,
            )[0, 0]

            depth_prior = 255.0 - midas
            depth_map = torch.abs(robust01(rendered_depth) - robust01(depth_prior))

            # 3. alpha / hole map
            alpha = to_2d(pkg_i.get("rend_alpha", pkg_i.get("alpha", None)))
            if alpha is None:
                alpha_map = torch.zeros((H, W), device=image.device)
            else:
                alpha_map = 1.0 - alpha.detach().clamp(0, 1)

            # 4. xview residual map
            xview_map = torch.zeros((H, W), device=image.device)
            xview_uid = None
            xview_inb = -1.0

            cam_j, inb, _ = self.pick_xview_neighbor(cam, pkg_i)

            if cam_j is not None and inb >= 0.10:
                pkg_j = render(cam_j, gaussians, self.pipe, self.background)

                _, extra = xview_reproj_depth_loss(
                    cam,
                    pkg_i,
                    cam_j,
                    pkg_j,
                    detach_j=True,
                    n_samples=0,
                    return_stats=True,
                    return_map=True,
                )

                if isinstance(extra, dict) and "err_map" in extra:
                    xview_map = extra["err_map"].detach()

                xview_uid = int(cam_j.uid)
                xview_inb = float(inb)

            score = (
                0.40 * robust01(photo_map)
                + 0.35 * robust01(xview_map)
                + 0.20 * robust01(depth_map)
                + 0.05 * robust01(alpha_map)
            )

            proposals = propose_patches(score)

            entries = []

            for k, p in enumerate(proposals):
                box = p["bbox"]

                label, conf, cues = heuristic_patch_label(
                    photo_map,
                    xview_map,
                    depth_map,
                    alpha_map,
                    box,
                )

                montage_name = f"view_{cam.uid}_patch_{k}.png"
                montage_path = os.path.join(out_dir, montage_name)

                save_diagnostic_montage(
                    image=image,
                    gt_image=gt,
                    photo_map=photo_map,
                    depth_map=depth_map,
                    xview_map=xview_map,
                    alpha_map=alpha_map,
                    box=box,
                    out_path=montage_path,
                )

                entries.append({
                    "bbox": [int(v) for v in box],
                    "score": float(p["score"]),
                    "label": label,
                    "confidence": float(conf),
                    "label_source": "heuristic_pending_qwen",
                    "diagnostic_path": montage_path,
                    "xview_uid": xview_uid,
                    "xview_inb": xview_inb,
                    "cues": cues,
                })

            patch_db["views"][str(cam.uid)] = entries
      
        # ------------------------------------------------------------
        # Scheme-B: pseudo-view generalization-risk patches
        # ------------------------------------------------------------
        if include_pseudo:
            pseudo_views = mine_pseudo_patches(
                scene=self.scene,
                gaussians=gaussians,
                pipe=self.pipe,
                background=self.background,
                iteration=iteration,
                out_dir=out_dir,
            )
        
            for k, v in pseudo_views.items():
                patch_db["views"][str(k)] = v
        else:
            print(f"[pseudo_mining@{iteration}] skipped")
                
        json_path = os.path.join(out_dir, "patches.json")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(patch_db, f, indent=2, ensure_ascii=False)

        print(f"[patch_mining] saved patches to {json_path}")
        return json_path