import torch
import torch.nn.functional as F

# --------- keep your helpers ----------
def _stable_normal_u8_to_float(n_u8: torch.Tensor, device="cuda"):
    # accept CHW or HWC
    if n_u8.dim() == 3 and n_u8.shape[0] != 3 and n_u8.shape[-1] == 3:
        n_u8 = n_u8.permute(2, 0, 1).contiguous()  # HWC -> CHW

    n = n_u8.to(device, non_blocking=True).float() / 255.0
    n = n * 2.0 - 1.0
    n = torch.nan_to_num(n, nan=0.0, posinf=0.0, neginf=0.0)
    n = F.normalize(n, dim=0, eps=1e-6)
    return n


def _erode_mask(m: torch.Tensor, k: int = 3):
    if k <= 1:
        return m
    x = m[None, None].float()
    inv = 1.0 - x
    pooled = F.max_pool2d(inv, kernel_size=k, stride=1, padding=k//2)
    return (pooled < 0.5)[0, 0]

def _ramp(iteration: int, start: int, end: int) -> float:
    if end <= start:
        return 1.0 if iteration >= start else 0.0
    t = (iteration - start) / max(1, (end - start))
    return float(max(0.0, min(1.0, t)))

# --------- NEW: per-view sign cache (avoid jitter) ----------
_SN_SIGN_CACHE = {}  # uid(int) -> sign_id(int 0..7)

@torch.no_grad()
def _align_prior_by_8sign_cached(n_prior_3hw: torch.Tensor,
                                n_pred_3hw: torch.Tensor,
                                mask_hw: torch.Tensor,
                                uid: int = -1,
                                refresh: bool = False):
    """
    Choose best sign flip among 8 combos (±x,±y,±z), score = mean dot on mask.
    Cache per-view so sign doesn't jitter across iterations.
    """
    device = n_prior_3hw.device
    dtype  = n_prior_3hw.dtype

    signs = torch.tensor(
        [
            [ 1, 1, 1],
            [ 1, 1,-1],
            [ 1,-1, 1],
            [ 1,-1,-1],
            [-1, 1, 1],
            [-1, 1,-1],
            [-1,-1, 1],
            [-1,-1,-1],
        ],
        device=device, dtype=dtype
    ).view(8, 3, 1, 1)

    # If cached and not refreshing, use cached sign id
    if (uid is not None) and (uid in _SN_SIGN_CACHE) and (not refresh):
        sid = int(_SN_SIGN_CACHE[uid])
        return n_prior_3hw * signs[sid], sid

    # Otherwise compute best sid
    cand = n_prior_3hw[None] * signs                 # [8,3,H,W]
    dot  = (cand * n_pred_3hw[None]).sum(dim=1)      # [8,H,W]

    w = mask_hw.to(dtype)[None]                      # [1,H,W]
    denom = w.sum().clamp_min(1.0)
    score = (dot * w).sum(dim=(1, 2)) / denom        # [8]

    sid = int(torch.argmax(score).item())
    if uid is not None and uid >= 0:
        _SN_SIGN_CACHE[uid] = sid
    return cand[sid], sid


def stable_normal_prior_term(
    iteration: int,
    args,
    viewpoint_cam,
    render_pkg: dict,
    alpha_th: float = 0.2,
    erode_k: int = 3,
    debug: bool = False,
):
    """
    TSGS-style stable normal loss:
      - use pixel-level valid mask (alpha + erosion + cosine threshold)
      - normalize by valid pixel count (avoid being diluted)
      - stable ramp weight, no dot_mean->conf weight jitter
      - per-view cached 8-sign alignment (avoid sign jitter)
    Return: already-weighted scalar tensor on CUDA.
    """
    device = "cuda"

    sn_w     = float(getattr(args, "stable_normal_weight", 0.0))
    sn_start = int(getattr(args, "stable_normal_start_iter", 0))
    sn_end   = int(getattr(args, "stable_normal_ramp_end", sn_start))

    if sn_w <= 0.0 or iteration < sn_start:
        return torch.zeros([], device=device)

    n_u8 = getattr(viewpoint_cam, "normal_image", None)
    if n_u8 is None:
        return torch.zeros([], device=device)

    # ---- pred normal: prefer view-space (TSGS uses camera/local space) ----
    if "rend_normal_view" in render_pkg:
        n_pred = render_pkg["rend_normal_view"]
    else:
        n_pred = render_pkg["rend_normal"]
    n_pred = F.normalize(n_pred, dim=0, eps=1e-6)  # [3,H,W]

    # ---- prior normal ----
    n_prior = _stable_normal_u8_to_float(n_u8, device=device)  # [3,h,w] in [-1,1], normalized

    # resize prior to pred
    H, W = n_pred.shape[1], n_pred.shape[2]
    if (n_prior.shape[1] != H) or (n_prior.shape[2] != W):
        n_prior = F.interpolate(n_prior[None], size=(H, W), mode="bilinear", align_corners=False)[0]
        n_prior = F.normalize(n_prior, dim=0, eps=1e-6)

    # ---- alpha mask (same idea as TSGS: only supervise reliable surface) ----
    alpha = render_pkg.get("rend_alpha", None)
    if alpha is not None:
        a2d = alpha[0] if alpha.dim() == 3 else alpha
        mask = (a2d > alpha_th)
    else:
        mask = torch.ones((H, W), device=device, dtype=torch.bool)

    mask = _erode_mask(mask, k=erode_k)
    if int(mask.sum().item()) < 1024:
        return torch.zeros([], device=device)

    # ---- per-view cached sign alignment (TSGS hard-coded -gt_normal; we generalize) ----
    uid = int(getattr(viewpoint_cam, "uid", -1))
    refresh_sign = bool(getattr(args, "stable_normal_refresh_sign", False))
    n_prior_aligned, sid = _align_prior_by_8sign_cached(
        n_prior_3hw=n_prior,
        n_pred_3hw=n_pred,
        mask_hw=mask,
        uid=uid,
        refresh=refresh_sign,
    )

    # ---- cosine & TSGS-style valid mask: (cos > thr) ----
    cos = (n_pred * n_prior_aligned).sum(dim=0).clamp(-1.0, 1.0)

    # threshold schedule (mimic TSGS):
    # early: use all masked pixels
    # later: keep only pixels with decent agreement
    thr_iter = int(getattr(args, "stable_normal_cos_threshold_iter", sn_start + 1000))
    thr_val  = float(getattr(args, "stable_normal_cos_threshold", 0.1))
    if iteration >= thr_iter:
        valid = mask & (cos > thr_val)
    else:
        valid = mask

    n_valid = int(valid.sum().item())
    if n_valid < 2048:
        # too few reliable pixels -> skip to avoid noisy gradients
        if debug:
            print(f"[stable_normal@{iteration}] skip (n_valid={n_valid}) thr={thr_val} sid={sid}")
        return torch.zeros([], device=device)

    # ---- loss normalized by valid pixels (TSGS style) ----
    # cosine loss: 1 - cos
    sn_loss = (1.0 - cos)[valid].sum() / (valid.sum().float() + 1e-6)

    # ---- ramp weight (stable, no conf jitter) ----
    ramp = _ramp(iteration, sn_start, sn_end)
    w_eff = sn_w * ramp

    if debug:
        dot_mean_all = cos[mask].mean().item()
        dot_mean_val = cos[valid].mean().item()
        print(
            f"[stable_normal@{iteration}] sid={sid} "
            f"dot_mean(mask)={dot_mean_all:.3f} dot_mean(valid)={dot_mean_val:.3f} "
            f"n_valid={n_valid} thr={'None' if iteration < thr_iter else thr_val} "
            f"w_eff={w_eff:.5g} sn_loss={sn_loss.item():.6f}"
        )

    return (n_pred.new_tensor(w_eff) * sn_loss)
