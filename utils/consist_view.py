import math
import torch
import torch.nn.functional as F


# =========================
# Camera matrix cache
# =========================

_K_cache = {}
_Kinv_cache = {}
_W2C_cache = {}
_C2W_cache = {}


def _cam_uid(cam):
    if hasattr(cam, "uid"):
        try:
            return int(cam.uid)
        except Exception:
            pass
    return int(id(cam))


def _device_key(device):
    dev = torch.device(device)
    if dev.type == "cuda":
        idx = dev.index
        if idx is None:
            idx = torch.cuda.current_device()
        return (dev.type, idx)
    return (dev.type, None)


def _cache_key(cam, device, dtype):
    H, W = _get_hw(cam)
    fovx = float(getattr(cam, "FoVx", 0.0))
    fovy = float(getattr(cam, "FoVy", 0.0))

    # 加入 id(cam)，避免多个 scene 中 uid 重复导致 cache 污染
    return (
        _cam_uid(cam),
        int(id(cam)),
        _device_key(device),
        str(dtype),
        int(H),
        int(W),
        float(fovx),
        float(fovy),
    )


def clear_consist_view_cache():
    _K_cache.clear()
    _Kinv_cache.clear()
    _W2C_cache.clear()
    _C2W_cache.clear()


def _get_hw(cam):
    if hasattr(cam, "original_image") and cam.original_image is not None:
        H, W = cam.original_image.shape[-2], cam.original_image.shape[-1]
        return int(H), int(W)

    if hasattr(cam, "image") and cam.image is not None:
        H, W = cam.image.shape[-2], cam.image.shape[-1]
        return int(H), int(W)

    if hasattr(cam, "depth_image") and cam.depth_image is not None:
        d = cam.depth_image
        if len(d.shape) == 3:
            H, W = d.shape[-2], d.shape[-1]
        else:
            H, W = d.shape[0], d.shape[1]
        return int(H), int(W)

    raise RuntimeError("Cannot infer camera image size.")


def _get_K_from_fov(cam, device="cuda"):
    H, W = _get_hw(cam)

    fx = 0.5 * W / math.tan(float(cam.FoVx) * 0.5)
    fy = 0.5 * H / math.tan(float(cam.FoVy) * 0.5)

    # GraphDECO / 3DGS 常见写法
    cx = 0.5 * (W - 1)
    cy = 0.5 * (H - 1)

    K = torch.tensor(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=torch.float32,
    )
    return K


def _get_W2C(cam, device="cuda"):
    if hasattr(cam, "world_view_transform"):
        W2C = cam.world_view_transform
        if W2C.shape == (4, 4):
            # GraphDECO 常见存法：
            # world_view_transform = getWorld2View2(...).T
            W2C = W2C.transpose(0, 1).contiguous()
        return W2C.to(device=device, dtype=torch.float32)

    # 兜底：COLMAP convention, x_cam = R x_world + T
    R = torch.tensor(cam.R, device=device, dtype=torch.float32)
    T = torch.tensor(cam.T, device=device, dtype=torch.float32).view(3, 1)

    W2C = torch.eye(4, device=device, dtype=torch.float32)
    W2C[:3, :3] = R
    W2C[:3, 3:4] = T
    return W2C


@torch.no_grad()
def get_K_Kinv(cam, device="cuda", dtype=torch.float32):
    key = _cache_key(cam, device, dtype)
    K = _K_cache.get(key, None)
    Ki = _Kinv_cache.get(key, None)

    if K is None or Ki is None:
        K = _get_K_from_fov(cam, device=device).to(dtype=dtype)
        Ki = torch.inverse(K)
        _K_cache[key] = K
        _Kinv_cache[key] = Ki

    return K, Ki


@torch.no_grad()
def get_W2C_C2W(cam, device="cuda", dtype=torch.float32):
    key = _cache_key(cam, device, dtype)
    W2C = _W2C_cache.get(key, None)
    C2W = _C2W_cache.get(key, None)

    if W2C is None or C2W is None:
        W2C = _get_W2C(cam, device=device).to(dtype=dtype)
        C2W = torch.inverse(W2C)
        _W2C_cache[key] = W2C
        _C2W_cache[key] = C2W

    return W2C, C2W


# =========================
# Tensor helpers
# =========================

def _as_2d(x):
    if x is None:
        return None

    if x.dim() == 4:
        # [B, C, H, W]
        return x[0, 0]

    if x.dim() == 3:
        # [1, H, W] or [C, H, W]
        return x[0]

    if x.dim() == 2:
        return x

    return x.squeeze()


def _get_pkg_2d(pkg, key, fallback_key=None):
    x = pkg.get(key, None)
    if x is None and fallback_key is not None:
        x = pkg.get(fallback_key, None)
    if x is None:
        return None
    return _as_2d(x)


def _prepare_mask(mask, H, W, device, mode="nearest"):
    if mask is None:
        return None

    mask = _as_2d(mask).to(device=device)

    if mask.shape[-2:] != (H, W):
        m = mask.float()[None, None]

        if mode == "nearest":
            mask = F.interpolate(m, size=(H, W), mode="nearest")[0, 0]
        else:
            mask = F.interpolate(
                m,
                size=(H, W),
                mode=mode,
                align_corners=False,
            )[0, 0]

    return mask.float()


def _sample_map_bilinear(x2d, u, v):
    """
    x2d: [H, W]
    u, v: [N], pixel coordinates in x/y order.
    return: [N]
    """
    H, W = x2d.shape

    gx = 2.0 * u / max(W - 1, 1) - 1.0
    gy = 2.0 * v / max(H - 1, 1) - 1.0

    grid = torch.stack([gx, gy], dim=-1).view(1, -1, 1, 2)

    y = F.grid_sample(
        x2d.float().view(1, 1, H, W),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )

    return y.view(-1)


def _sample_map_nearest(x2d, u, v):
    H, W = x2d.shape
    ui = torch.round(u).long().clamp(0, W - 1)
    vi = torch.round(v).long().clamp(0, H - 1)
    return x2d[vi, ui]


def _charbonnier(x, eps=1e-3):
    return torch.sqrt(x * x + eps * eps)


def _zero_xview_return(depth_i, reason, return_stats=False, return_map=False):
    loss = depth_i.new_tensor(0.0)

    if not return_stats and not return_map:
        return loss

    H, W = depth_i.shape
    extra = {
        "reason": reason,
        "N": 0,
        "N_valid_total": 0,
        "N_inb": 0,
        "base_count": 0,
        "loss": 0.0,
    }

    if return_map:
        extra["err_map"] = torch.zeros((H, W), device=depth_i.device, dtype=torch.float32)
        extra["weight_map"] = torch.zeros((H, W), device=depth_i.device, dtype=torch.float32)
        extra["count_map"] = torch.zeros((H, W), device=depth_i.device, dtype=torch.float32)

    return loss, extra


# =========================
# Quick overlap pre-check
# =========================

@torch.no_grad()
def quick_inb_ratio(
    cam_i,
    pkg_i,
    cam_j,
    n=512,
    alpha_th=0.2,
    z_eps=1e-6,
    mask_i=None,
):
    """
    从 view-i 的 rendered depth / alpha 中抽样像素，
    反投影到世界，再投影到 view-j。
    返回落在 view-j 视野内且 z>0 的比例。

    这个函数只用于快速选择 neighbor view，不参与反传。
    """
    depth_i = _get_pkg_2d(pkg_i, "depth")
    H, W = depth_i.shape
    Hj, Wj = _get_hw(cam_j)
    device = depth_i.device

    alpha_i = _get_pkg_2d(pkg_i, "rend_alpha", "alpha")

    valid = torch.isfinite(depth_i) & (depth_i > z_eps)

    if alpha_i is not None:
        valid = valid & (alpha_i > alpha_th)

    if mask_i is not None:
        mi = _prepare_mask(mask_i, H, W, device=device)
        valid = valid & (mi > 0.5)

    idx = torch.nonzero(valid, as_tuple=False)  # [M, 2]
    if idx.shape[0] < 64:
        return 0.0

    if n is not None and n > 0 and idx.shape[0] > n:
        perm = torch.randperm(idx.shape[0], device=device)[:n]
        idx = idx[perm]

    y = idx[:, 0].float()
    x = idx[:, 1].float()
    z = depth_i[idx[:, 0], idx[:, 1]]

    K_i, K_i_inv = get_K_Kinv(cam_i, device=device, dtype=torch.float32)
    K_j, _ = get_K_Kinv(cam_j, device=device, dtype=torch.float32)

    W2C_i, C2W_i = get_W2C_C2W(cam_i, device=device, dtype=torch.float32)
    W2C_j, _ = get_W2C_C2W(cam_j, device=device, dtype=torch.float32)

    ones = torch.ones_like(x)
    pix_i = torch.stack([x, y, ones], dim=1)  # [N, 3]

    xyz_i = (pix_i @ K_i_inv.T) * z[:, None]  # [N, 3]

    xyz_i_h = torch.cat(
        [xyz_i, torch.ones((xyz_i.shape[0], 1), device=device, dtype=xyz_i.dtype)],
        dim=1,
    )

    xyz_w = xyz_i_h @ C2W_i.T
    xyz_j = (xyz_w @ W2C_j.T)[:, :3]

    z_j = xyz_j[:, 2]
    z_safe = z_j.clamp_min(z_eps)

    uvw = xyz_j @ K_j.T
    u = uvw[:, 0] / z_safe
    v = uvw[:, 1] / z_safe

    inb = (
        (z_j > z_eps)
        & (u >= 0)
        & (u <= (Wj - 1))
        & (v >= 0)
        & (v <= (Hj - 1))
    )

    return float(inb.float().mean().item())


# =========================
# Main Xview loss
# =========================

def xview_reproj_depth_loss(
    cam_i,
    pkg_i,
    cam_j,
    pkg_j,
    tau_rel=0.08,
    alpha_th=0.2,
    n_samples=8192,
    detach_j=True,
    debug=False,
    debug_prefix="xview",

    # optional source / target masks
    mask_i=None,
    mask_j=None,
    mask_th=0.5,

    # sampling
    use_bilinear=True,

    # weights
    tau_occ=0.02,
    temp_occ=0.01,
    alpha_vis_th=0.3,
    temp_vis=0.05,
    sigma_dist=5e-4,
    w_err_min=0.05,

    # robust loss
    charb_eps=1e-3,
    z_eps=1e-6,

    # validity thresholds
    min_valid=128,
    min_inb=128,
    min_base=64,

    # diagnostics
    return_stats=False,
    return_map=False,
):
    """
    Cross-view rendered-depth reprojection consistency.

    view-i:
        rendered depth_i -> backproject -> world -> project to view-j

    compare:
        projected source depth z_ij
        vs.
        sampled target rendered depth depth_j(u2, v2)

    Main improvements over the original version:
        1. target maps are bilinearly sampled by default;
        2. target view can be detached by default;
        3. large residuals are not completely suppressed because w_err has clamp_min;
        4. supports mask_i / mask_j for patch-conditioned Xview;
        5. supports return_map for patch mining;
        6. supports return_stats for debugging.
    """

    depth_i = _get_pkg_2d(pkg_i, "depth")
    depth_j = _get_pkg_2d(pkg_j, "depth")

    if depth_i is None or depth_j is None:
        raise RuntimeError("pkg_i/pkg_j must contain key 'depth'.")

    device = depth_i.device
    H, W = depth_i.shape
    Hj, Wj = depth_j.shape

    if detach_j:
        depth_j = depth_j.detach()

    alpha_i = _get_pkg_2d(pkg_i, "rend_alpha", "alpha")
    alpha_j = _get_pkg_2d(pkg_j, "rend_alpha", "alpha")

    if alpha_j is not None and detach_j:
        alpha_j = alpha_j.detach()

    dist_j = _get_pkg_2d(pkg_j, "rend_dist")
    if dist_j is not None and detach_j:
        dist_j = dist_j.detach()

    # -----------------------------
    # source valid pixels
    # -----------------------------
    valid = torch.isfinite(depth_i) & (depth_i > z_eps)

    if alpha_i is not None:
        valid = valid & (alpha_i > alpha_th)

    if mask_i is not None:
        mi = _prepare_mask(mask_i, H, W, device=device)
        valid = valid & (mi > mask_th)

    idx_all = torch.nonzero(valid, as_tuple=False)  # [M, 2]
    N_valid_total = int(idx_all.shape[0])

    if N_valid_total < min_valid:
        return _zero_xview_return(
            depth_i,
            reason="too_few_source_valid",
            return_stats=return_stats,
            return_map=return_map,
        )

    idx = idx_all

    # n_samples <= 0 or None means use all valid pixels.
    if n_samples is not None and n_samples > 0 and idx.shape[0] > n_samples:
        perm = torch.randperm(idx.shape[0], device=device)[:int(n_samples)]
        idx = idx[perm]

    N = int(idx.shape[0])

    src_y_long = idx[:, 0].long()
    src_x_long = idx[:, 1].long()

    v = src_y_long.float()
    u = src_x_long.float()
    z = depth_i[src_y_long, src_x_long]

    # -----------------------------
    # camera matrices
    # -----------------------------
    K_i, K_i_inv = get_K_Kinv(cam_i, device=device, dtype=torch.float32)
    K_j, _ = get_K_Kinv(cam_j, device=device, dtype=torch.float32)

    W2C_i, C2W_i = get_W2C_C2W(cam_i, device=device, dtype=torch.float32)
    W2C_j, _ = get_W2C_C2W(cam_j, device=device, dtype=torch.float32)

    # -----------------------------
    # backproject source pixels to camera-i
    # -----------------------------
    ones = torch.ones_like(u)
    pix_i = torch.stack([u, v, ones], dim=1)  # [N, 3]

    xyz_i = (pix_i @ K_i_inv.T) * z[:, None]  # [N, 3]

    xyz_i_h = torch.cat(
        [
            xyz_i,
            torch.ones((xyz_i.shape[0], 1), device=device, dtype=xyz_i.dtype),
        ],
        dim=1,
    )  # [N, 4]

    # camera-i -> world -> camera-j
    xyz_w = xyz_i_h @ C2W_i.T
    xyz_j = (xyz_w @ W2C_j.T)[:, :3]  # [N, 3]

    z_ij_raw = xyz_j[:, 2]
    z_ij = z_ij_raw.clamp_min(z_eps)

    # project to view-j
    uvw = xyz_j @ K_j.T
    u2 = uvw[:, 0] / z_ij
    v2 = uvw[:, 1] / z_ij

    inb = (
        (z_ij_raw > z_eps)
        & (u2 >= 0)
        & (u2 <= (Wj - 1))
        & (v2 >= 0)
        & (v2 <= (Hj - 1))
    )

    N_inb = int(inb.sum().item())

    if debug:
        print(
            f"[{debug_prefix}] "
            f"N_valid_total={N_valid_total} "
            f"N_sample={N} "
            f"inb={N_inb}/{N} ({N_inb / max(N, 1):.3f})"
        )

    if N_inb < min_inb:
        return _zero_xview_return(
            depth_i,
            reason="too_few_inbound",
            return_stats=return_stats,
            return_map=return_map,
        )

    # keep only in-bound samples
    src_y_long = src_y_long[inb]
    src_x_long = src_x_long[inb]

    u2 = u2[inb]
    v2 = v2[inb]
    z_ij = z_ij[inb]

    # -----------------------------
    # sample target maps
    # -----------------------------
    if use_bilinear:
        dj = _sample_map_bilinear(depth_j, u2, v2)
    else:
        dj = _sample_map_nearest(depth_j, u2, v2)

    aj = None
    if alpha_j is not None:
        if use_bilinear:
            aj = _sample_map_bilinear(alpha_j, u2, v2)
        else:
            aj = _sample_map_nearest(alpha_j, u2, v2)

    dj_dist = None
    if dist_j is not None:
        if use_bilinear:
            dj_dist = _sample_map_bilinear(dist_j, u2, v2)
        else:
            dj_dist = _sample_map_nearest(dist_j, u2, v2)

    mj_val = None
    if mask_j is not None:
        mj = _prepare_mask(mask_j, Hj, Wj, device=device)
        if use_bilinear:
            mj_val = _sample_map_bilinear(mj, u2, v2)
        else:
            mj_val = _sample_map_nearest(mj, u2, v2)

    # -----------------------------
    # base validity
    # -----------------------------
    base = (
        torch.isfinite(dj)
        & torch.isfinite(z_ij)
        & (dj > z_eps)
        & (z_ij > z_eps)
    )

    if mj_val is not None:
        base = base & (mj_val > mask_th)

    base_count = int(base.sum().item())

    if base_count < min_base:
        return _zero_xview_return(
            depth_i,
            reason="too_few_base_valid",
            return_stats=return_stats,
            return_map=return_map,
        )

    # apply base mask
    src_y_long = src_y_long[base]
    src_x_long = src_x_long[base]

    dj = dj[base]
    z_ij = z_ij[base]

    if aj is not None:
        aj = aj[base]

    if dj_dist is not None:
        dj_dist = dj_dist[base]

    # -----------------------------
    # residual
    # -----------------------------
    rel = (dj - z_ij) / z_ij.clamp_min(z_eps)
    err = rel.abs()

    # -----------------------------
    # reliability weights
    # -----------------------------
    # Occlusion weight:
    # rel < 0 means target rendered surface is closer than projected source point.
    # This is often an occlusion case, so downweight it softly.
    w_occ = torch.sigmoid((rel + tau_occ) / temp_occ).detach()

    if aj is not None:
        w_vis = torch.sigmoid((aj - alpha_vis_th) / temp_vis).detach()
    else:
        w_vis = torch.ones_like(rel)

    if dj_dist is not None and sigma_dist is not None and sigma_dist > 0:
        dj_dist = dj_dist.clamp_min(0.0)
        w_dist = torch.exp(-dj_dist / sigma_dist).detach()
    else:
        w_dist = torch.ones_like(rel)

    if tau_rel is not None and tau_rel > 0:
        w_err = torch.exp(-err / tau_rel).clamp_min(w_err_min).detach()
    else:
        w_err = torch.ones_like(rel)

    w = w_occ * w_vis * w_dist * w_err
    w_sum = w.sum()

    if not torch.isfinite(w_sum) or w_sum.detach().item() < 1e-6:
        return _zero_xview_return(
            depth_i,
            reason="too_small_weight_sum",
            return_stats=return_stats,
            return_map=return_map,
        )

    loss_per = _charbonnier(rel, eps=charb_eps)
    loss = (w * loss_per).sum() / (w_sum + 1e-6)

    # -----------------------------
    # diagnostics
    # -----------------------------
    extra = None

    if return_stats or return_map or debug:
        with torch.no_grad():
            eff_n = (w_sum * w_sum / (w.square().sum() + 1e-6)).item()

            stats = {
                "reason": "ok",
                "N": int(N),
                "N_valid_total": int(N_valid_total),
                "N_inb": int(N_inb),
                "base_count": int(base_count),
                "inb_ratio": float(N_inb / max(N, 1)),
                "base_ratio": float(base_count / max(N_inb, 1)),
                "loss": float(loss.detach().item()),

                "rel_min": float(err.min().item()),
                "rel_med": float(err.median().item()),
                "rel_mean": float(err.mean().item()),
                "rel_max": float(err.max().item()),

                "w_min": float(w.min().item()),
                "w_med": float(w.median().item()),
                "w_mean": float(w.mean().item()),
                "w_max": float(w.max().item()),
                "w_sum": float(w_sum.item()),
                "w_eff_num": float(eff_n),
            }

            if aj is not None:
                stats["aj_min"] = float(aj.min().item())
                stats["aj_med"] = float(aj.median().item())
                stats["aj_max"] = float(aj.max().item())

            if dj_dist is not None:
                stats["dist_min"] = float(dj_dist.min().item())
                stats["dist_med"] = float(dj_dist.median().item())
                stats["dist_max"] = float(dj_dist.max().item())

            if debug:
                print(
                    f"[{debug_prefix}] "
                    f"base={base_count}/{N_inb} "
                    f"loss={stats['loss']:.6f} "
                    f"rel_abs min/med/mean/max="
                    f"{stats['rel_min']:.6f}/"
                    f"{stats['rel_med']:.6f}/"
                    f"{stats['rel_mean']:.6f}/"
                    f"{stats['rel_max']:.6f} "
                    f"w mean/sum/effN="
                    f"{stats['w_mean']:.6f}/"
                    f"{stats['w_sum']:.3f}/"
                    f"{stats['w_eff_num']:.1f}"
                )

                if aj is not None:
                    print(
                        f"[{debug_prefix}] "
                        f"aj min/med/max="
                        f"{stats['aj_min']:.4g}/"
                        f"{stats['aj_med']:.4g}/"
                        f"{stats['aj_max']:.4g}"
                    )

                if dj_dist is not None:
                    print(
                        f"[{debug_prefix}] "
                        f"dist min/med/max="
                        f"{stats['dist_min']:.4g}/"
                        f"{stats['dist_med']:.4g}/"
                        f"{stats['dist_max']:.4g}"
                    )

            extra = stats

    # -----------------------------
    # source-view residual map for patch mining
    # -----------------------------
    if return_map:
        with torch.no_grad():
            err_sum_map = torch.zeros((H, W), device=device, dtype=torch.float32)
            w_sum_map = torch.zeros((H, W), device=device, dtype=torch.float32)
            cnt_map = torch.zeros((H, W), device=device, dtype=torch.float32)

            err_val = err.detach().float()
            w_val = w.detach().float()
            one_val = torch.ones_like(err_val)

            err_sum_map.index_put_(
                (src_y_long, src_x_long),
                err_val,
                accumulate=True,
            )
            w_sum_map.index_put_(
                (src_y_long, src_x_long),
                w_val,
                accumulate=True,
            )
            cnt_map.index_put_(
                (src_y_long, src_x_long),
                one_val,
                accumulate=True,
            )

            err_map = err_sum_map / (cnt_map + 1e-6)
            weight_map = w_sum_map / (cnt_map + 1e-6)

            if extra is None:
                extra = {}

            extra["err_map"] = err_map
            extra["weight_map"] = weight_map
            extra["count_map"] = cnt_map

    if return_stats or return_map:
        return loss, extra

    return loss