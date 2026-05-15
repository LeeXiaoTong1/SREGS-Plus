# utils/xview_helper.py

import math
import torch
import torch.nn.functional as F

from utils.consist_view import _get_hw, _get_K_from_fov, _get_W2C, quick_inb_ratio


@torch.no_grad()
def get_alpha_map(render_pkg):
    alpha = render_pkg.get("rend_alpha", None)
    if alpha is None:
        return torch.ones_like(render_pkg["depth"][0])
    return alpha[0] if alpha.dim() == 3 else alpha


@torch.no_grad()
def robust_norm_map(x, valid_mask):
    if valid_mask.sum() < 16:
        return torch.zeros_like(x)
    xv = x[valid_mask]
    med = xv.median()
    mad = (xv - med).abs().median().clamp_min(1e-6)
    return (x - med) / (1.4826 * mad)


def weighted_corrcoef(x, y, w, eps=1e-8):
    valid = torch.isfinite(x) & torch.isfinite(y) & torch.isfinite(w) & (w > 0)
    if valid.sum() < 64:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)

    x = x[valid]
    y = y[valid]
    w = w[valid]

    wsum = w.sum().clamp_min(eps)
    mx = (w * x).sum() / wsum
    my = (w * y).sum() / wsum

    xv = x - mx
    yv = y - my

    cov = (w * xv * yv).sum() / wsum
    varx = (w * xv * xv).sum() / wsum
    vary = (w * yv * yv).sum() / wsum

    if varx < eps or vary < eps:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)

    return cov / torch.sqrt(varx * vary + eps)


def weighted_charbonnier(x, y, w, eps=1e-3):
    valid = torch.isfinite(x) & torch.isfinite(y) & torch.isfinite(w) & (w > 0)
    if valid.sum() < 64:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)

    diff = torch.sqrt((x[valid] - y[valid]) ** 2 + eps ** 2)
    ww = w[valid]
    return (ww * diff).sum() / ww.sum().clamp_min(1e-6)


@torch.no_grad()
def sample_map2d(map2d, u, v):
    H, W = map2d.shape[-2:]
    gx = 2.0 * u / max(W - 1, 1) - 1.0
    gy = 2.0 * v / max(H - 1, 1) - 1.0
    grid = torch.stack([gx, gy], dim=-1)[None, :, None, :]  # [1,N,1,2]

    val = F.grid_sample(
        map2d[None, None],
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return val[0, 0, :, 0]


def reproject_pixels_i_to_j(cam_i, cam_j, x, y, z):
    """
    将 cam_i 中的像素 (x,y,z) 重投影到 cam_j
    x,y,z: [N]
    return: u2, v2, z2, inb
    """
    device = z.device
    dtype = z.dtype

    K_i = _get_K_from_fov(cam_i).to(device=device, dtype=dtype)
    W2C_i = _get_W2C(cam_i).to(device=device, dtype=dtype)
    C2W_i = torch.inverse(W2C_i)

    K_j = _get_K_from_fov(cam_j).to(device=device, dtype=dtype)
    W2C_j = _get_W2C(cam_j).to(device=device, dtype=dtype)

    x = x.to(device=device, dtype=dtype)
    y = y.to(device=device, dtype=dtype)

    Xc = (x - K_i[0, 2]) / K_i[0, 0] * z
    Yc = (y - K_i[1, 2]) / K_i[1, 1] * z

    cam_i_h = torch.stack([Xc, Yc, z, torch.ones_like(z)], dim=-1)  # [N,4]
    world_h = (C2W_i @ cam_i_h.t()).t()
    cam_j_xyz = (W2C_j @ world_h.t()).t()[:, :3]

    z2 = cam_j_xyz[:, 2]
    u2 = K_j[0, 0] * cam_j_xyz[:, 0] / z2.clamp_min(1e-6) + K_j[0, 2]
    v2 = K_j[1, 1] * cam_j_xyz[:, 1] / z2.clamp_min(1e-6) + K_j[1, 2]

    H2, W2 = _get_hw(cam_j)
    inb = (z2 > 1e-6) & (u2 >= 0) & (u2 <= W2 - 1) & (v2 >= 0) & (v2 <= H2 - 1)
    return u2, v2, z2, inb


@torch.no_grad()
def build_xview_importance_map(render_pkg, prior_depth):
    """
    让 xview 聚焦在更有价值的区域：
    1) prior-render 深度冲突
    2) 深度边界
    3) 中等 alpha 区域
    4) 法向不一致区域
    """
    depth = render_pkg["depth"][0]
    alpha = get_alpha_map(render_pkg).clamp(0.0, 1.0)

    valid = (
        torch.isfinite(depth)
        & torch.isfinite(prior_depth)
        & (depth > 1e-6)
    )

    rend_inv = 1.0 / depth.clamp_min(1e-6)
    p = robust_norm_map(prior_depth, valid)
    r = robust_norm_map(rend_inv, valid)

    e_dp = (r - p).abs()
    e_dp[~valid] = 0.0
    if valid.sum() >= 16:
        q = e_dp[valid].quantile(0.90).clamp_min(1e-6)
        e_dp = (e_dp / q).clamp(0.0, 1.0)
    else:
        e_dp.zero_()

    grad = torch.zeros_like(depth)
    gx = (depth[:, 1:] - depth[:, :-1]).abs()
    gy = (depth[1:, :] - depth[:-1, :]).abs()
    grad[:, :-1] += gx
    grad[:-1, :] += gy
    grad[~valid] = 0.0

    if valid.sum() >= 16:
        qg = grad[valid].quantile(0.90).clamp_min(1e-6)
        grad = (grad / qg).clamp(0.0, 1.0)
    else:
        grad.zero_()

    # 中等 alpha 区域权重大，避免只盯着 alpha≈1 的稳定区
    a_mid = (4.0 * alpha * (1.0 - alpha)).clamp(0.0, 1.0)
    a_mix = 0.75 * a_mid + 0.25 * alpha

    if ("rend_normal" in render_pkg) and ("surf_normal" in render_pkg):
        rn = F.normalize(render_pkg["rend_normal"], dim=0)
        sn = F.normalize(render_pkg["surf_normal"], dim=0)
        nerr = (1.0 - (rn * sn).sum(dim=0)).clamp(0.0, 2.0) * 0.5
        if valid.sum() >= 16:
            qn = nerr[valid].quantile(0.90).clamp_min(1e-6)
            nerr = (nerr / qn).clamp(0.0, 1.0)
        else:
            nerr.zero_()
    else:
        nerr = torch.zeros_like(depth)

    imp = (
        0.45 * e_dp +
        0.25 * grad +
        0.20 * a_mix +
        0.10 * nerr
    )

    imp = imp * valid.float()
    if imp.sum() <= 1e-8:
        imp = valid.float()

    imp = imp / imp.sum().clamp_min(1e-8)
    return imp


@torch.no_grad()
def sample_pixels_from_prob(prob_map, n):
    H, W = prob_map.shape
    flat = prob_map.reshape(-1)
    valid = flat > 0

    if valid.sum() == 0:
        return None, None

    idx_valid = torch.where(valid)[0]
    p = flat[idx_valid]
    p = p / p.sum().clamp_min(1e-8)

    replace = idx_valid.numel() < n
    sel_local = torch.multinomial(p, num_samples=n, replacement=replace)
    sel = idx_valid[sel_local]

    y = torch.div(sel, W, rounding_mode="floor")
    x = sel % W
    return y, x


@torch.no_grad()
def uncertainty_aware_inb_ratio(cam_i, render_pkg_i, cam_j, imp_map, n=1024):
    """
    衡量 cam_j 对 cam_i 的“高价值区域”可见程度
    """
    ys, xs = sample_pixels_from_prob(imp_map, n)
    if ys is None:
        return 0.0

    depth_i = render_pkg_i["depth"][0]
    z = depth_i[ys, xs]

    u2, v2, z2, inb = reproject_pixels_i_to_j(cam_i, cam_j, xs, ys, z)
    w = imp_map[ys, xs]

    denom = w.sum().clamp_min(1e-8)
    score = w[inb].sum() / denom
    return float(score.item())


@torch.no_grad()
def moderate_baseline_score(cam_i, cam_j, scene_extent, target=0.15, sigma=0.12):
    """
    中等基线更优：太近没信息，太远投影不稳定
    """
    d = torch.norm(
        cam_i.camera_center.detach().cpu() - cam_j.camera_center.detach().cpu()
    ).item()
    d = d / max(scene_extent, 1e-8)
    return math.exp(-((d - target) ** 2) / (2 * sigma ** 2))


def weighted_xview_reproj_depth_loss(
    cam_i,
    render_pkg_i,
    cam_j,
    render_pkg_j,
    imp_map,
    n_samples=4096,
):
    """
    按 importance map 从难区采样，而不是主要靠高 alpha 稳定区
    """
    depth_i = render_pkg_i["depth"][0]
    depth_j = render_pkg_j["depth"][0]
    alpha_j = get_alpha_map(render_pkg_j).detach().clamp(0.0, 1.0)

    ys, xs = sample_pixels_from_prob(imp_map.detach(), n_samples)
    if ys is None or ys.numel() < 64:
        return None, {"num_valid": 0}

    z_i = depth_i[ys, xs]
    u2, v2, z2, inb = reproject_pixels_i_to_j(cam_i, cam_j, xs, ys, z_i)

    if inb.sum() < 64:
        return None, {"num_valid": int(inb.sum().item())}

    d2 = sample_map2d(depth_j, u2[inb], v2[inb])
    a2 = sample_map2d(alpha_j, u2[inb], v2[inb]).clamp(0.0, 1.0)

    valid = torch.isfinite(d2) & torch.isfinite(z2[inb]) & (d2 > 1e-6)
    if valid.sum() < 64:
        return None, {"num_valid": int(valid.sum().item())}

    rel = (torch.abs(d2[valid] - z2[inb][valid]) / d2[valid].clamp_min(1e-3)).clamp(0.0, 1.0)

    w = imp_map[ys, xs][inb][valid].detach()
    w = w
    w = w / w.sum().clamp_min(1e-8)

    loss = (w * torch.sqrt(rel ** 2 + 1e-6)).sum()

    stats = {
        "num_valid": int(valid.sum().item()),
        "mean_rel": float(rel.mean().item()),
        "a2_med": float(a2[valid].median().item()),
    }
    return loss, stats


@torch.no_grad()
def build_camera_nn_cache(train_cams):
    """
    预先缓存每个相机的近邻排序，避免训练时重复算。
    """
    if len(train_cams) < 2:
        return None, None, None

    centers_cpu = torch.stack([c.camera_center.detach().cpu() for c in train_cams], dim=0)
    uid2idx = {c.uid: i for i, c in enumerate(train_cams)}
    dist_mat = torch.cdist(centers_cpu, centers_cpu, p=2)
    nn_cache = torch.argsort(dist_mat, dim=1)
    return centers_cpu, uid2idx, nn_cache


@torch.no_grad()
def select_best_xview_camera(
    viewpoint_cam,
    render_pkg,
    prior_depth,
    train_cams,
    uid2idx,
    nn_cache,
    scene_extent,
    k_candidates=12,
):
    """
    选最适合当前“不确定区域”的邻视图。
    """
    if nn_cache is None or uid2idx is None or len(train_cams) < 2:
        return None, None, None

    imp_i = build_xview_importance_map(render_pkg, prior_depth)

    i_idx = uid2idx[viewpoint_cam.uid]
    cand_ids = nn_cache[i_idx, 1:k_candidates + 1].tolist()

    best_cam = None
    best_score = -1.0
    best_info = None

    for j_idx in cand_ids:
        cam_j = train_cams[j_idx]

        ov_std = quick_inb_ratio(
            viewpoint_cam,
            render_pkg,
            cam_j,
            n=512,
            alpha_th=1e-3,
        )

        ov_unc = uncertainty_aware_inb_ratio(
            viewpoint_cam,
            render_pkg,
            cam_j,
            imp_i,
            n=1024,
        )

        b_score = moderate_baseline_score(
            viewpoint_cam,
            cam_j,
            scene_extent,
            target=0.15,
            sigma=0.12,
        )

        score = 0.55 * ov_unc + 0.30 * ov_std + 0.15 * b_score

        if score > best_score:
            best_score = score
            best_cam = cam_j
            best_info = {
                "ov_unc": ov_unc,
                "ov_std": ov_std,
                "b_score": b_score,
                "score": score,
            }

    return best_cam, best_score, best_info