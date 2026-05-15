import math
import torch

from gaussian_renderer import render
from utils.consist_view import quick_inb_ratio


# ============================================================
# Pseudo camera object
# ============================================================

class PseudoCamera:
    """
    Minimal camera object compatible with gaussian_renderer.render().

    The renderer uses:
        image_height
        image_width
        FoVx
        FoVy
        znear / zfar
        world_view_transform
        full_proj_transform
        camera_center

    Some helper functions infer H/W from original_image, so we also attach
    a dummy original_image.
    """

    def __init__(
        self,
        uid,
        image_height,
        image_width,
        FoVx,
        FoVy,
        znear,
        zfar,
        world_view_transform,
        full_proj_transform,
        camera_center,
        device="cuda",
        anchor_uid=None,
    ):
        self.uid = int(uid)
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.FoVx = float(FoVx)
        self.FoVy = float(FoVy)
        self.znear = float(znear)
        self.zfar = float(zfar)

        self.world_view_transform = world_view_transform.to(device).float()
        self.full_proj_transform = full_proj_transform.to(device).float()
        self.camera_center = camera_center.to(device).float()

        self.anchor_uid = anchor_uid

        # Used only for shape inference in some utility functions.
        self.original_image = torch.zeros(
            3,
            self.image_height,
            self.image_width,
            device=device,
            dtype=torch.float32,
        )

        # Optional compatibility fields.
        self.image_name = f"pseudo_{self.uid}"
        self.depth_image = torch.zeros(
            self.image_height,
            self.image_width,
            device=device,
            dtype=torch.float32,
        )


# ============================================================
# Camera matrix helpers
# ============================================================

def _cam_hw(cam):
    if hasattr(cam, "image_height") and hasattr(cam, "image_width"):
        return int(cam.image_height), int(cam.image_width)
    if hasattr(cam, "original_image") and cam.original_image is not None:
        return int(cam.original_image.shape[-2]), int(cam.original_image.shape[-1])
    raise RuntimeError("Cannot infer camera H/W.")


def _get_world_view(cam):
    return cam.world_view_transform.float()


def _get_projection_from_anchor(anchor_cam):
    """
    In GraphDECO convention:
        full_proj_transform = world_view_transform @ projection_matrix

    So:
        projection_matrix = inverse(world_view_transform) @ full_proj_transform
    """
    WV = _get_world_view(anchor_cam)
    FP = anchor_cam.full_proj_transform.float()
    return torch.linalg.inv(WV) @ FP


def _stored_world_view_to_conventional_c2w(cam):
    """
    Camera stores world_view_transform = W2C.T.
    Convert it back to conventional column-vector C2W.
    """
    WV_stored = cam.world_view_transform.float()
    W2C = WV_stored.transpose(0, 1).contiguous()
    C2W = torch.linalg.inv(W2C)
    return C2W


def _conventional_c2w_to_stored_world_view(C2W):
    W2C = torch.linalg.inv(C2W)
    return W2C.transpose(0, 1).contiguous()


def _safe_norm(x, eps=1e-8):
    return x / (torch.linalg.norm(x) + eps)


def _camera_axes_world(cam):
    """
    Returns camera basis in world coordinates.

    Convention:
        x axis: right
        y axis: down
        z axis: forward
    """
    C2W = _stored_world_view_to_conventional_c2w(cam)
    R = C2W[:3, :3]

    right = _safe_norm(R[:, 0])
    down = _safe_norm(R[:, 1])
    forward = _safe_norm(R[:, 2])
    center = C2W[:3, 3]

    return right, down, forward, center


def _make_K(cam, device):
    H, W = _cam_hw(cam)
    fx = 0.5 * W / math.tan(float(cam.FoVx) * 0.5)
    fy = 0.5 * H / math.tan(float(cam.FoVy) * 0.5)
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


def _backproject_pixel_to_world(cam, u, v, z):
    """
    u, v, z are scalar tensors.
    """
    device = cam.world_view_transform.device
    K = _make_K(cam, device)
    K_inv = torch.linalg.inv(K)

    pix = torch.stack(
        [
            torch.as_tensor(u, device=device, dtype=torch.float32),
            torch.as_tensor(v, device=device, dtype=torch.float32),
            torch.ones((), device=device, dtype=torch.float32),
        ],
        dim=0,
    )

    Xc = (K_inv @ pix) * z

    C2W = _stored_world_view_to_conventional_c2w(cam).to(device)
    Xw = C2W[:3, :3] @ Xc + C2W[:3, 3]
    return Xw


def _estimate_anchor_target(anchor_cam, pkg_anchor):
    """
    Estimate a stable target point from valid central rendered area.
    This keeps pseudo views looking at current Gaussian support.
    """
    depth = pkg_anchor["depth"][0]
    alpha = pkg_anchor.get("rend_alpha", None)

    if alpha is not None:
        alpha = alpha[0] if alpha.dim() == 3 else alpha
        valid = (alpha > 0.45) & torch.isfinite(depth) & (depth > 1e-6)
    else:
        valid = torch.isfinite(depth) & (depth > 1e-6)

    H, W = depth.shape

    # Avoid image borders.
    pad_h = max(8, int(0.08 * H))
    pad_w = max(8, int(0.08 * W))
    valid[:pad_h, :] = False
    valid[-pad_h:, :] = False
    valid[:, :pad_w] = False
    valid[:, -pad_w:] = False

    idx = torch.nonzero(valid, as_tuple=False)

    if idx.shape[0] < 64:
        return None, None

    # Prefer the central valid region.
    cy = H * 0.5
    cx = W * 0.5
    dist2 = (idx[:, 0].float() - cy) ** 2 + (idx[:, 1].float() - cx) ** 2
    pick = idx[torch.argmin(dist2)]

    v = pick[0]
    u = pick[1]
    z = depth[v, u]

    Xw = _backproject_pixel_to_world(anchor_cam, u.float(), v.float(), z)

    d_med = torch.median(depth[valid]).detach()

    return Xw.detach(), d_med.detach()


def _look_at_world_view(anchor_cam, center, target):
    """
    Build a pseudo camera view matrix that looks at target while preserving
    anchor image orientation as much as possible.
    """
    device = anchor_cam.world_view_transform.device

    anchor_right, anchor_down, _, _ = _camera_axes_world(anchor_cam)
    anchor_right = anchor_right.to(device)
    anchor_down = anchor_down.to(device)

    forward = _safe_norm(target - center)

    # Keep image orientation close to anchor.
    right = torch.cross(anchor_down, forward, dim=0)

    if torch.linalg.norm(right) < 1e-6:
        right = anchor_right

    right = _safe_norm(right)
    down = _safe_norm(torch.cross(forward, right, dim=0))

    C2W = torch.eye(4, device=device, dtype=torch.float32)
    C2W[:3, 0] = right
    C2W[:3, 1] = down
    C2W[:3, 2] = forward
    C2W[:3, 3] = center

    return _conventional_c2w_to_stored_world_view(C2W)


def make_pseudo_camera(anchor_cam, center, target, uid):
    device = anchor_cam.world_view_transform.device

    H, W = _cam_hw(anchor_cam)

    znear = float(getattr(anchor_cam, "znear", 0.01))
    zfar = float(getattr(anchor_cam, "zfar", 100.0))

    WV = _look_at_world_view(anchor_cam, center, target)
    P = _get_projection_from_anchor(anchor_cam).to(device)
    FP = WV @ P

    camera_center = torch.linalg.inv(WV)[3, :3].detach()

    return PseudoCamera(
        uid=uid,
        image_height=H,
        image_width=W,
        FoVx=anchor_cam.FoVx,
        FoVy=anchor_cam.FoVy,
        znear=znear,
        zfar=zfar,
        world_view_transform=WV,
        full_proj_transform=FP,
        camera_center=camera_center,
        device=device,
        anchor_uid=getattr(anchor_cam, "uid", None),
    )


# ============================================================
# Serialization
# ============================================================

def pseudo_camera_to_json(cam):
    return {
        "uid": int(cam.uid),
        "image_height": int(cam.image_height),
        "image_width": int(cam.image_width),
        "FoVx": float(cam.FoVx),
        "FoVy": float(cam.FoVy),
        "znear": float(cam.znear),
        "zfar": float(cam.zfar),
        "world_view_transform": cam.world_view_transform.detach().cpu().tolist(),
        "full_proj_transform": cam.full_proj_transform.detach().cpu().tolist(),
        "camera_center": cam.camera_center.detach().cpu().tolist(),
        "anchor_uid": cam.anchor_uid,
    }


def pseudo_camera_from_json(data, device="cuda"):
    WV = torch.tensor(
        data["world_view_transform"],
        device=device,
        dtype=torch.float32,
    )
    FP = torch.tensor(
        data["full_proj_transform"],
        device=device,
        dtype=torch.float32,
    )
    C = torch.tensor(
        data["camera_center"],
        device=device,
        dtype=torch.float32,
    )

    return PseudoCamera(
        uid=int(data["uid"]),
        image_height=int(data["image_height"]),
        image_width=int(data["image_width"]),
        FoVx=float(data["FoVx"]),
        FoVy=float(data["FoVy"]),
        znear=float(data.get("znear", 0.01)),
        zfar=float(data.get("zfar", 100.0)),
        world_view_transform=WV,
        full_proj_transform=FP,
        camera_center=C,
        device=device,
        anchor_uid=data.get("anchor_uid", None),
    )


# ============================================================
# Render-and-reject pseudo sampling
# ============================================================

@torch.no_grad()
def _best_reference_train_cam(pseudo_cam, pseudo_pkg, train_cams):
    best_cam = None
    best_inb = -1.0

    for cam in train_cams:
        r = quick_inb_ratio(
            pseudo_cam,
            pseudo_pkg,
            cam,
            n=1024,
            alpha_th=0.2,
        )

        if torch.is_tensor(r):
            r = float(r.detach().cpu().item())
        else:
            r = float(r)

        if r > best_inb:
            best_inb = r
            best_cam = cam

    return best_cam, best_inb


@torch.no_grad()
def sample_valid_pseudo_cameras(
    train_cams,
    gaussians,
    pipe,
    background,
    max_total=1,
):
    """
    LLFF-friendly pseudo pose sampler.

    Strategy:
        anchor-view local perturbation
        + look-at anchor support target
        + render-and-reject
        + global top-k selection

    Difference from the old version:
        old: keep max_per_anchor pseudo cameras for each train camera
        new: collect all valid candidates from all train cameras, then keep only global top max_total

    Returns list of dict:
        {
            "camera": pseudo_cam,
            "pkg": render_pkg,
            "anchor_uid": int,
            "ref_cam": train_cam,
            "ref_uid": int,
            "valid_ratio": float,
            "inb": float,
            "score": float,
        }
    """
    all_candidates = []
    pseudo_uid = -100000

    if train_cams is None or len(train_cams) == 0:
        return []

    for anchor in train_cams:
        device = anchor.world_view_transform.device

        # Render anchor view to estimate a stable target point.
        pkg_anchor = render(anchor, gaussians, pipe, background)

        target, d_med = _estimate_anchor_target(anchor, pkg_anchor)
        if target is None or d_med is None:
            continue

        right, down, _, center_anchor = _camera_axes_world(anchor)
        right = right.to(device)
        down = down.to(device)
        center_anchor = center_anchor.to(device)

        # Small parallax perturbations.
        # These are deliberately conservative for LLFF forward-facing scenes.
        angle_degs = [2.0, 3.5]

        base_dirs = [
            right,
            -right,
            down,
            -down,
            _safe_norm(right + down),
            _safe_norm(right - down),
            _safe_norm(-right + down),
            _safe_norm(-right - down),
        ]

        for angle_deg in angle_degs:
            delta = d_med * math.tan(math.radians(angle_deg))

            for direction in base_dirs:
                pseudo_uid -= 1

                C_new = center_anchor + delta * direction

                pseudo_cam = make_pseudo_camera(
                    anchor_cam=anchor,
                    center=C_new,
                    target=target,
                    uid=pseudo_uid,
                )

                # Render pseudo view.
                pkg_p = render(pseudo_cam, gaussians, pipe, background)

                depth = pkg_p["depth"][0]
                alpha = pkg_p.get("rend_alpha", None)

                if alpha is not None:
                    alpha = alpha[0] if alpha.dim() == 3 else alpha
                    valid = (alpha > 0.2) & torch.isfinite(depth) & (depth > 1e-6)
                else:
                    valid = torch.isfinite(depth) & (depth > 1e-6)

                valid_ratio = float(valid.float().mean().detach().cpu().item())

                # Reject almost-empty / mostly-background pseudo views.
                # Do not make this too strict, because patch-level validity will be checked later.
                if valid_ratio < 0.08:
                    continue

                # Choose the best real train view as proxy reference view.
                ref_cam, inb = _best_reference_train_cam(
                    pseudo_cam,
                    pkg_p,
                    train_cams,
                )

                if ref_cam is None or inb < 0.15:
                    continue

                # Ranking score:
                # valid_ratio: whether pseudo render has enough Gaussian support
                # inb: whether pseudo points can be reprojected into a train view
                score = 0.60 * valid_ratio + 0.40 * float(inb)

                all_candidates.append(
                    {
                        "camera": pseudo_cam,
                        "pkg": pkg_p,
                        "anchor_uid": int(anchor.uid),
                        "ref_cam": ref_cam,
                        "ref_uid": int(ref_cam.uid),
                        "valid_ratio": float(valid_ratio),
                        "inb": float(inb),
                        "score": float(score),
                    }
                )

    if len(all_candidates) == 0:
        return []

    all_candidates = sorted(
        all_candidates,
        key=lambda x: x["score"],
        reverse=True,
    )

    return all_candidates[:max_total]