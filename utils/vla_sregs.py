"""VLA-SREGS core loop.

Core chain:
    Pseudo View -> Qwen finds overfitting mask -> 2D mask maps to Gaussians
    -> Gaussian-level anti-overfitting update.

Current focus: avoid the idle failure mode
    regions=0, geo_g=0, app_g=0.

Key changes:
  1. Pseudo-view sampling is ranked by image-space projection displacement,
     not only camera-center novelty or in-bound ratio.
  2. The sampler no longer silently falls back to train-like high-inb views.
  3. Qwen receives a stronger diagnostic panel and, if it still returns no
     regions, an explicit risk fallback can keep the loop debuggable.
"""

import json
import os
import random
import shlex
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

from scene.cameras import PseudoCamera


_VALID_CATEGORIES = {"geometry_ambiguity", "appearance_collapse"}


@dataclass
class VLARegion:
    category: str
    box: Tuple[int, int, int, int]
    confidence: float = 1.0


def _getattr(args, name: str, default):
    return getattr(args, name, default)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _normalize_category(category: str) -> Optional[str]:
    c = str(category).strip().lower().replace("-", "_").replace(" ", "_")
    if c in _VALID_CATEGORIES:
        return c
    if c in {"geometry", "structure_error", "structural_error", "depth_error", "shape_error", "coverage_hole"}:
        return "geometry_ambiguity"
    if c in {"appearance", "texture", "texture_underfit", "color_error", "background_collapse", "blur"}:
        return "appearance_collapse"
    return None


def _tensor_to_uint8_image(t: torch.Tensor, normalize: bool = False) -> np.ndarray:
    with torch.no_grad():
        x = t.detach().float().cpu()
        if x.ndim == 3 and x.shape[0] in (1, 3):
            x = x[0] if x.shape[0] == 1 else x.permute(1, 2, 0)
        elif x.ndim != 2:
            x = x.squeeze()
        arr = x.numpy()
        if normalize:
            finite = np.isfinite(arr)
            if finite.any():
                lo, hi = np.percentile(arr[finite], [5, 95])
                arr = (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)
            else:
                arr = np.zeros_like(arr)
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        arr = np.clip(arr, 0.0, 1.0)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=-1)
        return (arr * 255.0 + 0.5).astype(np.uint8)


def _save_tensor_png(t: torch.Tensor, path: str, normalize: bool = False) -> None:
    Image.fromarray(_tensor_to_uint8_image(t, normalize=normalize)).save(path)


def _draw_regions(image_path: str, regions: List[VLARegion], out_path: str) -> None:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for r in regions:
        x1, y1, x2, y2 = r.box
        label = "G" if r.category == "geometry_ambiguity" else "A"
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        draw.text((x1 + 2, max(0, y1 - 14)), f"{label}:{r.confidence:.2f}", fill=(255, 0, 0))
    img.save(out_path)


class VLASREGSController:
    def __init__(self, scene, pipe, background: torch.Tensor, args, render_func):
        self.scene = scene
        self.pipe = pipe
        self.background = background
        self.args = args
        self.render_func = render_func

        self.enabled = bool(_getattr(args, "vla_enable", False))
        self.start_iter = int(_getattr(args, "vla_start_iter", 5000))
        self.interval = int(_getattr(args, "vla_interval", 500))
        self.mask_ttl = int(_getattr(args, "vla_mask_ttl", 500))

        # Pseudo-view sampler. Defaults are intentionally more aggressive than the previous version.
        self.sampler = str(_getattr(args, "vla_sampler", "aggressive") or "aggressive").lower()
        self.num_candidates = int(_getattr(args, "vla_num_candidates", 64))
        self.score_num_points = int(_getattr(args, "vla_score_num_points", 20000))
        self.inb_min = float(_getattr(args, "vla_inb_min", 0.45))
        self.inb_max = float(_getattr(args, "vla_inb_max", 0.85))
        self.strict_inb = bool(_getattr(args, "vla_strict_inb", True))
        self.novelty_min = float(_getattr(args, "vla_novelty_min", 0.08))
        self.novelty_max = float(_getattr(args, "vla_novelty_max", 0.80))
        self.img_disp_min = float(_getattr(args, "vla_img_disp_min", 0.06))
        self.img_disp_max = float(_getattr(args, "vla_img_disp_max", 0.45))
        self.img_disp_samples = int(_getattr(args, "vla_img_disp_samples", 4096))
        self.perturb_trials = int(_getattr(args, "vla_perturb_trials", 96))
        self.perturb_radius_min = float(_getattr(args, "vla_perturb_radius_min", 0.25))
        self.perturb_radius_max = float(_getattr(args, "vla_perturb_radius_max", 0.90))

        # Region and mask control.
        self.min_region_conf = float(_getattr(args, "vla_min_region_conf", 0.20))
        self.max_regions = int(_getattr(args, "vla_max_regions", 6))
        self.max_gaussians_per_type = int(_getattr(args, "vla_max_gaussians_per_type", 40000))
        self.depth_gate = float(_getattr(args, "vla_depth_gate", 0.0))
        self.force_regions = bool(_getattr(args, "vla_force_regions", True))
        self.force_region_min = int(_getattr(args, "vla_force_region_min", 2))

        # Gaussian anti-overfitting response.
        self.geo_opacity_reg = float(_getattr(args, "vla_geo_opacity_reg", 0.005))
        self.geo_scale_reg = float(_getattr(args, "vla_geo_scale_reg", 0.0005))
        self.app_sh_reg = float(_getattr(args, "vla_app_sh_reg", 0.0015))
        self.geo_opacity_growth_scale = float(_getattr(args, "vla_geo_opacity_growth_scale", 0.15))
        self.geo_scaling_grad_scale = float(_getattr(args, "vla_geo_scaling_grad_scale", 0.50))
        self.app_sh_grad_scale = float(_getattr(args, "vla_app_sh_grad_scale", 0.10))
        self.app_dc_grad_scale = float(_getattr(args, "vla_app_dc_grad_scale", 0.50))

        # Qwen worker.
        self.qwen_cmd = str(_getattr(args, "vla_qwen_cmd", "") or "")
        self.qwen_env = str(_getattr(args, "vla_qwen_env", "qwen3vl") or "qwen3vl")
        self.qwen_worker = str(_getattr(args, "vla_qwen_worker", "tools/qwen_vla_mask_worker.py") or "tools/qwen_vla_mask_worker.py")
        self.qwen_model = str(_getattr(args, "vla_qwen_model", "") or "")
        self.qwen_timeout = int(_getattr(args, "vla_qwen_timeout", 240))
        self.qwen_verbose = bool(_getattr(args, "vla_qwen_verbose", False))
        self.fallback_miner = bool(_getattr(args, "vla_fallback_miner", False))
        self.require_qwen = bool(_getattr(args, "vla_require_qwen", False))
        self.qwen_online = bool(_getattr(args, "qwen_online", False))

        self.diag_dir = os.path.join(scene.model_path, "vla_sregs")
        if self.enabled:
            _ensure_dir(self.diag_dir)

        self.active_geo_mask: Optional[torch.Tensor] = None
        self.active_app_mask: Optional[torch.Tensor] = None
        self.active_until: int = -1
        self.last_gaussian_count: int = 0
        self.last_info: Dict[str, float] = {}
        self._pseudo_stack: List = []

    # ------------------------------------------------------------------
    # Public hooks
    # ------------------------------------------------------------------
    def maybe_apply(self, iteration: int, loss: torch.Tensor, gaussians) -> torch.Tensor:
        if not self.enabled:
            return loss
        self._drop_expired(iteration, gaussians)
        if iteration >= self.start_iter and self.interval > 0 and (iteration - self.start_iter) % self.interval == 0:
            self.mine_once(iteration, gaussians)
        return self._add_gaussian_regularizers(loss, gaussians)

    def apply_gradient_modulation(self, iteration: int, gaussians) -> None:
        if not self.enabled:
            return
        self._drop_expired(iteration, gaussians)
        geo, app = self.active_geo_mask, self.active_app_mask
        if geo is None or app is None or geo.shape[0] != gaussians.get_xyz.shape[0]:
            return

        if geo.any():
            if getattr(gaussians, "_opacity", None) is not None and gaussians._opacity.grad is not None:
                g = gaussians._opacity.grad
                neg = g[geo] < 0
                g_geo = g[geo]
                g_geo = torch.where(neg, g_geo * self.geo_opacity_growth_scale, g_geo)
                g[geo] = g_geo
            if getattr(gaussians, "_scaling", None) is not None and gaussians._scaling.grad is not None:
                gaussians._scaling.grad[geo] *= self.geo_scaling_grad_scale

        if app.any():
            if getattr(gaussians, "_features_rest", None) is not None and gaussians._features_rest.grad is not None:
                gaussians._features_rest.grad[app] *= self.app_sh_grad_scale
            if getattr(gaussians, "_features_dc", None) is not None and gaussians._features_dc.grad is not None:
                gaussians._features_dc.grad[app] *= self.app_dc_grad_scale

    def sync_after_topology_change(self, gaussians) -> None:
        if not self.enabled:
            return
        n = int(gaussians.get_xyz.shape[0])
        if self.last_gaussian_count and n != self.last_gaussian_count:
            self.clear_active_masks()
        self.last_gaussian_count = n

    def clear_active_masks(self) -> None:
        self.active_geo_mask = None
        self.active_app_mask = None
        self.active_until = -1

    # ------------------------------------------------------------------
    # Mining
    # ------------------------------------------------------------------
    def mine_once(self, iteration: int, gaussians) -> Dict[str, float]:
        with torch.no_grad():
            pseudo_cam, stats = self._select_pseudo_camera(gaussians)
            if pseudo_cam is None:
                self.last_info = {"vla/selected": 0.0}
                return self.last_info

            render_pkg = self.render_func(pseudo_cam, gaussians, self.pipe, self.background)
            iter_dir = os.path.join(self.diag_dir, f"iter_{iteration:06d}")
            _ensure_dir(iter_dir)

            rgb_path = os.path.join(iter_dir, "pseudo_rgb.png")
            depth_path = os.path.join(iter_dir, "pseudo_depth.png")
            alpha_path = os.path.join(iter_dir, "pseudo_alpha.png")
            panel_path = os.path.join(iter_dir, "diagnostic_panel.png")
            qwen_out_path = os.path.join(iter_dir, "qwen_regions.json")
            overlay_path = os.path.join(iter_dir, "qwen_overlay.png")

            _save_tensor_png(render_pkg["render"], rgb_path, normalize=False)
            _save_tensor_png(render_pkg["depth"][0], depth_path, normalize=True)
            _save_tensor_png(render_pkg["rend_alpha"][0], alpha_path, normalize=False)
            self._save_panel(rgb_path, depth_path, alpha_path, pseudo_cam, panel_path)

            regions = self._invoke_qwen(rgb_path, panel_path, qwen_out_path, iter_dir)
            region_source = "qwen"
            if not regions and (self.force_regions or self.fallback_miner):
                regions = self._risk_regions(render_pkg, min_regions=self.force_region_min)
                region_source = "risk_fallback"
                with open(qwen_out_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "regions": [r.__dict__ for r in regions],
                        "fallback": True,
                        "source": region_source,
                        "reason": "Qwen returned no valid regions; using render-risk top-k boxes for debugging/anti-idle."
                    }, f, indent=2)

            regions = regions[: self.max_regions]
            if regions:
                _draw_regions(rgb_path, regions, overlay_path)

            geo_mask, app_mask = self._regions_to_gaussian_masks(regions, pseudo_cam, gaussians, render_pkg)
            self.active_geo_mask = geo_mask
            self.active_app_mask = app_mask
            self.active_until = iteration + self.mask_ttl
            self.last_gaussian_count = int(gaussians.get_xyz.shape[0])

            info = {
                "vla/selected": 1.0,
                "vla/pseudo_inb": float(stats.get("inb", 0.0)),
                "vla/pseudo_novelty": float(stats.get("novelty", 0.0)),
                "vla/pseudo_disp": float(stats.get("disp", 0.0)),
                "vla/pseudo_score": float(stats.get("score", 0.0)),
                "vla/regions": float(len(regions)),
                "vla/geo_gaussians": float(geo_mask.sum().item()),
                "vla/app_gaussians": float(app_mask.sum().item()),
            }
            self.last_info = info

            summary = dict(info)
            summary.update({
                "active_until": self.active_until,
                "sampler": self.sampler,
                "region_source": region_source,
                "regions": [
                    {"category": r.category, "box": list(r.box), "confidence": r.confidence} for r in regions
                ],
            })
            with open(os.path.join(iter_dir, "vla_summary.json"), "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

            print(
                f"[VLA@{iteration}] inb={info['vla/pseudo_inb']:.3f} "
                f"nov={info['vla/pseudo_novelty']:.3f} disp={info['vla/pseudo_disp']:.3f} "
                f"src={region_source} regions={len(regions)} "
                f"geo_g={int(geo_mask.sum())} app_g={int(app_mask.sum())}"
            )
            return info

    def _drop_expired(self, iteration: int, gaussians) -> None:
        if self.active_geo_mask is None:
            return
        if iteration > self.active_until or self.active_geo_mask.shape[0] != gaussians.get_xyz.shape[0]:
            self.clear_active_masks()

    def _add_gaussian_regularizers(self, loss: torch.Tensor, gaussians) -> torch.Tensor:
        geo, app = self.active_geo_mask, self.active_app_mask
        if geo is None or app is None or geo.shape[0] != gaussians.get_xyz.shape[0]:
            return loss
        add_terms = []
        if geo.any():
            add_terms.append(self.geo_opacity_reg * gaussians.get_opacity[geo].mean())
            if self.geo_scale_reg > 0:
                add_terms.append(self.geo_scale_reg * gaussians.get_scaling[geo].max(dim=1).values.mean())
        if app.any() and getattr(gaussians, "_features_rest", None) is not None:
            add_terms.append(self.app_sh_reg * gaussians._features_rest[app].abs().mean())
        return loss + sum(add_terms) if add_terms else loss

    # ------------------------------------------------------------------
    # Pseudo-view sampling
    # ------------------------------------------------------------------
    def _select_pseudo_camera(self, gaussians):
        candidates = []
        if self.sampler in {"scene", "hybrid", "aggressive"}:
            candidates.extend(self._draw_scene_pseudo_candidates())
        if self.sampler in {"perturb", "hybrid", "aggressive"}:
            candidates.extend(self._draw_perturbed_candidates())
        if not candidates:
            return None, {"inb": 0.0, "novelty": 0.0, "disp": 0.0, "score": 0.0}

        scored = []
        for cam in candidates:
            inb = self._camera_inbound_ratio(cam, gaussians)
            novelty = self._nearest_train_distance(cam)
            disp = self._image_displacement_ratio(cam, gaussians)
            score = self._view_score(inb, novelty, disp)
            scored.append((score, inb, novelty, disp, cam))

        # Prefer views that are still visible but cause real image-space motion.
        valid = [
            x for x in scored
            if x[1] >= self.inb_min and x[3] >= self.img_disp_min and self.novelty_min <= x[2] <= self.novelty_max
        ]
        if self.strict_inb:
            strict_valid = [x for x in valid if x[1] <= self.inb_max]
            if strict_valid:
                valid = strict_valid

        if not valid:
            # Do not fall back to safe train-like views. Pick the most image-displaced visible views.
            visible = [x for x in scored if x[1] >= max(0.25, self.inb_min * 0.7)]
            pool = visible if visible else scored
            pool = sorted(pool, key=lambda x: (x[3], -x[1]), reverse=True)
            valid = pool[:max(1, min(len(pool), len(scored) // 4 or 1))]

        score, inb, novelty, disp, cam = sorted(valid, key=lambda x: x[0], reverse=True)[0]
        return cam, {"inb": float(inb), "novelty": float(novelty), "disp": float(disp), "score": float(score)}

    def _draw_scene_pseudo_candidates(self) -> List[PseudoCamera]:
        cameras = self.scene.getPseudoCameras()
        if cameras is None or len(cameras) == 0 or cameras[0] is None:
            return []
        if not self._pseudo_stack:
            self._pseudo_stack = list(cameras)
            random.shuffle(self._pseudo_stack)
        n_try = max(1, min(self.num_candidates, len(self._pseudo_stack)))
        out = [self._pseudo_stack.pop() for _ in range(n_try)]
        if len(self._pseudo_stack) == 0:
            self._pseudo_stack = list(cameras)
            random.shuffle(self._pseudo_stack)
        return out

    def _draw_perturbed_candidates(self) -> List[PseudoCamera]:
        try:
            cams = self.scene.getTrainCameras()
        except Exception:
            return []
        if not cams:
            return []
        n_try = max(1, self.perturb_trials)
        out = []
        extent = max(float(getattr(self.scene, "cameras_extent", 1.0)), 1e-6)
        for _ in range(n_try):
            base = random.choice(cams)
            if len(cams) >= 2:
                other = random.choice(cams)
                c0 = base.camera_center.detach().float().cpu().numpy()
                c1 = other.camera_center.detach().float().cpu().numpy()
                alpha = np.random.uniform(0.15, 0.85)
                center = (1.0 - alpha) * c0 + alpha * c1
            else:
                center = base.camera_center.detach().float().cpu().numpy()

            direction = np.random.normal(size=3).astype(np.float32)
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            radius = np.random.uniform(self.perturb_radius_min, self.perturb_radius_max) * extent
            center = center + radius * direction
            out.append(self._camera_from_center_and_base(center, base))
        return out

    def _camera_from_center_and_base(self, center_np: np.ndarray, base_cam) -> PseudoCamera:
        R = np.array(base_cam.R, dtype=np.float32)
        T = -(R.T @ center_np.astype(np.float32))
        return PseudoCamera(R=R, T=T, FoVx=base_cam.FoVx, FoVy=base_cam.FoVy,
                            width=base_cam.image_width, height=base_cam.image_height)

    def _nearest_train_camera(self, cam):
        try:
            train = self.scene.getTrainCameras()
        except Exception:
            return None
        if not train:
            return None
        cc = cam.camera_center.detach()
        return min(train, key=lambda t: torch.norm(cc - t.camera_center.detach()).item())

    def _nearest_train_distance(self, cam) -> float:
        nearest = self._nearest_train_camera(cam)
        if nearest is None:
            return 0.0
        extent = max(float(getattr(self.scene, "cameras_extent", 1.0)), 1e-6)
        return float(torch.norm(cam.camera_center.detach() - nearest.camera_center.detach()).item() / extent)

    def _view_score(self, inb: float, novelty: float, disp: float) -> float:
        if inb < self.inb_min:
            inb_score = max(0.0, inb / max(self.inb_min, 1e-6))
        elif self.strict_inb and inb > self.inb_max:
            inb_score = max(0.0, 1.0 - (inb - self.inb_max) / max(1.0 - self.inb_max, 1e-6))
        else:
            inb_score = 1.0

        nov_mid = 0.5 * (self.novelty_min + self.novelty_max)
        nov_half = max(0.5 * (self.novelty_max - self.novelty_min), 1e-6)
        novelty_score = max(0.0, 1.0 - abs(novelty - nov_mid) / nov_half)

        disp_mid = 0.5 * (self.img_disp_min + self.img_disp_max)
        disp_half = max(0.5 * (self.img_disp_max - self.img_disp_min), 1e-6)
        disp_score = max(0.0, 1.0 - abs(disp - disp_mid) / disp_half)
        if disp < self.img_disp_min:
            disp_score *= max(0.0, disp / max(self.img_disp_min, 1e-6))

        return 0.55 * disp_score + 0.25 * novelty_score + 0.20 * inb_score

    def _camera_inbound_ratio(self, camera, gaussians) -> float:
        xyz = gaussians.get_xyz.detach()
        if xyz.shape[0] == 0:
            return 0.0
        if xyz.shape[0] > self.score_num_points:
            ids = torch.randperm(xyz.shape[0], device=xyz.device)[: self.score_num_points]
            xyz = xyz[ids]
        _, _, _, valid = self._project_xyz(camera, xyz)
        return float(valid.float().mean().item())

    def _image_displacement_ratio(self, camera, gaussians) -> float:
        nearest = self._nearest_train_camera(camera)
        if nearest is None:
            return 0.0
        xyz = gaussians.get_xyz.detach()
        if xyz.shape[0] == 0:
            return 0.0
        n = min(int(self.img_disp_samples), int(xyz.shape[0]))
        if xyz.shape[0] > n:
            idx = torch.randperm(xyz.shape[0], device=xyz.device)[:n]
            xyz = xyz[idx]
        px1, py1, _, v1 = self._project_xyz(camera, xyz)
        px0, py0, _, v0 = self._project_xyz(nearest, xyz)
        valid = v1 & v0
        if valid.sum().item() < 32:
            return 0.0
        W, H = float(camera.image_width), float(camera.image_height)
        dx = (px1[valid] - px0[valid]) / max(W, 1.0)
        dy = (py1[valid] - py0[valid]) / max(H, 1.0)
        disp = torch.sqrt(dx * dx + dy * dy)
        return float(torch.quantile(disp.clamp(0, 2), 0.75).item())

    def _project_xyz(self, camera, xyz: torch.Tensor):
        n = xyz.shape[0]
        ones = torch.ones((n, 1), dtype=xyz.dtype, device=xyz.device)
        xyz_h = torch.cat([xyz, ones], dim=1)
        clip = xyz_h @ camera.full_proj_transform
        w = clip[:, 3:4]
        valid_w = torch.isfinite(w[:, 0]) & (w[:, 0].abs() > 1e-8)
        safe_w = torch.where(w.abs() > 1e-8, w, torch.ones_like(w))
        ndc = clip[:, :3] / safe_w
        W, H = int(camera.image_width), int(camera.image_height)
        px = (ndc[:, 0] * 0.5 + 0.5) * float(W - 1)
        py = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * float(H - 1)
        valid = valid_w & torch.isfinite(px) & torch.isfinite(py) & (px >= 0) & (px <= W - 1) & (py >= 0) & (py <= H - 1)
        cam = xyz_h @ camera.world_view_transform
        depth = cam[:, 2]
        return px, py, depth, valid

    # ------------------------------------------------------------------
    # Qwen / fallback mining
    # ------------------------------------------------------------------
    def _invoke_qwen(self, rgb_path: str, panel_path: str, out_path: str, iter_dir: str) -> List[VLARegion]:
        use_qwen = self.qwen_online or bool(self.qwen_cmd) or bool(self.qwen_model)
        if not use_qwen:
            if self.require_qwen:
                raise RuntimeError("VLA-SREGS requires Qwen, but --qwen_online/--vla_qwen_model/--vla_qwen_cmd is not set.")
            return []

        if self.qwen_cmd:
            cmd = shlex.split(self.qwen_cmd.format(image=rgb_path, panel=panel_path, out=out_path, model=self.qwen_model))
        else:
            cmd = ["conda", "run", "-n", self.qwen_env, "python", self.qwen_worker,
                   "--image", rgb_path, "--panel", panel_path, "--out", out_path, "--force_regions"]
            if self.qwen_model:
                cmd += ["--model", self.qwen_model]

        error_path = os.path.join(iter_dir, "qwen_error.log")
        cmd_path = os.path.join(iter_dir, "qwen_cmd.txt")
        with open(cmd_path, "w", encoding="utf-8") as f:
            f.write(" ".join(shlex.quote(x) for x in cmd) + "\n")

        env = os.environ.copy()
        env.setdefault("HF_HUB_DISABLE_XET", "1")
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                  timeout=self.qwen_timeout, env=env)
        except Exception as exc:
            with open(error_path, "w", encoding="utf-8") as f:
                f.write(str(exc))
            if self.require_qwen:
                raise
            print(f"[VLA] Qwen call failed: {exc}. See {error_path}")
            return []

        if proc.returncode != 0:
            msg = (proc.stderr or proc.stdout or "").strip()
            with open(error_path, "w", encoding="utf-8") as f:
                f.write("CMD:\n" + " ".join(shlex.quote(x) for x in cmd) + "\n\n")
                f.write("STDOUT:\n" + (proc.stdout or "") + "\n\n")
                f.write("STDERR:\n" + (proc.stderr or "") + "\n")
            if self.require_qwen:
                raise RuntimeError(f"Qwen worker failed with code {proc.returncode}. See {error_path}\n{msg}")
            show = msg if self.qwen_verbose else msg[:1000]
            print(f"[VLA] Qwen worker failed, skip this mining step. See {error_path}\n{show}")
            return []

        return self._load_regions(out_path)

    def _load_regions(self, path: str) -> List[VLARegion]:
        if not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return []
        raw_regions = data.get("regions", data if isinstance(data, list) else [])
        regions: List[VLARegion] = []
        for item in raw_regions:
            if not isinstance(item, dict):
                continue
            cat = _normalize_category(item.get("category", item.get("type", "")))
            conf = _safe_float(item.get("confidence", item.get("score", 1.0)), 1.0)
            box = item.get("box", item.get("bbox", None))
            if cat is None or conf < self.min_region_conf or box is None or len(box) != 4:
                continue
            try:
                x1, y1, x2, y2 = [int(round(float(v))) for v in box]
            except Exception:
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            regions.append(VLARegion(category=cat, box=(x1, y1, x2, y2), confidence=conf))
        return regions

    def _risk_regions(self, render_pkg, min_regions: int = 2) -> List[VLARegion]:
        alpha = render_pkg["rend_alpha"][0].detach()
        depth = render_pkg["depth"][0].detach()
        rgb = render_pkg["render"].detach()
        H, W = alpha.shape

        # Simple differentiable-free risk cues on pseudo render.
        low_alpha = (1.0 - alpha.clamp(0, 1))
        dzx = torch.zeros_like(depth); dzy = torch.zeros_like(depth)
        dzx[:, 1:] = (depth[:, 1:] - depth[:, :-1]).abs()
        dzy[1:, :] = (depth[1:, :] - depth[:-1, :]).abs()
        depth_edge = (dzx + dzy)
        finite = torch.isfinite(depth_edge)
        if finite.any():
            q = torch.quantile(depth_edge[finite], 0.95).clamp_min(1e-6)
            depth_edge = (depth_edge / q).clamp(0, 1)
        else:
            depth_edge = torch.zeros_like(depth_edge)

        gray = rgb.mean(dim=0)
        gx = torch.zeros_like(gray); gy = torch.zeros_like(gray)
        gx[:, 1:] = (gray[:, 1:] - gray[:, :-1]).abs()
        gy[1:, :] = (gray[1:, :] - gray[:-1, :]).abs()
        texture_edge = (gx + gy).clamp(0, 1)

        risk_geo = 0.65 * low_alpha + 0.35 * depth_edge
        risk_app = (1.0 - texture_edge).clamp(0, 1) * alpha.clamp(0, 1)

        gh, gw = 8, 8
        cell_h, cell_w = max(1, H // gh), max(1, W // gw)
        geo_scores, app_scores = [], []
        for iy in range(gh):
            for ix in range(gw):
                y1, y2 = iy * cell_h, H if iy == gh - 1 else (iy + 1) * cell_h
                x1, x2 = ix * cell_w, W if ix == gw - 1 else (ix + 1) * cell_w
                box = (x1, y1, x2, y2)
                geo_scores.append((float(risk_geo[y1:y2, x1:x2].mean().item()), box))
                app_scores.append((float(risk_app[y1:y2, x1:x2].mean().item()), box))
        geo_scores.sort(key=lambda x: x[0], reverse=True)
        app_scores.sort(key=lambda x: x[0], reverse=True)

        regions: List[VLARegion] = []
        for score, box in geo_scores[:max(1, min_regions)]:
            regions.append(VLARegion("geometry_ambiguity", box, max(0.30, min(0.75, score))))
        for score, box in app_scores[:max(0, min_regions - len(regions))]:
            regions.append(VLARegion("appearance_collapse", box, max(0.30, min(0.65, score))))
        return regions

    # ------------------------------------------------------------------
    # 2D mask -> Gaussian mask
    # ------------------------------------------------------------------
    def _regions_to_gaussian_masks(self, regions: List[VLARegion], camera, gaussians, render_pkg):
        n = int(gaussians.get_xyz.shape[0])
        device = gaussians.get_xyz.device
        geo = torch.zeros((n,), dtype=torch.bool, device=device)
        app = torch.zeros((n,), dtype=torch.bool, device=device)
        if not regions or n == 0:
            return geo, app

        px, py, z, valid = self._project_xyz(camera, gaussians.get_xyz.detach())
        if "radii" in render_pkg and render_pkg["radii"].shape[0] == n:
            valid = valid & (render_pkg["radii"].detach() > 0)

        depth_map = render_pkg["depth"][0].detach() if self.depth_gate > 0 and "depth" in render_pkg else None
        W, H = int(camera.image_width), int(camera.image_height)
        for r in regions:
            x1, y1, x2, y2 = r.box
            x1 = max(0, min(W - 1, x1)); x2 = max(0, min(W - 1, x2))
            y1 = max(0, min(H - 1, y1)); y2 = max(0, min(H - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            m = valid & (px >= x1) & (px <= x2) & (py >= y1) & (py <= y2)
            if depth_map is not None and m.any():
                xi = px[m].round().long().clamp(0, W - 1)
                yi = py[m].round().long().clamp(0, H - 1)
                d = depth_map[yi, xi]
                zz = z[m]
                denom = d.abs().clamp_min(1e-6)
                keep = torch.isfinite(d) & (d > 0) & ((zz - d).abs() / denom < self.depth_gate)
                ids = torch.nonzero(m, as_tuple=False).squeeze(-1)
                mm = torch.zeros_like(m)
                mm[ids[keep]] = True
                m = mm
            if r.category == "geometry_ambiguity":
                geo |= m
            elif r.category == "appearance_collapse":
                app |= m

        geo = self._cap_mask_by_opacity(geo, gaussians, self.max_gaussians_per_type)
        app = self._cap_mask_by_opacity(app, gaussians, self.max_gaussians_per_type)
        return geo, app

    def _cap_mask_by_opacity(self, mask: torch.Tensor, gaussians, limit: int) -> torch.Tensor:
        if limit <= 0 or mask.sum().item() <= limit:
            return mask
        idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        opacity = gaussians.get_opacity.detach()[idx, 0]
        keep_local = torch.topk(opacity, k=limit, largest=True).indices
        new_mask = torch.zeros_like(mask)
        new_mask[idx[keep_local]] = True
        return new_mask

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def _nearest_train_image(self, pseudo_cam) -> Optional[torch.Tensor]:
        nearest = self._nearest_train_camera(pseudo_cam)
        if nearest is None or not hasattr(nearest, "original_image"):
            return None
        return nearest.original_image.detach()

    def _save_panel(self, rgb_path: str, depth_path: str, alpha_path: str, pseudo_cam, panel_path: str) -> None:
        rgb = Image.open(rgb_path).convert("RGB")
        depth = Image.open(depth_path).convert("RGB")
        alpha = Image.open(alpha_path).convert("RGB")
        nearest = self._nearest_train_image(pseudo_cam)
        ref = Image.new("RGB", rgb.size, (0, 0, 0)) if nearest is None else Image.fromarray(_tensor_to_uint8_image(nearest, normalize=False)).resize(rgb.size)

        W, H = rgb.size
        max_side = int(_getattr(self.args, "vla_panel_side", 512))
        scale = min(1.0, max_side / max(W, H))
        if scale < 1.0:
            size = (max(1, int(W * scale)), max(1, int(H * scale)))
            rgb, depth, alpha, ref = rgb.resize(size), depth.resize(size), alpha.resize(size), ref.resize(size)
            W, H = size
        panel = Image.new("RGB", (W * 2, H * 2), (0, 0, 0))
        panel.paste(rgb, (0, 0)); panel.paste(depth, (W, 0)); panel.paste(alpha, (0, H)); panel.paste(ref, (W, H))
        draw = ImageDraw.Draw(panel)
        draw.text((8, 8), "A pseudo render", fill=(255, 255, 255))
        draw.text((W + 8, 8), "B pseudo depth", fill=(255, 255, 255))
        draw.text((8, H + 8), "C pseudo alpha", fill=(255, 255, 255))
        draw.text((W + 8, H + 8), "D nearest train image", fill=(255, 255, 255))
        panel.save(panel_path)
