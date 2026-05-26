"""VLA-SREGS core loop.

Core chain:
    Pseudo View -> Qwen finds overfitting mask -> 2D mask maps to Gaussians
    -> Gaussian-level anti-overfitting update.

This version fixes three practical failure modes observed in early runs:
  1) Qwen saw only pseudo RGB; now the default worker receives the diagnostic panel.
  2) Qwen errors were truncated; now the full command/stderr is saved per mining step.
  3) FSGS-style pseudo pools may be too close to train views; now the sampler can
     create on-demand perturbed pseudo cameras and rank them by novelty + visibility.
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
    c = str(category).strip().lower()
    c = c.replace("-", "_").replace(" ", "_")
    if c in _VALID_CATEGORIES:
        return c
    if c in {"geometry", "structure_error", "structural_error", "depth_error", "shape_error"}:
        return "geometry_ambiguity"
    if c in {"appearance", "texture", "texture_underfit", "color_error", "background_collapse", "blur"}:
        return "appearance_collapse"
    return None


def _tensor_to_uint8_image(t: torch.Tensor, normalize: bool = False) -> np.ndarray:
    """Convert CxHxW or HxW tensor to HxWx3 uint8."""
    with torch.no_grad():
        x = t.detach().float().cpu()
        if x.ndim == 3 and x.shape[0] in (1, 3):
            if x.shape[0] == 1:
                x = x[0]
            else:
                x = x.permute(1, 2, 0)
        elif x.ndim == 2:
            pass
        else:
            x = x.squeeze()
        arr = x.numpy()
        if normalize:
            finite = np.isfinite(arr)
            if finite.any():
                lo, hi = np.percentile(arr[finite], [5, 95])
                if hi > lo:
                    arr = (arr - lo) / (hi - lo)
                else:
                    arr = np.zeros_like(arr)
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
    """Core VLA-SREGS controller."""

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
        self.num_candidates = int(_getattr(args, "vla_num_candidates", 8))
        self.score_num_points = int(_getattr(args, "vla_score_num_points", 20000))
        self.inb_min = float(_getattr(args, "vla_inb_min", 0.55))
        self.inb_max = float(_getattr(args, "vla_inb_max", 0.82))
        self.strict_inb = bool(_getattr(args, "vla_strict_inb", False))
        self.sampler = str(_getattr(args, "vla_sampler", "hybrid") or "hybrid").lower()
        self.novelty_min = float(_getattr(args, "vla_novelty_min", 0.08))
        self.novelty_max = float(_getattr(args, "vla_novelty_max", 0.45))
        self.perturb_trials = int(_getattr(args, "vla_perturb_trials", 32))
        self.perturb_radius_min = float(_getattr(args, "vla_perturb_radius_min", 0.08))
        self.perturb_radius_max = float(_getattr(args, "vla_perturb_radius_max", 0.35))
        self.min_region_conf = float(_getattr(args, "vla_min_region_conf", 0.35))
        self.max_regions = int(_getattr(args, "vla_max_regions", 6))
        self.max_gaussians_per_type = int(_getattr(args, "vla_max_gaussians_per_type", 40000))
        self.depth_gate = float(_getattr(args, "vla_depth_gate", 0.0))

        self.geo_opacity_reg = float(_getattr(args, "vla_geo_opacity_reg", 0.005))
        self.geo_scale_reg = float(_getattr(args, "vla_geo_scale_reg", 0.0005))
        self.app_sh_reg = float(_getattr(args, "vla_app_sh_reg", 0.0015))
        self.geo_opacity_growth_scale = float(_getattr(args, "vla_geo_opacity_growth_scale", 0.15))
        self.geo_scaling_grad_scale = float(_getattr(args, "vla_geo_scaling_grad_scale", 0.50))
        self.app_sh_grad_scale = float(_getattr(args, "vla_app_sh_grad_scale", 0.10))
        self.app_dc_grad_scale = float(_getattr(args, "vla_app_dc_grad_scale", 0.50))

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
        self._last_selected_stats: Dict[str, float] = {"inb": 0.0, "novelty": 0.0, "score": 0.0}

    def maybe_apply(self, iteration: int, loss: torch.Tensor, gaussians) -> torch.Tensor:
        if not self.enabled:
            return loss
        self._drop_expired(iteration, gaussians)
        if iteration >= self.start_iter and self.interval > 0 and (iteration - self.start_iter) % self.interval == 0:
            self.mine_once(iteration, gaussians)
        return self._add_gaussian_regularizers(iteration, loss, gaussians)

    def apply_gradient_modulation(self, iteration: int, gaussians) -> None:
        if not self.enabled:
            return
        self._drop_expired(iteration, gaussians)
        geo = self.active_geo_mask
        app = self.active_app_mask
        if geo is None or app is None:
            return
        if geo.shape[0] != gaussians.get_xyz.shape[0]:
            self.clear_active_masks()
            return

        if geo.any():
            if getattr(gaussians, "_opacity", None) is not None and gaussians._opacity.grad is not None:
                g = gaussians._opacity.grad
                neg = g[geo] < 0  # negative grad increases opacity after optimizer step
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
            if not regions and self.fallback_miner:
                regions = self._fallback_regions(render_pkg, pseudo_cam)
                with open(qwen_out_path, "w", encoding="utf-8") as f:
                    json.dump({"regions": [r.__dict__ for r in regions], "fallback": True}, f, indent=2)

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
                "vla/pseudo_score": float(stats.get("score", 0.0)),
                "vla/regions": float(len(regions)),
                "vla/geo_gaussians": float(geo_mask.sum().item()),
                "vla/app_gaussians": float(app_mask.sum().item()),
            }
            self.last_info = info

            summary = dict(info)
            summary["active_until"] = self.active_until
            summary["sampler"] = self.sampler
            summary["regions"] = [
                {"category": r.category, "box": list(r.box), "confidence": r.confidence} for r in regions
            ]
            with open(os.path.join(iter_dir, "vla_summary.json"), "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

            print(
                f"[VLA@{iteration}] inb={info['vla/pseudo_inb']:.3f} "
                f"nov={info['vla/pseudo_novelty']:.3f} regions={len(regions)} "
                f"geo_g={int(geo_mask.sum())} app_g={int(app_mask.sum())}"
            )
            return info

    def _drop_expired(self, iteration: int, gaussians) -> None:
        if self.active_geo_mask is None:
            return
        if iteration > self.active_until:
            self.clear_active_masks()
            return
        if self.active_geo_mask.shape[0] != gaussians.get_xyz.shape[0]:
            self.clear_active_masks()

    def _add_gaussian_regularizers(self, iteration: int, loss: torch.Tensor, gaussians) -> torch.Tensor:
        geo = self.active_geo_mask
        app = self.active_app_mask
        if geo is None or app is None:
            return loss
        if geo.shape[0] != gaussians.get_xyz.shape[0]:
            self.clear_active_masks()
            return loss

        add_terms = []
        if geo.any():
            opacity = gaussians.get_opacity[geo]
            add_terms.append(self.geo_opacity_reg * opacity.mean())
            if self.geo_scale_reg > 0:
                scale = gaussians.get_scaling[geo]
                add_terms.append(self.geo_scale_reg * scale.max(dim=1).values.mean())
        if app.any() and getattr(gaussians, "_features_rest", None) is not None:
            add_terms.append(self.app_sh_reg * gaussians._features_rest[app].abs().mean())

        if add_terms:
            return loss + sum(add_terms)
        return loss

    # ------------------------------------------------------------------
    # Pseudo-view sampling
    # ------------------------------------------------------------------
    def _select_pseudo_camera(self, gaussians):
        candidates = []
        if self.sampler in {"scene", "hybrid"}:
            candidates.extend(self._draw_scene_pseudo_candidates())
        if self.sampler in {"perturb", "hybrid"}:
            candidates.extend(self._draw_perturbed_candidates())

        if not candidates:
            return None, {"inb": 0.0, "novelty": 0.0, "score": 0.0}

        scored = []
        for cam in candidates:
            inb = self._camera_inbound_ratio(cam, gaussians)
            novelty = self._nearest_train_distance(cam)
            score = self._view_score(inb, novelty)
            scored.append((score, inb, novelty, cam))

        def is_valid(item):
            _, inb, novelty, _ = item
            if inb < self.inb_min:
                return False
            if self.strict_inb and inb > self.inb_max:
                return False
            return self.novelty_min <= novelty <= self.novelty_max

        valid = [x for x in scored if is_valid(x)]
        if not valid:
            # Relax novelty first, because LLFF scenes may keep most Gaussians in frame even for useful views.
            valid = [x for x in scored if x[1] >= self.inb_min]
        if not valid:
            valid = scored

        score, inb, novelty, cam = sorted(valid, key=lambda x: x[0], reverse=True)[0]
        return cam, {"inb": float(inb), "novelty": float(novelty), "score": float(score)}

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
        centers = [c.camera_center.detach().float().cpu().numpy() for c in cams]
        for _ in range(n_try):
            base = random.choice(cams)
            if len(cams) >= 2:
                other = random.choice(cams)
                c0 = base.camera_center.detach().float().cpu().numpy()
                c1 = other.camera_center.detach().float().cpu().numpy()
                alpha = np.random.uniform(0.25, 0.75)
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
        # getWorld2View2 constructs world-to-camera as [R.T, T]. For a desired camera center C,
        # T = -R.T @ C.
        T = -(R.T @ center_np.astype(np.float32))
        return PseudoCamera(
            R=R,
            T=T,
            FoVx=base_cam.FoVx,
            FoVy=base_cam.FoVy,
            width=base_cam.image_width,
            height=base_cam.image_height,
        )

    def _nearest_train_distance(self, cam) -> float:
        try:
            train = self.scene.getTrainCameras()
        except Exception:
            return 0.0
        if not train:
            return 0.0
        cc = cam.camera_center.detach()
        d = [torch.norm(cc - t.camera_center.detach()).item() for t in train]
        extent = max(float(getattr(self.scene, "cameras_extent", 1.0)), 1e-6)
        return float(min(d) / extent)

    def _view_score(self, inb: float, novelty: float) -> float:
        # Visibility gate: below inb_min is unsafe. Above inb_max is not necessarily bad in LLFF,
        # so it is penalized only when strict_inb is enabled.
        if inb < self.inb_min:
            inb_score = max(0.0, inb / max(self.inb_min, 1e-6))
        elif self.strict_inb and inb > self.inb_max:
            inb_score = max(0.0, 1.0 - (inb - self.inb_max) / max(1.0 - self.inb_max, 1e-6))
        else:
            inb_score = 1.0

        mid = 0.5 * (self.novelty_min + self.novelty_max)
        half = max(0.5 * (self.novelty_max - self.novelty_min), 1e-6)
        novelty_score = max(0.0, 1.0 - abs(novelty - mid) / half)
        return 0.65 * novelty_score + 0.35 * inb_score

    def _camera_inbound_ratio(self, camera, gaussians) -> float:
        xyz = gaussians.get_xyz.detach()
        if xyz.shape[0] == 0:
            return 0.0
        if xyz.shape[0] > self.score_num_points:
            ids = torch.randperm(xyz.shape[0], device=xyz.device)[: self.score_num_points]
            xyz = xyz[ids]
        _, _, _, valid = self._project_xyz(camera, xyz)
        return float(valid.float().mean().item())

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
        valid = (
            valid_w
            & torch.isfinite(px)
            & torch.isfinite(py)
            & (px >= 0)
            & (px <= W - 1)
            & (py >= 0)
            & (py <= H - 1)
        )
        cam = xyz_h @ camera.world_view_transform
        depth = cam[:, 2]
        return px, py, depth, valid

    # ------------------------------------------------------------------
    # Qwen / VLM mining
    # ------------------------------------------------------------------
    def _invoke_qwen(self, rgb_path: str, panel_path: str, out_path: str, iter_dir: str) -> List[VLARegion]:
        use_qwen = self.qwen_online or bool(self.qwen_cmd) or bool(self.qwen_model)
        if not use_qwen:
            if self.require_qwen:
                raise RuntimeError("VLA-SREGS requires Qwen, but --qwen_online/--vla_qwen_model/--vla_qwen_cmd is not set.")
            return []

        if self.qwen_cmd:
            cmd = shlex.split(
                self.qwen_cmd.format(image=rgb_path, panel=panel_path, out=out_path, model=self.qwen_model)
            )
        else:
            cmd = [
                "conda", "run", "-n", self.qwen_env,
                "python", self.qwen_worker,
                "--image", rgb_path,
                "--panel", panel_path,
                "--out", out_path,
            ]
            if self.qwen_model:
                cmd += ["--model", self.qwen_model]

        error_path = os.path.join(iter_dir, "qwen_error.log")
        cmd_path = os.path.join(iter_dir, "qwen_cmd.txt")
        with open(cmd_path, "w", encoding="utf-8") as f:
            f.write(" ".join(shlex.quote(x) for x in cmd) + "\n")

        env = os.environ.copy()
        env.setdefault("HF_HUB_DISABLE_XET", "1")
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.qwen_timeout,
                env=env,
            )
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

    def _fallback_regions(self, render_pkg, camera) -> List[VLARegion]:
        """Debug-only miner. It is off by default because it is not a VLM signal."""
        alpha = render_pkg["rend_alpha"][0].detach()
        H, W = alpha.shape
        low = alpha < 0.15
        if low.float().mean() < 0.01:
            return []
        gh, gw = 8, 8
        cell_h, cell_w = max(1, H // gh), max(1, W // gw)
        scores = []
        for iy in range(gh):
            for ix in range(gw):
                y1, y2 = iy * cell_h, H if iy == gh - 1 else (iy + 1) * cell_h
                x1, x2 = ix * cell_w, W if ix == gw - 1 else (ix + 1) * cell_w
                score = low[y1:y2, x1:x2].float().mean().item()
                if score > 0.25:
                    scores.append((score, (x1, y1, x2, y2)))
        scores = sorted(scores, key=lambda x: x[0], reverse=True)[:2]
        return [VLARegion("geometry_ambiguity", b, float(s)) for s, b in scores]

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

        if self.depth_gate > 0 and "depth" in render_pkg:
            depth_map = render_pkg["depth"][0].detach()
        else:
            depth_map = None

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
        try:
            cams = self.scene.getTrainCameras()
        except Exception:
            return None
        if not cams:
            return None
        pc = pseudo_cam.camera_center.detach()
        best = None
        best_d = float("inf")
        for cam in cams:
            d = torch.norm(pc - cam.camera_center.detach()).item()
            if d < best_d:
                best_d = d
                best = cam
        if best is None or not hasattr(best, "original_image"):
            return None
        return best.original_image.detach()

    def _save_panel(self, rgb_path: str, depth_path: str, alpha_path: str, pseudo_cam, panel_path: str) -> None:
        rgb = Image.open(rgb_path).convert("RGB")
        depth = Image.open(depth_path).convert("RGB")
        alpha = Image.open(alpha_path).convert("RGB")
        nearest = self._nearest_train_image(pseudo_cam)
        if nearest is None:
            ref = Image.new("RGB", rgb.size, (0, 0, 0))
        else:
            ref = Image.fromarray(_tensor_to_uint8_image(nearest, normalize=False)).resize(rgb.size)

        W, H = rgb.size
        max_side = int(_getattr(self.args, "vla_panel_side", 512))
        scale = min(1.0, max_side / max(W, H))
        if scale < 1.0:
            size = (max(1, int(W * scale)), max(1, int(H * scale)))
            rgb = rgb.resize(size)
            depth = depth.resize(size)
            alpha = alpha.resize(size)
            ref = ref.resize(size)
            W, H = size
        panel = Image.new("RGB", (W * 2, H * 2), (0, 0, 0))
        panel.paste(rgb, (0, 0))
        panel.paste(depth, (W, 0))
        panel.paste(alpha, (0, H))
        panel.paste(ref, (W, H))
        draw = ImageDraw.Draw(panel)
        draw.text((8, 8), "pseudo render", fill=(255, 255, 255))
        draw.text((W + 8, 8), "pseudo depth", fill=(255, 255, 255))
        draw.text((8, H + 8), "pseudo alpha", fill=(255, 255, 255))
        draw.text((W + 8, H + 8), "nearest train image", fill=(255, 255, 255))
        panel.save(panel_path)
