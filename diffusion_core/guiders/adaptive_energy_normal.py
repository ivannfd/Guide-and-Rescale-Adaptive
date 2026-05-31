from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from diffusion_core.guiders.guidance_editing import latent2image, GuidanceEditing


@dataclass
class AdaptiveStepInfo:
    step_index: int
    timestep: Any
    best_scale: float
    best_score: float
    mu: float
    sigma: float


class OnlineNormalScaleSearch:

    def __init__(
            self,
            K: int,
            M: int,
            mu0: float,
            sigma0: float,
            clip: Tuple[float, float] = (0.0, 10.0),
            update_every: int = 1,
            min_sigma: float = 1e-6,
            rng: Optional[np.random.Generator] = None,
    ):
        if K < 1:
            raise ValueError("K must be >= 1")
        if M < 1 or M > K:
            raise ValueError("M must be in [1, K]")
        self.K = int(K)
        self.M = int(M)
        self.mu = float(mu0)
        self.sigma = float(sigma0)
        self.clip = (float(clip[0]), float(clip[1]))
        self.update_every = int(update_every)
        self.min_sigma = float(min_sigma)
        self.rng = rng or np.random.default_rng()

    def sample(self) -> np.ndarray:
        s = self.rng.normal(self.mu, self.sigma, size=self.K).astype(np.float32)
        return np.clip(s, self.clip[0], self.clip[1])

    def update(self, step_index: int, top_scales: np.ndarray) -> None:
        if (step_index % self.update_every) != 0:
            return
        self.mu = float(np.mean(top_scales))
        self.sigma = float(max(np.std(top_scales) + 1e-6, self.min_sigma))


def _metric_latent_l2_to_inv0(
        x0_latents: torch.Tensor, data_dict: Dict[str, Any], guider_obj: "AdaptiveGuidanceEditing"
) -> float:
    ref = guider_obj.inv_latents[0].to(x0_latents.device).to(x0_latents.dtype)
    return float(-torch.mean((x0_latents - ref) ** 2).detach().item())


def _metric_latent_l1_to_inv0(
        x0_latents: torch.Tensor, data_dict: Dict[str, Any], guider_obj: "AdaptiveGuidanceEditing"
) -> float:
    ref = guider_obj.inv_latents[0].to(x0_latents.device).to(x0_latents.dtype)
    return float(-torch.mean(torch.abs(x0_latents - ref)).detach().item())


def _load_callable(path: str) -> Callable:
    if ":" in path:
        mod_name, attr = path.split(":", 1)
    else:
        parts = path.split(".")
        mod_name, attr = ".".join(parts[:-1]), parts[-1]
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr)
    if not callable(fn):
        raise TypeError(f"Metric '{path}' is not callable")
    return fn


def build_metric(
        metric_cfg: Optional[Dict[str, Any]]
) -> Callable[[torch.Tensor, Dict[str, Any], "AdaptiveGuidanceEditing"], float]:
    if metric_cfg is None:
        return _metric_latent_l2_to_inv0

    mtype = metric_cfg.get("type", "latent_l2_to_inv0")
    if mtype == "latent_l2_to_inv0":
        return _metric_latent_l2_to_inv0
    if mtype == "latent_l1_to_inv0":
        return _metric_latent_l1_to_inv0
    if mtype == "callable":
        path = metric_cfg.get("path")
        if not path:
            raise ValueError("metric.type='callable' requires metric.path")
        return _load_callable(path)
    if mtype == "imagereward":
        ckpt = metric_cfg.get("ckpt", "ImageReward-v1.0")
        device_str = metric_cfg.get("device", None)

        ir_model = None

        def _metric_imagereward(
                x0_latents: torch.Tensor, data_dict: Dict[str, Any], guider_obj: "AdaptiveGuidanceEditing"
        ) -> float:
            nonlocal ir_model
            prompt = getattr(guider_obj, "_trg_prompt", None) or ""

            if ir_model is None:
                import ImageReward as RM

                dev = (
                    torch.device(device_str)
                    if device_str is not None
                    else torch.device(getattr(guider_obj.model, "device", "cuda"))
                )
                ir_model = RM.load(ckpt).to(dev).eval()

            img_np = latent2image(x0_latents, guider_obj.model, return_type="np")[0]
            img_pil = Image.fromarray(img_np).convert("RGB")

            return float(ir_model.score(prompt, img_pil))

        return _metric_imagereward
    if mtype == "lpips":
        def _metric_lpips(x0_latents: torch.Tensor, data_dict: Dict[str, Any],
                          guider_obj: "AdaptiveGuidanceEditing") -> float:
            dev = x0_latents.device
            img_np = latent2image(x0_latents, guider_obj.model, return_type="np")[0]
            img_pil = Image.fromarray(img_np).convert("RGB")
            lp = guider_obj._lpips_src_edit_pil(img_pil, dev)
            return float(-lp)

        return _metric_lpips
    raise ValueError(f"Unknown adaptive metric type: {mtype}")


class AdaptiveGuidanceEditing(GuidanceEditing):

    def __init__(self, model, config):
        super().__init__(model, config)

        cfg = config.get("adaptive_gscale", None)
        self._adaptive_enabled = bool(cfg and cfg.get("enabled", False))
        self._adaptive_cfg = cfg or {}

        self._adaptive_target = self._adaptive_cfg.get("target_guider", "features_map_l2")

        self._respect_base_schedule = bool(self._adaptive_cfg.get("respect_base_schedule", True))

        self._lpips_drop_frac = float(self._adaptive_cfg.get("lpips_drop_frac", 0.25))  # drop worst 25%
        self._lpips_backbone = str(self._adaptive_cfg.get("lpips_backbone", "alex"))
        self._lpips_model = None
        self._lpips_TF = None
        self._src_pil: Optional[Image.Image] = None
        self._src_lpips_tensor: Optional[torch.Tensor] = None

        metric_cfg = self._adaptive_cfg.get("metric", {}) or {}
        self._metric_type = str(metric_cfg.get("type", "latent_l2_to_inv0"))

        self._ir_model = None
        self._ir_ckpt = str(metric_cfg.get("ckpt", "ImageReward-v1.0"))
        self._ir_device_str = metric_cfg.get("device", None)

        if self._adaptive_enabled:
            K = int(self._adaptive_cfg.get("K", 8))
            M = int(self._adaptive_cfg.get("M", 3))
            mu0 = float(self._adaptive_cfg.get("mu0", 1.0))
            sigma0 = float(self._adaptive_cfg.get("sigma0", 1.0))
            clip_min = float(self._adaptive_cfg.get("clip_min", 0.0))
            clip_max = float(self._adaptive_cfg.get("clip_max", 10.0))
            update_every = int(self._adaptive_cfg.get("update_every", 1))

            self._scale_search = OnlineNormalScaleSearch(
                K=K,
                M=M,
                mu0=mu0,
                sigma0=sigma0,
                clip=(clip_min, clip_max),
                update_every=update_every,
            )

            self._metric = build_metric(self._adaptive_cfg.get("metric", None))
        else:
            self._scale_search = None
            self._metric = None

        self.adaptive_history: list[AdaptiveStepInfo] = []

        self._inv_prompt: Optional[str] = None
        self._trg_prompt: Optional[str] = None

    def train(
            self,
            image_gt,
            inv_prompt: str,
            trg_prompt: str,
            control_image=None,
            verbose: bool = False,
    ):
        self._inv_prompt = inv_prompt
        self._trg_prompt = trg_prompt
        try:
            self._src_pil = image_gt.convert("RGB")
        except Exception:
            self._src_pil = None
        self._src_lpips_tensor = None

        return super().train(image_gt, inv_prompt, trg_prompt, control_image, verbose)

    def _split_guidance_components(
            self,
            data_dict: Dict[str, Any],
            diffusion_iter: int,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Optional[torch.Tensor], float]:
        noises_fixed: Dict[str, torch.Tensor] = {"uncond": data_dict["uncond_unet"]}

        for name, (guider, g_scale) in self.guiders.items():
            if guider.grad_guider:
                noises_fixed[name] = self._get_scale(g_scale, diffusion_iter) * guider(data_dict)

        latent = data_dict["latent"]
        energy_grads: Dict[str, torch.Tensor] = {}
        target_grad_raw: Optional[torch.Tensor] = None
        target_base_scale: float = 0.0

        for name, (guider, g_scale) in self.guiders.items():
            if guider.grad_guider:
                continue

            energy_raw = guider(data_dict)
            if not torch.is_tensor(energy_raw):
                continue
            if torch.allclose(
                    energy_raw.detach(),
                    torch.tensor(0.0, device=energy_raw.device, dtype=energy_raw.dtype),
            ):
                continue

            grad_raw = torch.autograd.grad(
                energy_raw,
                latent,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )[0]
            if grad_raw is None:
                continue

            energy_grads[name] = grad_raw

            if name == self._adaptive_target:
                target_grad_raw = grad_raw
                target_base_scale = float(self._get_scale(g_scale, diffusion_iter))

        return noises_fixed, energy_grads, target_grad_raw, target_base_scale

    @torch.no_grad()
    def _compose_noise_pred(self, noises: Dict[str, torch.Tensor], index: int) -> torch.Tensor:
        scales = self.noise_rescaler(noises, index)
        first = next(iter(noises.values()))
        out = torch.zeros_like(first)
        for k, v in noises.items():
            out = out + scales[k] * v
        return out

    def _ensure_lpips(self, device: torch.device):
        if self._lpips_model is not None:
            return
        import lpips
        from torchvision.transforms import functional as TF
        self._lpips_TF = TF
        self._lpips_model = lpips.LPIPS(net=self._lpips_backbone).to(device).eval()

    def _lpips_src_edit_pil(self, edit_pil: Image.Image, device: torch.device) -> float:
        if self._src_pil is None:
            return float("inf")

        self._ensure_lpips(device)

        if self._src_lpips_tensor is None:
            src = self._lpips_TF.to_tensor(self._src_pil).unsqueeze(0).to(device)
            self._src_lpips_tensor = src * 2.0 - 1.0  # [-1,1]

        edt = self._lpips_TF.to_tensor(edit_pil).unsqueeze(0).to(device)
        edt = edt * 2.0 - 1.0

        val = self._lpips_model(self._src_lpips_tensor, edt)
        return float(val.squeeze().detach().cpu().item())

    def _ensure_imagereward(self, device: torch.device):
        if self._ir_model is not None:
            return
        import ImageReward as RM
        self._ir_model = RM.load(self._ir_ckpt).to(device).eval()

    def _adaptive_choose_noise_pred(
            self,
            *,
            latents: torch.Tensor,
            timestep: Any,
            diffusion_iter: int,
            noises_fixed: Dict[str, torch.Tensor],
            fixed_other: Optional[torch.Tensor],
            target_grad_raw: Optional[torch.Tensor],
            target_base_scale: float,
            data_dict: Dict[str, Any],
    ) -> torch.Tensor:
        if (not self._adaptive_enabled) or (self._scale_search is None) or (target_grad_raw is None):
            other = fixed_other
            if target_grad_raw is not None and target_base_scale != 0.0:
                base = fixed_other if fixed_other is not None else 0.0
                other = base + target_base_scale * target_grad_raw

            noises = dict(noises_fixed)
            if other is not None:
                noises["other"] = other
            return self._compose_noise_pred(noises, diffusion_iter)

        if self._respect_base_schedule and (target_base_scale == 0.0):
            noises = dict(noises_fixed)
            if fixed_other is not None:
                noises["other"] = fixed_other
            return self._compose_noise_pred(noises, diffusion_iter)

        latents_detached = latents.detach()

        scales = self._scale_search.sample()
        K = int(scales.shape[0])

        ir_scores = np.empty((K,), dtype=np.float32)
        lpips_vals = np.empty((K,), dtype=np.float32)
        noise_preds: list[torch.Tensor] = []

        metric_device = latents_detached.device

        use_ir_fast = (self._metric_type == "imagereward")

        if use_ir_fast:
            dev = (
                torch.device(self._ir_device_str)
                if self._ir_device_str is not None
                else metric_device
            )
            self._ensure_imagereward(dev)

        prompt = getattr(self, "_trg_prompt", None) or ""

        with torch.no_grad():
            for j, s in enumerate(scales):
                other = fixed_other
                if other is None:
                    other = float(s) * target_grad_raw
                else:
                    other = other + float(s) * target_grad_raw

                noises = dict(noises_fixed)
                if other is not None:
                    noises["other"] = other

                noise_pred = self._compose_noise_pred(noises, diffusion_iter)
                noise_preds.append(noise_pred)

                out = self.model.scheduler.step_backward(noise_pred, timestep, latents_detached)
                x0 = out.pred_original_sample

                img_np = latent2image(x0, self.model, return_type="np")[0]
                img_pil = Image.fromarray(img_np).convert("RGB")

                lpips_vals[j] = float(self._lpips_src_edit_pil(img_pil, metric_device))

                if use_ir_fast:
                    ir_scores[j] = float(self._ir_model.score(prompt, img_pil))
                else:
                    ir_scores[j] = float(self._metric(x0, data_dict, self))

        drop_frac = float(self._lpips_drop_frac)
        keep_n = max(1, int(np.ceil(K * (1.0 - drop_frac))))
        keep_idx = np.argsort(lpips_vals)[:keep_n]

        ir_keep = ir_scores[keep_idx]
        order = np.argsort(ir_keep)[::-1]
        elite_n = min(int(self._scale_search.M), keep_n)
        elite_idx = keep_idx[order[:elite_n]]

        top_scales = scales[elite_idx]
        self._scale_search.update(diffusion_iter, top_scales)

        best_i = int(elite_idx[0])
        best_scale = float(scales[best_i])
        best_score = float(ir_scores[best_i])

        self.adaptive_history.append(
            AdaptiveStepInfo(
                step_index=diffusion_iter,
                timestep=timestep,
                best_scale=best_scale,
                best_score=best_score,
                mu=self._scale_search.mu,
                sigma=self._scale_search.sigma,
            )
        )

        return noise_preds[best_i]

    def edit(self):
        self.model.scheduler.set_timesteps(self.model.scheduler.num_inference_steps)
        latents = self.start_latent
        self.latents_stack = []
        self.adaptive_history = []

        for i, timestep in enumerate(self.model.scheduler.timesteps):
            data_dict = self._construct_data_dict(latents, i, timestep)

            noises_fixed, energy_grads, target_grad_raw, target_base_scale = self._split_guidance_components(data_dict,
                                                                                                             i)

            fixed_other: Optional[torch.Tensor] = None
            for name, grad_raw in energy_grads.items():
                if name == self._adaptive_target:
                    continue
                _, g_scale = self.guiders[name]
                scale_here = float(self._get_scale(g_scale, i))
                if scale_here == 0.0:
                    continue
                contrib = scale_here * grad_raw
                fixed_other = contrib if fixed_other is None else (fixed_other + contrib)

            noise_pred = self._adaptive_choose_noise_pred(
                latents=latents,
                timestep=timestep,
                diffusion_iter=i,
                noises_fixed=noises_fixed,
                fixed_other=fixed_other,
                target_grad_raw=target_grad_raw,
                target_base_scale=target_base_scale,
                data_dict=data_dict,
            )

            latents = self._step(noise_pred, timestep, latents)

            for g_name, (guider, _) in self.guiders.items():
                if not guider.grad_guider:
                    guider.clear_outputs()

            del data_dict
            torch.cuda.empty_cache()

        self._model_unpatch(self.model)

        return latent2image(latents, self.model)[0]

