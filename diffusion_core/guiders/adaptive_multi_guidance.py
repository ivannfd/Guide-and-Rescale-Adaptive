from __future__ import annotations

import importlib
import itertools
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, List

import numpy as np
import torch
from PIL import Image

from diffusion_core.guiders.guidance_editing import latent2image, GuidanceEditing


@dataclass
class AdaptiveStepInfo:
    step_index: int
    timestep: Any
    best_scales: Dict[str, float]
    best_score: float
    best_lpips: float
    mu: Dict[str, float]
    sigma: Dict[str, float]


class AdaptiveScaleSearch:
    def __init__(
        self,
        *,
        name: str,
        cfg: Dict[str, Any],
    ):
        self.name = str(name)

        self.enabled = bool(cfg.get("enabled", True))
        self.respect_base_schedule = bool(cfg.get("respect_base_schedule", True))

        self.K = int(cfg.get("K", 8))
        self.M = int(cfg.get("M", 3))

        if self.K < 1:
            raise ValueError(f"adaptive_gscale.targets.{name}.K must be >= 1")

        if self.M < 1 or self.M > self.K:
            raise ValueError(f"adaptive_gscale.targets.{name}.M must be in [1, K]")

        self.mu0 = float(cfg.get("mu0", 1.0))
        self.sigma0 = float(cfg.get("sigma0", 1.0))

        self.mu = self.mu0
        self.sigma = self.sigma0

        self.clip = (
            float(cfg.get("clip_min", 0.0)),
            float(cfg.get("clip_max", 10.0)),
        )

        self.update_every = int(cfg.get("update_every", 1))
        self.min_sigma = float(cfg.get("min_sigma", 1e-6))

        self.rng = np.random.default_rng()

    def reset(self, seed: Optional[int] = None) -> None:
        self.mu = self.mu0
        self.sigma = self.sigma0
        self.rng = np.random.default_rng(seed)

    def sample(self) -> np.ndarray:
        s = self.rng.normal(
            loc=self.mu,
            scale=self.sigma,
            size=self.K,
        ).astype(np.float32)

        return np.clip(s, self.clip[0], self.clip[1])

    def update(self, step_index: int, top_scales: np.ndarray) -> None:
        if top_scales.size == 0:
            return

        if (step_index % self.update_every) != 0:
            return

        self.mu = float(np.mean(top_scales))
        self.sigma = float(max(np.std(top_scales) + 1e-6, self.min_sigma))


def _metric_latent_l2_to_inv0(
    x0_latents: torch.Tensor,
    data_dict: Dict[str, Any],
    guider_obj: "AdaptiveGuidanceEditing",
) -> float:
    ref = guider_obj.inv_latents[0].to(x0_latents.device).to(x0_latents.dtype)
    return float(-torch.mean((x0_latents - ref) ** 2).detach().item())


def _metric_latent_l1_to_inv0(
    x0_latents: torch.Tensor,
    data_dict: Dict[str, Any],
    guider_obj: "AdaptiveGuidanceEditing",
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
    metric_cfg: Optional[Dict[str, Any]],
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
            x0_latents: torch.Tensor,
            data_dict: Dict[str, Any],
            guider_obj: "AdaptiveGuidanceEditing",
        ) -> float:
            nonlocal ir_model

            prompt = getattr(guider_obj, "_trg_prompt", None) or ""

            if ir_model is None:
                import ImageReward as RM

                dev = (
                    torch.device(device_str)
                    if device_str is not None
                    else torch.device(getattr(guider_obj.model, "device", x0_latents.device))
                )

                ir_model = RM.load(ckpt).to(dev).eval()

            img_np = latent2image(x0_latents, guider_obj.model, return_type="np")[0]
            img_pil = Image.fromarray(img_np).convert("RGB")

            return float(ir_model.score(prompt, img_pil))

        return _metric_imagereward

    if mtype == "lpips":
        def _metric_lpips(
            x0_latents: torch.Tensor,
            data_dict: Dict[str, Any],
            guider_obj: "AdaptiveGuidanceEditing",
        ) -> float:
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

        self._lpips_drop_frac = float(self._adaptive_cfg.get("lpips_drop_frac", 0.25))
        self._lpips_backbone = str(self._adaptive_cfg.get("lpips_backbone", "alex"))

        self._lpips_model = None
        self._lpips_TF = None
        self._src_pil: Optional[Image.Image] = None
        self._src_lpips_tensor: Optional[torch.Tensor] = None

        metric_cfg = self._adaptive_cfg.get("metric", {}) or {}
        self._metric_type = str(metric_cfg.get("type", "imagereward"))

        self._ir_model = None
        self._ir_ckpt = str(metric_cfg.get("ckpt", "ImageReward-v1.0"))
        self._ir_device_str = metric_cfg.get("device", None)

        self._metric = build_metric(metric_cfg)

        self._target_searches: Dict[str, AdaptiveScaleSearch] = {}

        if self._adaptive_enabled:
            targets_cfg = self._adaptive_cfg.get("targets", None)

            if targets_cfg is None:
                target_name = str(self._adaptive_cfg.get("target_guider", "cfg"))
                targets_cfg = {
                    target_name: self._adaptive_cfg,
                }

            for target_name in targets_cfg:
                target_cfg = targets_cfg[target_name]
                if bool(target_cfg.get("enabled", True)):
                    self._target_searches[str(target_name)] = AdaptiveScaleSearch(
                        name=str(target_name),
                        cfg=target_cfg,
                    )

        self.adaptive_history: List[AdaptiveStepInfo] = []

        self._inv_prompt: Optional[str] = None
        self._trg_prompt: Optional[str] = None

    def reset_adaptive_searches(self, seed: Optional[int] = None) -> None:
        for idx, search in enumerate(self._target_searches.values()):
            local_seed = None if seed is None else seed + idx * 100_003
            search.reset(local_seed)

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

    def _ensure_lpips(self, device: torch.device) -> None:
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
            self._src_lpips_tensor = src * 2.0 - 1.0

        edt = self._lpips_TF.to_tensor(edit_pil).unsqueeze(0).to(device)
        edt = edt * 2.0 - 1.0

        val = self._lpips_model(self._src_lpips_tensor, edt)
        return float(val.squeeze().detach().cpu().item())

    def _ensure_imagereward(self, device: torch.device) -> None:
        if self._ir_model is not None:
            return

        import ImageReward as RM

        self._ir_model = RM.load(self._ir_ckpt).to(device).eval()

    @torch.no_grad()
    def _compose_noise_pred(
        self,
        noises: Dict[str, torch.Tensor],
        index: int,
    ) -> torch.Tensor:
        scales = self.noise_rescaler(noises, index)

        first = next(iter(noises.values()))
        out = torch.zeros_like(first)

        for k, v in noises.items():
            out = out + scales[k] * v

        return out

    def _target_is_active(
        self,
        *,
        target_name: str,
        raw: Optional[torch.Tensor],
        base_scale: float,
    ) -> bool:
        if not self._adaptive_enabled:
            return False

        search = self._target_searches.get(target_name)
        if search is None:
            return False

        if raw is None:
            return False

        if search.respect_base_schedule and base_scale == 0.0:
            return False

        return True

    def _split_guidance_components(
        self,
        data_dict: Dict[str, Any],
        diffusion_iter: int,
    ):
        noises_fixed: Dict[str, torch.Tensor] = {
            "uncond": data_dict["uncond_unet"],
        }

        target_noise_raws: Dict[str, torch.Tensor] = {}
        target_energy_grads: Dict[str, torch.Tensor] = {}
        target_base_scales: Dict[str, float] = {}

        fixed_other: Optional[torch.Tensor] = None

        latent = data_dict["latent"]

        for name, (guider, g_scale) in self.guiders.items():
            if not guider.grad_guider:
                continue

            raw = guider(data_dict)
            base_scale = float(self._get_scale(g_scale, diffusion_iter))

            if name in self._target_searches:
                target_noise_raws[name] = raw
                target_base_scales[name] = base_scale
            else:
                if base_scale != 0.0:
                    noises_fixed[name] = base_scale * raw

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

            base_scale = float(self._get_scale(g_scale, diffusion_iter))

            if name in self._target_searches:
                target_energy_grads[name] = grad_raw
                target_base_scales[name] = base_scale
            else:
                if base_scale == 0.0:
                    continue

                contrib = base_scale * grad_raw
                fixed_other = contrib if fixed_other is None else fixed_other + contrib

        return (
            noises_fixed,
            fixed_other,
            target_noise_raws,
            target_energy_grads,
            target_base_scales,
        )

    def _build_candidate_noise_pred(
        self,
        *,
        noises_fixed: Dict[str, torch.Tensor],
        fixed_other: Optional[torch.Tensor],
        target_noise_raws: Dict[str, torch.Tensor],
        target_energy_grads: Dict[str, torch.Tensor],
        candidate_scales: Dict[str, float],
        diffusion_iter: int,
    ) -> torch.Tensor:
        noises = dict(noises_fixed)


        for name, raw in target_noise_raws.items():
            if name not in candidate_scales:
                continue

            noises[name] = float(candidate_scales[name]) * raw

        other = fixed_other

        for name, grad_raw in target_energy_grads.items():
            if name not in candidate_scales:
                continue

            contrib = float(candidate_scales[name]) * grad_raw
            other = contrib if other is None else other + contrib

        if other is not None:
            noises["other"] = other

        return self._compose_noise_pred(noises, diffusion_iter)

    def _build_base_noise_pred(
        self,
        *,
        noises_fixed: Dict[str, torch.Tensor],
        fixed_other: Optional[torch.Tensor],
        target_noise_raws: Dict[str, torch.Tensor],
        target_energy_grads: Dict[str, torch.Tensor],
        target_base_scales: Dict[str, float],
        diffusion_iter: int,
    ) -> torch.Tensor:
        noises = dict(noises_fixed)

        for name, raw in target_noise_raws.items():
            base_scale = float(target_base_scales.get(name, 0.0))
            if base_scale != 0.0:
                noises[name] = base_scale * raw

        other = fixed_other

        for name, grad_raw in target_energy_grads.items():
            base_scale = float(target_base_scales.get(name, 0.0))
            if base_scale == 0.0:
                continue

            contrib = base_scale * grad_raw
            other = contrib if other is None else other + contrib

        if other is not None:
            noises["other"] = other

        return self._compose_noise_pred(noises, diffusion_iter)

    def _adaptive_choose_noise_pred(
        self,
        *,
        latents: torch.Tensor,
        timestep: Any,
        diffusion_iter: int,
        noises_fixed: Dict[str, torch.Tensor],
        fixed_other: Optional[torch.Tensor],
        target_noise_raws: Dict[str, torch.Tensor],
        target_energy_grads: Dict[str, torch.Tensor],
        target_base_scales: Dict[str, float],
        data_dict: Dict[str, Any],
    ) -> torch.Tensor:
        active_targets: List[str] = []

        for name, raw in target_noise_raws.items():
            base_scale = float(target_base_scales.get(name, 0.0))
            if self._target_is_active(target_name=name, raw=raw, base_scale=base_scale):
                active_targets.append(name)

        for name, grad_raw in target_energy_grads.items():
            base_scale = float(target_base_scales.get(name, 0.0))
            if self._target_is_active(target_name=name, raw=grad_raw, base_scale=base_scale):
                active_targets.append(name)

        if len(active_targets) == 0:
            return self._build_base_noise_pred(
                noises_fixed=noises_fixed,
                fixed_other=fixed_other,
                target_noise_raws=target_noise_raws,
                target_energy_grads=target_energy_grads,
                target_base_scales=target_base_scales,
                diffusion_iter=diffusion_iter,
            )

        latents_detached = latents.detach()

        samples_by_target: Dict[str, np.ndarray] = {
            name: self._target_searches[name].sample()
            for name in active_targets
        }

        target_names = list(samples_by_target.keys())
        target_values = [samples_by_target[name] for name in target_names]

        candidates: List[Dict[str, float]] = []

        for values in itertools.product(*target_values):
            candidates.append({
                name: float(value)
                for name, value in zip(target_names, values)
            })

        n_candidates = len(candidates)

        scores = np.empty((n_candidates,), dtype=np.float32)
        lpips_vals = np.empty((n_candidates,), dtype=np.float32)
        noise_preds: List[torch.Tensor] = []

        metric_device = latents_detached.device
        use_ir_fast = self._metric_type == "imagereward"

        if use_ir_fast:
            dev = (
                torch.device(self._ir_device_str)
                if self._ir_device_str is not None
                else metric_device
            )
            self._ensure_imagereward(dev)

        prompt = getattr(self, "_trg_prompt", None) or ""

        with torch.no_grad():
            for j, candidate_scales in enumerate(candidates):
                noise_pred = self._build_candidate_noise_pred(
                    noises_fixed=noises_fixed,
                    fixed_other=fixed_other,
                    target_noise_raws=target_noise_raws,
                    target_energy_grads=target_energy_grads,
                    candidate_scales=candidate_scales,
                    diffusion_iter=diffusion_iter,
                )

                noise_preds.append(noise_pred)

                out = self.model.scheduler.step_backward(
                    noise_pred,
                    timestep,
                    latents_detached,
                )

                x0 = out.pred_original_sample

                img_np = latent2image(x0, self.model, return_type="np")[0]
                img_pil = Image.fromarray(img_np).convert("RGB")

                lpips_vals[j] = float(self._lpips_src_edit_pil(img_pil, metric_device))

                if use_ir_fast:
                    scores[j] = float(self._ir_model.score(prompt, img_pil))
                else:
                    scores[j] = float(self._metric(x0, data_dict, self))

        drop_frac = float(self._lpips_drop_frac)
        keep_n = max(1, int(np.ceil(n_candidates * (1.0 - drop_frac))))
        keep_idx = np.argsort(lpips_vals)[:keep_n]

        scores_keep = scores[keep_idx]
        order = np.argsort(scores_keep)[::-1]

        max_M = max(self._target_searches[name].M for name in active_targets)
        elite_n = min(max_M, keep_n)

        elite_idx = keep_idx[order[:elite_n]]

        best_i = int(elite_idx[0])
        best_scales = candidates[best_i]
        best_score = float(scores[best_i])
        best_lpips = float(lpips_vals[best_i])

        for name in active_targets:
            search = self._target_searches[name]
            m = min(search.M, len(elite_idx))

            top_values = np.asarray(
                [candidates[int(idx)][name] for idx in elite_idx[:m]],
                dtype=np.float32,
            )

            search.update(diffusion_iter, top_values)

        self.adaptive_history.append(
            AdaptiveStepInfo(
                step_index=diffusion_iter,
                timestep=timestep,
                best_scales={
                    name: float(value)
                    for name, value in best_scales.items()
                },
                best_score=best_score,
                best_lpips=best_lpips,
                mu={
                    name: float(self._target_searches[name].mu)
                    for name in active_targets
                },
                sigma={
                    name: float(self._target_searches[name].sigma)
                    for name in active_targets
                },
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

            (
                noises_fixed,
                fixed_other,
                target_noise_raws,
                target_energy_grads,
                target_base_scales,
            ) = self._split_guidance_components(data_dict, i)

            noise_pred = self._adaptive_choose_noise_pred(
                latents=latents,
                timestep=timestep,
                diffusion_iter=i,
                noises_fixed=noises_fixed,
                fixed_other=fixed_other,
                target_noise_raws=target_noise_raws,
                target_energy_grads=target_energy_grads,
                target_base_scales=target_base_scales,
                data_dict=data_dict,
            )

            latents = self._step(noise_pred, timestep, latents)

            for _, (guider, _) in self.guiders.items():
                if not guider.grad_guider:
                    guider.clear_outputs()

            del data_dict
            torch.cuda.empty_cache()

        self._model_unpatch(self.model)

        return latent2image(latents, self.model)[0]
