import random

from omegaconf import OmegaConf


def _clone_cfg(cfg):
    return OmegaConf.create(OmegaConf.to_container(cfg))


def _find_guider(cfg, guider_name: str):
    for guider in cfg.guiders:
        if guider.get("name") == guider_name:
            return guider

    raise ValueError(f"Guider '{guider_name}' is not present in the config.")


def _stepwise(val: float, zero_from: int, total: int = 50):
    if not 0 <= zero_from <= total:
        raise ValueError(f"zero_from must be in [0, {total}], got {zero_from}.")

    return [val] * zero_from + [0.0] * (total - zero_from)


def generate_nonstyle_candidates(base_cfg, max_trials: int, seed: int = 0):
    if max_trials <= 0:
        return []

    rng = random.Random(seed)
    candidates = []

    for g_mul in [0.5, 1.0, 1.5, 2.0]:
        for self_attn_mul in [0.5, 1.0, 2.0]:
            for app_mul in [0.5, 1.0, 2.0]:
                for steps in [25, 30, 35]:
                    for init0, init1 in [(0.2, 3.0), (0.33, 3.0), (0.5, 3.0)]:
                        cfg = _clone_cfg(base_cfg)

                        guider = _find_guider(cfg, "self_attn_map_l2_appearance")
                        guider.g_scale = float(guider.g_scale) * g_mul
                        guider.kwargs.self_attn_gs = (
                            float(guider.kwargs.self_attn_gs) * self_attn_mul
                        )
                        guider.kwargs.app_gs = float(guider.kwargs.app_gs) * app_mul
                        guider.kwargs.total_first_steps = int(steps)

                        cfg.noise_rescaling_setup.init_setup = [float(init0), float(init1)]

                        params = {
                            "mode": "nonstyle",
                            "g_scale": float(guider.g_scale),
                            "self_attn_gs": float(guider.kwargs.self_attn_gs),
                            "app_gs": float(guider.kwargs.app_gs),
                            "total_first_steps": int(guider.kwargs.total_first_steps),
                            "init_setup": [float(init0), float(init1)],
                        }
                        candidates.append((cfg, params))

    rng.shuffle(candidates)
    return candidates[:max_trials]


def generate_style_candidates(
    base_cfg,
    max_trials: int,
    seed: int = 0,
    total_steps: int = 50,
):
    if max_trials <= 0:
        return []

    rng = random.Random(seed)
    candidates = []

    for val_attn in [50000.0, 75000.0, 100000.0, 200000.0]:
        for val_feat in [0.5, 1.25, 2.5, 5.0]:
            for zero_from in [20, 22, 25, 28, 30]:
                for init0, init1 in [(1.0, 1.0), (1.5, 1.5), (2.0, 2.0)]:
                    cfg = _clone_cfg(base_cfg)

                    g_attn = _find_guider(cfg, "self_attn_map_l2")
                    g_feat = _find_guider(cfg, "features_map_l2")

                    g_attn.g_scale = _stepwise(float(val_attn), int(zero_from), total_steps)
                    g_feat.g_scale = _stepwise(float(val_feat), int(zero_from), total_steps)

                    cfg.noise_rescaling_setup.init_setup = [float(init0), float(init1)]

                    params = {
                        "mode": "style",
                        "self_attn_value": float(val_attn),
                        "features_value": float(val_feat),
                        "zero_from_step": int(zero_from),
                        "init_setup": [float(init0), float(init1)],
                    }
                    candidates.append((cfg, params))

    rng.shuffle(candidates)
    return candidates[:max_trials]
