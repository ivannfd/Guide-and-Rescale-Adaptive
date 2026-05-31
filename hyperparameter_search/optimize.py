import csv
import json
import random
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from openpyxl import Workbook
from PIL import Image

from diffusion_core.guiders import GuidanceEditing
from diffusion_core.utils import load_512

from .pie_data import get_entry_fields
from .search_space import generate_nonstyle_candidates, generate_style_candidates


def load_old_metrics_csv(path: Path) -> dict:
    old = {}

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            name = row["Name"]
            old[name] = {
                "LPIPS": float(row["LPIPS"]),
                "ImageReward": float(row["ImageReward"]),
            }
            if row.get("CLIPScore"):
                old[name]["CLIPScore"] = float(row["CLIPScore"])

    return old


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _serialize_params(params) -> Optional[str]:
    if params is None:
        return None

    try:
        return json.dumps(params, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(params)


def _resolve_output_path(
    out_path: Optional[Union[str, Path]],
    out_csv: Optional[Union[str, Path]],
) -> Path:
    if out_path is None:
        out_path = out_csv

    if out_path is None:
        raise ValueError("Either out_path or out_csv must be provided.")

    out_path = Path(out_path)

    if out_path.suffix.lower() != ".xlsx":
        out_path = out_path.with_suffix(".xlsx")

    return out_path


def optimize_all(
    mapping: dict,
    images_root: Path,
    model,
    evaluator,
    config_nonstyle,
    config_style,
    old_metrics: dict,
    out_path: Optional[Union[str, Path]] = None,
    out_csv: Optional[Union[str, Path]] = None,
    max_trials: int = 10,
    seed: int = 42,
    verbose_search: bool = False,
) -> Path:
    if max_trials <= 0:
        raise ValueError(f"max_trials must be positive, got {max_trials}.")

    set_global_seed(seed)

    out_path = _resolve_output_path(out_path, out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "results"
    worksheet.append([
        "name",
        "old_ImageReward",
        "new_ImageReward",
        "old_LPIPS",
        "new_LPIPS",
        "new_params",
    ])
    workbook.save(out_path)

    images_root = Path(images_root)

    for image_idx, (_, entry) in enumerate(mapping.items()):
        name, init_prompt, edit_prompt, edit_type = get_entry_fields(entry)

        if name not in old_metrics:
            print(f"[WARN] missing baseline metrics for '{name}', skip")
            continue

        old_lp = float(old_metrics[name]["LPIPS"])
        old_ir = float(old_metrics[name]["ImageReward"])

        image_path = images_root / name
        if not image_path.exists():
            print(f"[WARN] missing image: {image_path}, skip")
            continue

        src_img = Image.fromarray(load_512(str(image_path))).convert("RGB")

        if edit_type >= 8:
            candidates = generate_style_candidates(
                config_style,
                max_trials=max_trials,
                seed=seed,
            )
        else:
            candidates = generate_nonstyle_candidates(
                config_nonstyle,
                max_trials=max_trials,
                seed=seed,
            )

        best_metrics = None
        best_params = None
        image_seed = seed + image_idx

        for trial_idx, (cfg, params) in enumerate(candidates):
            set_global_seed(image_seed)

            guidance = GuidanceEditing(model, cfg)
            edit_img = guidance(src_img, init_prompt, edit_prompt, verbose=verbose_search)
            edit_pil = Image.fromarray(edit_img).convert("RGB")

            new_ir = float(evaluator.imagereward_edit_prompt(edit_pil, edit_prompt))
            new_lpips = float(evaluator.lpips_src_edit(src_img, edit_pil))
            is_improved = new_ir > old_ir and new_lpips < old_lp

            print(
                f"{name} | trial={trial_idx} | "
                f"IR: {new_ir:.6f} (old {old_ir:.6f}) | "
                f"LPIPS: {new_lpips:.6f} (old {old_lp:.6f}) | "
                f"improved={is_improved}"
            )

            if is_improved:
                best_metrics = {
                    "ImageReward": new_ir,
                    "LPIPS": new_lpips,
                }
                best_params = params
                break

        if best_metrics is None:
            row = [
                name,
                old_ir,
                old_ir,
                old_lp,
                old_lp,
                None,
            ]
        else:
            row = [
                name,
                old_ir,
                float(best_metrics["ImageReward"]),
                old_lp,
                float(best_metrics["LPIPS"]),
                _serialize_params(best_params),
            ]

        worksheet.append(row)
        workbook.save(out_path)

    return out_path
