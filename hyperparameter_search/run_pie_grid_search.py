import os
from pathlib import Path
from typing import Optional

import click


@click.command(context_settings={"show_default": True})
@click.option(
    "--mapping-path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Path to the PIE-Bench mapping_file.json.",
)
@click.option(
    "--images-root",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    required=True,
    help="Directory with PIE-Bench images.",
)
@click.option(
    "--old-metrics-csv",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="CSV with baseline metrics: Name, LPIPS, ImageReward.",
)
@click.option(
    "--config-nonstyle",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("configs/ours_nonstyle_best.yaml"),
    help="Config for non-style edits.",
)
@click.option(
    "--config-style",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("configs/ours_style_best.yaml"),
    help="Config for style edits.",
)
@click.option(
    "--out-path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=Path("pie_metrics_final_optimized.xlsx"),
    help="Path for the grid-search results.",
)
@click.option(
    "--scheduler-name",
    default="ddim_50_eps",
    help="Scheduler name from diffusion_schedulers_registry.",
)
@click.option(
    "--model-name",
    default="stable-diffusion-v1-4",
    help="Model name from diffusion_models_registry.",
)
@click.option(
    "--max-trials",
    type=click.IntRange(min=1),
    default=15,
    help="Maximum number of candidates per image.",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Seed for shuffling the grid and comparing candidates reproducibly.",
)
@click.option(
    "--cuda-visible-devices",
    default=None,
    help="CUDA_VISIBLE_DEVICES value",
)
@click.option(
    "--verbose-search",
    is_flag=True,
    help="Print verbose logs from the diffusion pipeline.",
)
@click.option(
    "--deterministic/--no-deterministic",
    default=True,
    help="Enable project-level deterministic settings before the run.",
)
def main(
    mapping_path: Path,
    images_root: Path,
    old_metrics_csv: Path,
    config_nonstyle: Path,
    config_style: Path,
    out_path: Path,
    scheduler_name: str,
    model_name: str,
    max_trials: int,
    seed: int,
    cuda_visible_devices: Optional[str],
    verbose_search: bool,
    deterministic: bool,
):
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    import torch
    from omegaconf import OmegaConf

    from diffusion_core.utils import use_deterministic
    from .metrics import EditingMetrics
    from .modeling import get_model, get_scheduler
    from .optimize import load_old_metrics_csv, optimize_all
    from .pie_data import load_mapping

    if deterministic:
        use_deterministic()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    click.echo(f"Device: {device}")
    click.echo(f"Scheduler: {scheduler_name}")
    click.echo(f"Model: {model_name}")

    scheduler = get_scheduler(scheduler_name)
    model = get_model(scheduler, model_name, device)

    mapping = load_mapping(mapping_path)
    old_metrics = load_old_metrics_csv(old_metrics_csv)
    nonstyle_cfg = OmegaConf.load(config_nonstyle)
    style_cfg = OmegaConf.load(config_style)

    evaluator = EditingMetrics(device=device)

    saved_path = optimize_all(
        mapping=mapping,
        images_root=images_root,
        model=model,
        evaluator=evaluator,
        config_nonstyle=nonstyle_cfg,
        config_style=style_cfg,
        old_metrics=old_metrics,
        out_path=out_path,
        max_trials=max_trials,
        seed=seed,
        verbose_search=verbose_search,
    )

    click.echo(f"Saved: {saved_path.resolve()}")


if __name__ == "__main__":
    main()
