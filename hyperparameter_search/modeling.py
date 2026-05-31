from diffusion_core import diffusion_models_registry, diffusion_schedulers_registry


def get_scheduler(scheduler_name: str):
    if scheduler_name not in diffusion_schedulers_registry:
        raise ValueError(
            f"Incorrect scheduler type: {scheduler_name}. "
            f"Possible values: {list(diffusion_schedulers_registry.keys())}"
        )

    return diffusion_schedulers_registry[scheduler_name]()


def get_model(scheduler, model_name: str, device):
    if model_name not in diffusion_models_registry:
        raise ValueError(
            f"Incorrect model name: {model_name}. "
            f"Possible values: {list(diffusion_models_registry.keys())}"
        )

    model = diffusion_models_registry[model_name](scheduler)
    model.to(device)
    return model
