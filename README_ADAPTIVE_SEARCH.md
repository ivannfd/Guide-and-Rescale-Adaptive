# Adaptive Search

## 1. Перебор по сетке

Перебор по сетке используется для поиска гиперпараметров Guide-and-Rescale на датасете PIE-Bench. Для каждого изображения строится набор кандидатов, каждый кандидат прогоняется через pipeline редактирования, после чего результат сравнивается с baseline-метриками.

Кандидат считается успешным, если одновременно:

- `ImageReward` выше baseline;
- `LPIPS` ниже baseline.

После того, как нашли успешного кандидата, поиск останавливается. Если подходящий кандидат не найден, в итоговую таблицу записываются baseline-значения.

Основной файл запуска:

```text
hyperparameter_search/run_pie_grid_search.py
```

Пример запуска:

```bash
python -m hyperparameter_search.run_pie_grid_search \
  --mapping-path /path/to/pie-bench/mapping_file.json \
  --images-root /path/to/pie-bench/annotation_images \
  --old-metrics-csv /path/to/pie_metrics.csv \
  --config-nonstyle configs/ours_nonstyle_best.yaml \
  --config-style configs/ours_style_best.yaml \
  --out-path pie_metrics_final_optimized.xlsx \
  --cuda-visible-devices 0 \
  --max-trials 15 \
  --seed 42
```

Все параметры CLI:

```bash
python -m hyperparameter_search.run_pie_grid_search --help
```

Файлы:

- `hyperparameter_search/optimize.py` - основной цикл перебора и сохранение результата;
- `hyperparameter_search/search_space.py` - генерация сетки гиперпараметров;
- `hyperparameter_search/metrics.py` - расчет `LPIPS`, `CLIPScore`, `ImageReward`;
- `hyperparameter_search/modeling.py` - загрузка scheduler и Stable Diffusion модели;
- `hyperparameter_search/pie_data.py` - чтение PIE-Bench mapping.

Входные данные:

- `--mapping-path` - путь к `mapping_file.json` из PIE-Bench;
- `--images-root` - директория с изображениями, пути внутри нее берутся из `image_path`;
- `--old-metrics-csv` - baseline CSV с колонками `Name`, `LPIPS`, `ImageReward`.

Результат сохраняется в `.xlsx` со столбцами:

- `name`;
- `old_ImageReward`;
- `new_ImageReward`;
- `old_LPIPS`;
- `new_LPIPS`;
- `new_params`.

`new_params` содержит JSON с параметрами первого кандидата, который улучшил обе целевые метрики. Если такого кандидата нет, поле остается пустым.

## 2. Адаптивный поиск гиперпараметров

Адаптивные стратегии лежат в:

```text
diffusion_core/guiders/
```

Они выбирают guidance-scale прямо во время denoising. На каждом DDIM-шаге стратегия пробует несколько значений scale, строит для них `pred_original_sample`, оценивает кандидатов и выбирает лучший scale для текущего шага.

### `diffusion_core/guiders/adaptive_cfg_normal.py`

Подбирает `cfg scale` через online-распределение `Normal(mu, sigma)`.

На каждом DDIM-шаге стратегия:

- сэмплирует `K` значений `cfg scale`;
- отсекает худшие варианты по `LPIPS`;
- выбирает лучшие по основной метрике;
- обновляет `mu` и `sigma` по elite-кандидатам.

### `diffusion_core/guiders/adaptive_cfg_grid.py`

Подбирает `cfg scale` по фиксированной сетке.

Значения берутся из:

```python
config.adaptive_gscale.coeff
```

На каждом шаге проверяются все значения из списка, затем выбирается лучший кандидат после `LPIPS` фильтрации.

### `diffusion_core/guiders/adaptive_energy_normal.py`

Подбирает scale для одного energy-guider через `Normal(mu, sigma)`.

Target-guider задается в конфиге через:

```python
config.adaptive_gscale.target_guider
```

Например, это может быть `features_map_l2`, `self_attn_map_l2` или другой guider, который возвращает energy.

### `diffusion_core/guiders/adaptive_energy_grid.py`

Подбирает scale для одного energy-guider по фиксированной сетке.

Значения берутся из:

```python
config.adaptive_gscale.coeff
```

### `diffusion_core/guiders/adaptive_multi_guidance.py`

Подбирает несколько guidance-scale одновременно.

Стратегия поддерживает несколько targets, например:

- `cfg`;
- `features_map_l2`;
- `self_attn_map_l2_appearance`;
- другие grad-guiders или energy-guiders из конфига.

Для каждого target задается свой search config: `K`, `M`, `mu0`, `sigma0`, `clip_min`, `clip_max`. На каждом шаге строятся комбинации scale для всех активных targets, затем выбирается лучший кандидат после `LPIPS` фильтрации.

## 3. Подключение адаптивного поиска

Пример для одного изображения:

```python
from pathlib import Path

import torch
from omegaconf import OmegaConf
from PIL import Image

from diffusion_core.guiders.adaptive_cfg_grid import AdaptiveGuidanceEditing
from diffusion_core.utils import load_512
from hyperparameter_search.modeling import get_model, get_scheduler


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

scheduler = get_scheduler("ddim_50_eps")
model = get_model(scheduler, "stable-diffusion-v1-4", device)

config = OmegaConf.load("configs/ours_nonstyle_best.yaml")

config.adaptive_gscale = {
    "enabled": True,
    "coeff": [3.0, 5.0, 7.5, 10.0, 12.0],
    "metric": {
        "type": "imagereward",
        "ckpt": "ImageReward-v1.0",
    },
    "lpips_drop_frac": 0.25,
    "lpips_backbone": "alex",
}

image_path = Path("example_images/zebra.jpeg")
source_image = Image.fromarray(load_512(str(image_path))).convert("RGB")

source_prompt = "a photo of a zebra"
target_prompt = "a photo of a horse"

guidance = AdaptiveGuidanceEditing(model, config)
edited_image = guidance(
    source_image,
    source_prompt,
    target_prompt,
    verbose=False,
)

Image.fromarray(edited_image).save("adaptive_result.png")
```

В этом примере адаптивно подбирается `cfg scale`, возможные значения берутся из фиксированного списка `coeff`.

Чтобы поменять стратегию, достаточно заменить импорт и секцию `adaptive_gscale`.

Для online-поиска `cfg scale` через `Normal(mu, sigma)`:

```python
from diffusion_core.guiders.adaptive_cfg_normal import AdaptiveGuidanceEditing

config.adaptive_gscale = {
    "enabled": True,
    "K": 8,
    "M": 3,
    "mu0": 7.5,
    "sigma0": 1.0,
    "clip_min": 3.0,
    "clip_max": 12.0,
    "metric": {"type": "imagereward"},
    "lpips_drop_frac": 0.25,
}
```

Если один и тот же объект `AdaptiveGuidanceEditing` используется для нескольких изображений, перед новым изображением нужно сбрасывать состояние поиска:

```python
guidance.reset_adaptive_searches(seed=42)
```
