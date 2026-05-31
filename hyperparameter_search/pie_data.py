import json
from pathlib import Path


def clean_prompt(prompt: str) -> str:
    return prompt.replace("[", "").replace("]", "")


def load_mapping(mapping_path: Path) -> dict:
    with mapping_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_entry_fields(entry: dict):
    name = entry["image_path"]
    init_prompt = clean_prompt(entry["original_prompt"])
    edit_prompt = clean_prompt(entry["editing_prompt"])
    edit_type = int(entry["editing_type_id"])
    return name, init_prompt, edit_prompt, edit_type
