from typing import Optional, Union

import ImageReward as RM
import lpips
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class EditingMetrics:
    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        lpips_backbone: str = "alex",
        clip_model_name: str = "openai/clip-vit-large-patch14",
        imagereward_ckpt: str = "ImageReward-v1.0",
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.lpips = lpips.LPIPS(net=lpips_backbone).to(self.device).eval()

        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.device).eval()
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

        self.ir = RM.load(imagereward_ckpt).to(self.device)

    @torch.no_grad()
    def lpips_src_edit(self, src_pil: Image.Image, edit_pil: Image.Image) -> float:
        src_pil = src_pil.convert("RGB")
        edit_pil = edit_pil.convert("RGB")

        # [0, 255] -> [0, 1]
        src_tensor = TF.to_tensor(src_pil).unsqueeze(0).to(self.device)
        edit_tensor = TF.to_tensor(edit_pil).unsqueeze(0).to(self.device)

        # [0, 1] -> [-1, 1]
        src = src_tensor * 2.0 - 1.0
        edit = edit_tensor * 2.0 - 1.0

        val = self.lpips(src, edit)

        return float(val.squeeze().detach().cpu())

    @torch.no_grad()
    def clipscore_edit_prompt(self, edit_pil: Image.Image, prompt: str) -> float:
        edit_pil = edit_pil.convert("RGB")

        inputs = self.clip_processor(
            text=[prompt],
            images=[edit_pil],
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        out = self.clip_model(**inputs)
        img = out.image_embeds
        txt = out.text_embeds

        img = img / img.norm(dim=-1, keepdim=True)
        txt = txt / txt.norm(dim=-1, keepdim=True)

        cos = (img * txt).sum(dim=-1)

        return float(cos.item())

    def imagereward_edit_prompt(self, edit_pil: Image.Image, prompt: str) -> float:
        return float(self.ir.score(prompt, edit_pil.convert("RGB")))

    def __call__(
        self,
        src_img: Image.Image,
        edit_img: Union[Image.Image, np.ndarray],
        target_prompt: str,
    ) -> dict:
        src_pil = src_img.convert("RGB")
        if isinstance(edit_img, Image.Image):
            edit_pil = edit_img.convert("RGB")
        else:
            edit_pil = Image.fromarray(edit_img).convert("RGB")

        return {
            "LPIPS": self.lpips_src_edit(src_pil, edit_pil),
            "CLIPScore": self.clipscore_edit_prompt(edit_pil, target_prompt),
            "ImageReward": self.imagereward_edit_prompt(edit_pil, target_prompt),
        }
