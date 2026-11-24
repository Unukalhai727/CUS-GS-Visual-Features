from typing import override
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoImageProcessor, CLIPVisionModelWithProjection
from .types import VisualEncoder


class CLIPEncoder(VisualEncoder):
    model_id = "openai/clip-vit-large-patch14-336"
    logit_scale = 100.0
    
    def __init__(self):
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = CLIPVisionModelWithProjection.from_pretrained(self.model_id, device_map="auto")
        self.patch_size = self.model.config.patch_size

    @override
    @torch.inference_mode()
    def __call__(self, image_path: Path) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to("cuda")
        outputs = self.model(**inputs)
        embedding = self.model.visual_projection(outputs.last_hidden_state[:, 1:])
        embedding = embedding.view(inputs.pixel_values.shape[2] // self.patch_size, inputs.pixel_values.shape[3] // self.patch_size, -1)

        return embedding.to(device="cpu", dtype=torch.bfloat16)
