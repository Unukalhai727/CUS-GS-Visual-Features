from typing import override
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoImageProcessor, Dinov2Model
from .types import VisualEncoder


class DINOv2Encoder(VisualEncoder):
    model_id = "facebook/dinov2-giant"
    
    def __init__(self):
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model: Dinov2Model = Dinov2Model.from_pretrained(self.model_id, device_map="auto")
        self.patch_size = self.model.config.patch_size

    @override
    @torch.inference_mode()
    def __call__(self, image_path: Path) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to("cuda")
        outputs = self.model(**inputs).last_hidden_state
        embedding = outputs[:, 1:].view(inputs.pixel_values.shape[2] // self.patch_size, inputs.pixel_values.shape[3] // self.patch_size, -1)
        return embedding.to(device="cpu", dtype=torch.bfloat16)
