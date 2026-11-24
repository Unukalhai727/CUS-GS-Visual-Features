from typing import override
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as T

from visual_encoder import VisualEncoder
from .modeling_clip import clip


class CLIPEncoder(VisualEncoder):
    model_id = "ViT-L/14@336px"
    patch_size = 14
    logit_scale = 100.0
    
    def __init__(self):
        self.model, self.preprocess = clip.load(self.model_id, device="cuda")
        self.transform = T.Compose([
            T.Resize(336),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    @override
    @torch.inference_mode()
    def __call__(self, image_path: Path) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        inputs = self.preprocess(image).unsqueeze(0).to("cuda") # pyright: ignore[reportAttributeAccessIssue]
        _, embedding = self.model.encode_image(inputs)
        embedding = embedding.view(inputs.shape[2] // self.patch_size, inputs.shape[3] // self.patch_size, -1)

        return embedding.to(device="cpu", dtype=torch.bfloat16)
