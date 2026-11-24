from typing import override
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, MllamaModel
from .types import VisualEncoder


class LlaMaVEncoder(VisualEncoder):
    model_id = "Llama-3.2-11B-Vision-Instruct"
    prompt = [{
        "role": "user", 
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe this image:"}
        ]
    }]

    def __init__(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = MllamaModel.from_pretrained(self.model_id, device_map="auto")

    @override
    @torch.inference_mode()
    def __call__(self, image_path: Path) -> torch.Tensor:
        image = Image.open(image_path).convert('RGB').resize((1120, 1120))
        input_text = self.processor.apply_chat_template(self.prompt, add_generation_prompt=True)
        inputs = self.processor(image, input_text, return_tensors="pt").to("cuda")
        outputs = self.model.vision_model(
            pixel_values=inputs.pixel_values,
            aspect_ratio_ids=inputs.aspect_ratio_ids,
            aspect_ratio_mask=inputs.aspect_ratio_mask,
        ).last_hidden_state
        cross_attention_states = self.model.multi_modal_projector(outputs)
        img_ = cross_attention_states.view(-1, outputs.shape[3], self.model.hidden_size)[:, 1:, :].view(4, 40, 40, self.model.hidden_size)
        embedding = cross_attention_states.new_zeros((80, 80, self.model.hidden_size))
        embedding[:40, :40] = img_[0, :]
        embedding[:40, 40:] = img_[1, :]
        embedding[40:, :40] = img_[2, :]
        embedding[40:, 40:] = img_[3, :]
        return embedding.to(device="cpu", dtype=torch.bfloat16)
