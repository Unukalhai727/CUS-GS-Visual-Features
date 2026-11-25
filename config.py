from dataclasses import dataclass
from typing import ClassVar
from encoder import VisualEncoder, CLIPEncoder, SigLIPEncoder, DINOv2Encoder, LlaMaVEncoder


@dataclass
class EncoderConfig:
    cls: type[VisualEncoder]
    threshold: float

@dataclass
class Config:
    encoders: ClassVar[dict[str, EncoderConfig]] = {
        "clip": EncoderConfig(cls=CLIPEncoder, threshold=0.8),
        "siglip": EncoderConfig(cls=SigLIPEncoder, threshold=0.85),
        "dinov2": EncoderConfig(cls=DINOv2Encoder, threshold=0.75),
        "llamav": EncoderConfig(cls=LlaMaVEncoder, threshold=0.6),
    }
