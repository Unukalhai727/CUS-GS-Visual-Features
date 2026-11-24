from abc import ABC, abstractmethod
from pathlib import Path
import torch


class VisualEncoder(ABC):
    model_id: str

    @abstractmethod
    def __call__(self, image_path: Path) -> torch.Tensor:
        """提取单张图片的特征，返回 [H, W, C] 的 CPU bfloat16 张量。"""
        pass
