from pathlib import Path
import torch
from tqdm import tqdm

from visual_encoder import VisualEncoder
from clip import CLIPEncoder
from siglip import SigLIPEncoder
from dinov2 import DINOv2Encoder
from llamav import LlaMaVEncoder

ENCODERS: dict[str, type[VisualEncoder]] = {
    "clip": CLIPEncoder,
    "siglip": SigLIPEncoder,
    "dinov2": DINOv2Encoder,
    "llamav": LlaMaVEncoder,
}

def create(name: str) -> VisualEncoder:
    return ENCODERS[name]()

def run(name: str, scene: Path, encoder: VisualEncoder):
    dst = scene / "llms" / name / "embeds"
    dst.mkdir(parents=True, exist_ok=True)
    for img_path in tqdm(sorted((scene / "images").iterdir()), desc=f"{scene}-{name}"):
        emb = encoder(img_path)
        torch.save(emb, dst / f"{img_path.stem}.pth")

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-s", "--scene", type=Path, required=True, help="Scene name")
    parser.add_argument("-e", "--encoder", type=str, required=True, choices=ENCODERS, help="Encoder name")
    args = parser.parse_args()

    encoder = create(args.encoder)
    run(args.encoder, args.scene, encoder)
