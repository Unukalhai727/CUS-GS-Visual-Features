from pathlib import Path
import torch
from tqdm import tqdm

from algorithm import SimilarityReducer
from config import Config


def main(name: str, scene: Path):
    embedding_dir = scene / "features" / name / "embeds"
    embedding_dir.mkdir(parents=True, exist_ok=True)

    encoder = Config.encoders[name].cls()
    threshold = Config.encoders[name].threshold
    reducer = SimilarityReducer(threshold=threshold)

    for image_path in tqdm(sorted((scene / "images").iterdir()), desc=f"{scene.name}::{name}", ncols=120):
        embedding = encoder(image_path)
        reducer(embedding)
        torch.save(embedding, embedding_dir / f"{image_path.stem}.pth")

    torch.save(reducer.result, embedding_dir.parent / f"mb{int(threshold * 100)}.pth")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-s", "--scene", type=Path, required=True, help="Scene name")
    parser.add_argument("-e", "--encoder", type=str, required=True, choices=Config.encoders, help="Encoder name")
    args = parser.parse_args()

    main(args.encoder, args.scene)
