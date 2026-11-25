from pathlib import Path
import torch
from tqdm import tqdm

from algorithm import similarity_reduction
from config import Config


def main(name: str, scene: Path):
    embedding_dir = scene / "features" / name / "embeds"
    embedding_dir.mkdir(parents=True, exist_ok=True)

    encoder = Config.encoders[name].cls()
    threshold = Config.encoders[name].threshold

    embedding_list: list[torch.Tensor] = []
    for image_path in tqdm(sorted((scene / "images").iterdir()), desc=f"{scene.name}::{name}", ncols=120):
        embedding = encoder(image_path)
        embedding_list.append(embedding.view(-1, embedding.shape[-1]))
        torch.save(embedding, embedding_dir / f"{image_path.stem}.pth")

    flattened_embedding = torch.cat(embedding_list, dim=0)
    del encoder, embedding, embedding_list
    torch.cuda.empty_cache()

    reduced_embedding = similarity_reduction(flattened_embedding.cuda(), threshold=threshold, desc=f"{scene.name}::{name}").cpu()
    torch.save(reduced_embedding, embedding_dir.parent / f"mb{int(threshold * 100)}.pth")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-s", "--scene", type=Path, required=True, help="Scene name")
    parser.add_argument("-e", "--encoder", type=str, required=True, choices=Config.encoders, help="Encoder name")
    args = parser.parse_args()

    main(args.encoder, args.scene)
