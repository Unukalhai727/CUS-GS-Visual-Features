from pathlib import Path
import torch
from tqdm import tqdm


ENCODERS_CFG: dict[str, int] = {
    "clip": 80,
    "siglip": 85,
    "dinov2": 75,
    "llamav": 60,
}

def similarity_reduction(embedding: torch.Tensor, threshold: float, chunk_size=1000, desc="") -> torch.Tensor:
    with torch.no_grad():
        normalized_embedding = (embedding / embedding.norm(dim=1, keepdim=True))
        filtered_indices: list[int] = []
        used_mask = torch.zeros(embedding.shape[0], dtype=torch.bool, device=normalized_embedding.device)
        for i in tqdm(range(0, embedding.shape[0], chunk_size), desc, ncols=120):
            chunk = normalized_embedding[i: i + chunk_size]
            similarity_chunk = (chunk @ normalized_embedding.T)
            for j in range(chunk.shape[0]):
                if used_mask[i + j] == 0:
                    similar_indices = torch.nonzero(similarity_chunk[j] >= threshold).squeeze(1)
                    if not used_mask[similar_indices].any():
                        filtered_indices.append(i + j)
                        used_mask[similar_indices] = True
        return embedding[filtered_indices]

def run(name: str, scene: Path, threshold: int):
    dst = scene / "features" / name / f"mb{threshold}.pth"
    emb_dir = scene / "features" / name / "embeds"

    emb_list: list[torch.Tensor] = []
    for emb_file in emb_dir.iterdir():
        emb = torch.load(emb_file, map_location="cuda") # [H, W, C]
        emb = emb.view(-1, emb.shape[-1]) # [H, W, C] -> [H*W, C]
        emb_list.append(emb)

    embedding = torch.cat(emb_list, dim=0) # [N, C]
    reduced_embedding = similarity_reduction(embedding, threshold=threshold / 100, desc=f"{scene.name}::{name}")
    torch.save(reduced_embedding.to(device="cpu", dtype=torch.bfloat16), dst)

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-s", "--scene", type=Path, required=True, help="Scene name")
    parser.add_argument("-e", "--encoder", type=str, required=True, choices=ENCODERS_CFG, help="Encoder name")
    args = parser.parse_args()

    threshold = ENCODERS_CFG[args.encoder]
    run(args.encoder, args.scene, threshold)
