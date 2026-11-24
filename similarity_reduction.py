from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm

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

if __name__=="__main__":
    root = Path("/home/fcx/projects/Undefined-GS/data/m3")
    scenes = ["train"]
    llm_cfg = [
        # ("clip", 100, False),
        # ("siglip", 100, False),
        # ("dinov2", 75, False),
        ("seem_u", 100, False),
        # ("llama3", 100, False),
        # ("llamav", 60, True),
        # ("clip_ii", 100, False),
        # ("siglip_i", 100, False),
        # ("llamav_i", 60, True),
    ]

    for scene in scenes:
        for llm, threshold, is_downsample in llm_cfg:
            dst = root / scene / "llms" / llm / f"mb{threshold}.pth"
            # if dst.exists():
            #     print(f"Reduced embedding for {scene} - {llm} with threshold {threshold} already exists. Skipping.")
            #     continue
            print(f"")
            embed_dir = root / scene / "llms" / llm / "embeds"
            embeddings = []
            for emb_file in embed_dir.iterdir():
                embedding = torch.load(emb_file, map_location="cuda") # [H, W, C]
                if is_downsample:
                    embedding = F.interpolate(embedding.permute(2,0,1).unsqueeze(0), size=(embedding.shape[0]//2, embedding.shape[1]//2), mode='bilinear', align_corners=False).squeeze(0).permute(1,2,0)
                embedding = embedding.view(-1, embedding.shape[-1]) # [H, W, C] -> [H*W, C]
                embeddings.append(embedding)
            embeddings = torch.cat(embeddings, dim=0) # [N, C]
            reduced_embedding = similarity_reduction(embeddings, threshold=threshold / 100, desc=f"{scene}/{llm}/mb{threshold}")
            torch.save(reduced_embedding.to(device="cpu", dtype=torch.bfloat16), dst)
