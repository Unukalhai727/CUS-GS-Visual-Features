import torch
from tqdm import tqdm


@torch.inference_mode()
def similarity_reduction(embedding: torch.Tensor, threshold: float, chunk_size=1000, desc="") -> torch.Tensor:
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
