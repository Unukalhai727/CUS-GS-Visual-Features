import torch
from torch.nn import functional as F


class SimilarityReducer:
    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self.memory: torch.Tensor | None = None
        self.normalized_memory: torch.Tensor | None = None

    @torch.inference_mode()
    def __call__(self, embedding: torch.Tensor) -> None:
        embedding = embedding.view(-1, embedding.shape[-1]).cuda()
        normalized_embedding = F.normalize(embedding.float(), dim=1)

        if self.normalized_memory is not None:
            covered = normalized_embedding.new_full((normalized_embedding.shape[0],), -torch.inf)
            for memory_chunk in self.normalized_memory.split(8192):
                similarity = normalized_embedding @ memory_chunk.T
                covered = torch.maximum(covered, similarity.amax(dim=1))
            keep = covered < self.threshold
            embedding = embedding[keep]
            normalized_embedding = normalized_embedding[keep]

        selected_embeddings: list[torch.Tensor] = []
        selected_normalized: list[torch.Tensor] = []
        while normalized_embedding.shape[0] > 0:
            selected_embeddings.append(embedding[0].clone())
            selected_normalized.append(normalized_embedding[0].clone())

            representative = normalized_embedding[0]
            keep = normalized_embedding @ representative < self.threshold
            embedding = embedding[keep]
            normalized_embedding = normalized_embedding[keep]

        if selected_embeddings:
            memory = torch.stack(selected_embeddings)
            normalized_memory = torch.stack(selected_normalized)
            if self.memory is None:
                self.memory = memory
                self.normalized_memory = normalized_memory
            else:
                assert self.normalized_memory is not None
                self.memory = torch.cat((self.memory, memory), dim=0)
                self.normalized_memory = torch.cat((self.normalized_memory, normalized_memory), dim=0)

    @property
    def result(self) -> torch.Tensor:
        assert self.memory is not None
        return self.memory.cpu()
