from functools import lru_cache

from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


class EmbeddingService:
    def __init__(self) -> None:
        self.model = get_model()

    def embed_documents(self, chunks: list[str], batch_size: int = 32) -> list[list[float]]:
        embeddings = []

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            batch_embeddings = self.model.encode(
                batch,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            embeddings.extend(batch_embeddings.tolist())

        return embeddings

    def embed_query(self, question: str) -> list[float]:
        return self.model.encode(question, normalize_embeddings=True).tolist()
