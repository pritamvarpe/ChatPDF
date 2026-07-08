import re

import chromadb
from chromadb.errors import ChromaError


class VectorDBService:
    def __init__(self, persist_directory: str) -> None:
        self.client = chromadb.PersistentClient(path=persist_directory)

    def _collection_name(self, user_id: str) -> str:
        safe_user_id = re.sub(r"[^a-zA-Z0-9_-]", "_", user_id)
        return f"user_{safe_user_id}"[:63]

    def get_user_collection(self, user_id: str):
        return self.client.get_or_create_collection(name=self._collection_name(user_id))

    def reset_user_collection(self, user_id: str) -> None:
        collection_name = self._collection_name(user_id)
        try:
            self.client.delete_collection(collection_name)
        except (ChromaError, ValueError):
            pass
        self.client.get_or_create_collection(name=collection_name)

    def add_document_chunks(
        self,
        user_id: str,
        document_id: str,
        chunks: list[str],
        embeddings: list[list[float]],
        source_name: str,
        batch_size: int = 64,
    ) -> None:
        collection = self.get_user_collection(user_id)

        for start in range(0, len(chunks), batch_size):
            end = start + batch_size
            batch_chunks = chunks[start:end]
            batch_embeddings = embeddings[start:end]
            ids = [f"{document_id}_{index}" for index in range(start, start + len(batch_chunks))]
            metadatas = [
                {"document_id": document_id, "chunk_id": index, "source": source_name}
                for index in range(start, start + len(batch_chunks))
            ]

            collection.add(
                ids=ids,
                documents=batch_chunks,
                embeddings=batch_embeddings,
                metadatas=metadatas,
            )

    def search(self, user_id: str, query_embedding: list[float], n_results: int = 4):
        collection = self.get_user_collection(user_id)
        return collection.query(query_embeddings=[query_embedding], n_results=n_results)
