from abc import ABC, abstractmethod
from tqdm import tqdm


class EmbeddingModel(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def embed_document(self, document: str) -> list[float]:
        pass

    @abstractmethod
    def embed_document_list(self, document_list: list[str]) -> list[list[float]]:
        for document in tqdm(document_list):
            self.embed_document(document)
