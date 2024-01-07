from langchain.embeddings import HuggingFaceEmbeddings
from EmbeddingModel import EmbeddingModel

class EmbeddingModelHuggingface(EmbeddingModel):

    """
    Embedding model following the OpenAI API.
    """
    
    def __init__(self,
                 model_name:str = 'sentence-transformers/all-MiniLM-L6-v2',
                 device = "cpu"):
        """
        Args:
            model_name (str): name of the model to use
            device (str): device to use for the model
        """

        self.model_id = model_name
        self.device = device
    
        model_kwargs = {'device': device}
        self.embeddings_mod = HuggingFaceEmbeddings(
            model_name = model_name,
            model_kwargs=model_kwargs
        )


    def embed_document(self, document: str) -> list[float]:
        """
        Embeds a document
        Args:
            document (str): document to embed

        """
        return self.embeddings_mod.embed_query(document)
        

    def embed_document_list(self, documents: list[str]) -> list[list[float]]:
        """
        Embeds a list of documents

        Args:
            documents (list[str]): documents to embed

        """
        documents = [str(document) for document in documents]
        return self.embeddings_mod.embed_documents(documents)