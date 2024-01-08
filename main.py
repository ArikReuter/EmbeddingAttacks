from Embeddings.EmbeddingModelOpenAI import EmbeddingModelOpenAI
from data import data_utils

if __name__ == "__main__":
    data = data_utils.load_token_data()

    embeddings_model = EmbeddingModelOpenAI()
    corpus_embeddings = embeddings_model.embed_document_list(
        data["text"].str.join(" ")
    )
