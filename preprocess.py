from Embeddings.EmbeddingModelOpenAI import EmbeddingModelOpenAI
from data import data_utils
import pickle
import time

if __name__ == "__main__":
    data = data_utils.load_token_data()

    embeddings_model = EmbeddingModelOpenAI()
    
    data['text'] = data['text'].apply(lambda x: ' '.join(x)).tolist()  # join tokens to text

    dict_cunk_lis = []

    for doc_id, text, author in (zip(data.index, data["text"], data["author"])):
        dict_cunk_lis.extend(data_utils.create_chunk_dict_list(text, {"id": doc_id, "author": author}))
    
    datetime = time.strftime("%Y%m%d_%H%M%S")
    with open(f"out/gutenberg_chunked/test01_{datetime}.pickle", "wb") as f:
        pickle.dump(dict_cunk_lis, f, protocol=3)  #use protocol 3 for compatibility with colab and python 3.6 ?


    chunk_lis = [chunk_dict["text"] for chunk_dict in dict_cunk_lis]

    corpus_embeddings = embeddings_model.embed_document_list(chunk_lis)

    assert len(corpus_embeddings) == len(dict_cunk_lis), "Number of embeddings and number of chunks do not match"

    for i, chunk_dict in enumerate(dict_cunk_lis):
        chunk_dict["embedding"] = corpus_embeddings[i]

    with open(f"out/gutenberg_chunked/test01_embeddings_{datetime}.pickle", "wb") as f:
        pickle.dump(dict_cunk_lis, f, protocol=3)

    print("Done")