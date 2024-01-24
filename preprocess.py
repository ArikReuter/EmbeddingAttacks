from Embeddings.EmbeddingModelOpenAI import EmbeddingModelOpenAI
from data import data_utils
import pickle
import time
import os


import logging

logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    embeddings_model = EmbeddingModelOpenAI()

    data = data_utils.load_token_data()
    data["text"] = (
        data["text"].apply(lambda x: " ".join(x)).tolist()
    )  # join tokens to text

    dict_chunk_list = []
    for doc_id, text, author in zip(data.index, data["text"], data["author"]):
        dict_chunk_list.extend(
            data_utils.create_chunk_dict_list(text, {"id": doc_id, "author": author})
        )

    # create folder if not exists
    if not os.path.exists("out/gutenberg_chunked"):
        os.makedirs("out/gutenberg_chunked")

    datetime = time.strftime("%Y%m%d_%H%M%S")
    data_dump_path = os.path.join(
        os.path.dirname(__file__),
        "out",
        "gutenberg_chunked",
        f"test01_{datetime}.pickle",
    )
    with open(data_dump_path, "wb") as f:
        pickle.dump(
            dict_chunk_list, f, protocol=3
        )  # use protocol 3 for compatibility with colab and python 3.6 ?
        logger.info(f"Successfully saved chunked data to {data_dump_path}.")

    chunk_list = [chunk_dict["text"] for chunk_dict in dict_chunk_list]
    corpus_embeddings = embeddings_model.embed_document_list(chunk_list)

    assert len(corpus_embeddings) == len(
        dict_chunk_list
    ), "Number of embeddings and number of chunks do not match."

    for i, chunk_dict in enumerate(dict_chunk_list):
        chunk_dict["embedding"] = corpus_embeddings[i]

    data_dump_path = os.path.join(
        os.path.dirname(__file__),
        "out",
        "gutenberg_chunked",
        f"test01_embeddings_{datetime}.pickle",
    )
    with open(data_dump_path, "wb") as f:
        pickle.dump(dict_chunk_list, f, protocol=3)
        logger.info(f"Successfully saved embedding data to {data_dump_path}.")

    logger.info("Done")
