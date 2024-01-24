import glob
import time

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import pickle
import os

from tqdm import tqdm

# Data
from data import data_utils

# Embedding
from Embeddings.EmbeddingModelOpenAI import EmbeddingModelOpenAI

from globals import EMBEDDING_FLAG

import logging

logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    if EMBEDDING_FLAG:
        datetime = time.strftime("%Y%m%d_%H%M%S")
        if not os.path.exists("out/reddit_chunked"):
            os.makedirs("out/reddit_chunked")

        reddit_train = pd.read_csv(
            os.path.join(os.getcwd(), "data", "reddit", "reddit_train_df_v1_1.csv")
        )
        reddit_test = pd.read_csv(
            os.path.join(os.getcwd(), "data", "reddit", "reddit_test_df_v1_1.csv")
        )

        # Drop and samples with empty utterance
        reddit_train = reddit_train.dropna(subset=["utterance"])
        reddit_test = reddit_test.dropna(subset=["utterance"])

        # ---------
        # Chunking
        # ---------
        chunked_data = {}
        for split, data in zip(["train", "test"], [reddit_train, reddit_test]):
            logger.info(f"Chunking {split} data...")
            dict_chunk_list = []

            for sample in tqdm(data.itertuples()):
                dict_chunk_list.extend(
                    data_utils.create_chunk_dict_list(
                        text=sample.utterance,
                        metadata={
                            "id": sample.id,
                            "thread_id": sample.thread_id,
                            "author": sample.author,
                            "subreddit": sample.subreddit,
                            "is_submission": sample.is_submission,
                            "is_sensitive": sample.is_sensitive,
                            "n_sentences": sample.n_sentences,
                        },
                    )
                )

            chunked_data[split] = dict_chunk_list

            data_dump_path = os.path.join(
                os.path.dirname(__file__),
                "out",
                "reddit_chunked",
                f"reddit_{split}_chunks_{datetime}.pickle",
            )

            with open(data_dump_path, "wb") as f:
                pickle.dump(dict_chunk_list, f, protocol=3)
                logger.info(f"Successfully saved chunked data to {data_dump_path}.")

        logger.info(
            f"Successfully chunked:"
            f" {len(chunked_data['train'])} training samples & "
            f" {len(chunked_data['test'])} test samples."
        )

        # ---------
        # Embedding
        # ---------
        embeddings_model = EmbeddingModelOpenAI()
        for split, dict_chunk_list in chunked_data.items():
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
                "reddit_chunked",
                f"reddit_{split}_embeddings_{datetime}.pickle",
            )
            with open(data_dump_path, "wb") as f:
                pickle.dump(dict_chunk_list, f, protocol=3)
                logger.info(f"Successfully saved embedding data to {data_dump_path}.")

        logger.info(
            f"Successfully embedded:"
            f" {len(chunked_data['train'])} training samples & "
            f" {len(chunked_data['test'])} test samples."
        )
    else:
        logger.info("Loading predefined embeddings...")
        chunked_data = {}

        files = [
            files[2]
            for files in os.walk(
                os.path.join(os.path.dirname(__file__), "out", "reddit_chunked")
            )
        ][0]

        for split in ["train", "test"]:
            # Load chunked data
            file = [file for file in files if split in file and "chunks" in file][0]
            with open(
                os.path.join(os.path.dirname(__file__), "out", "reddit_chunked", file),
                "rb",
            ) as f:
                chunked_data[split] = pickle.load(f)

            file = [file for file in files if split in file and "embedding" in file][0]
            with open(
                os.path.join(os.path.dirname(__file__), "out", "reddit_chunked", file),
                "rb",
            ) as f:
                corpus_embeddings = pickle.load(f)

            for i, chunk_dict in enumerate(chunked_data[split]):
                chunk_dict["embedding"] = corpus_embeddings[i]

        logger.info(
            f"Successfully loaded:"
            f" {len(chunked_data['train'])} training samples & "
            f" {len(chunked_data['test'])} test samples."
        )

        pass
