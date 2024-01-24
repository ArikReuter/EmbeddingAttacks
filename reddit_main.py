import glob
import itertools
import random
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
from globals import EMBEDDING_FLAG, BINARY_FLAG

# Classification
from autogluon.tabular import TabularDataset, TabularPredictor

import logging

logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    datetime = time.strftime("%Y%m%d_%H%M%S")
    if not os.path.exists("out/reddit_chunked"):
        os.makedirs("out/reddit_chunked")

    if EMBEDDING_FLAG:
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
            file = [file for file in files if split in file and "embedding" in file][0]
            with open(
                os.path.join(os.path.dirname(__file__), "out", "reddit_chunked", file),
                "rb",
            ) as f:
                chunked_data[split] = pickle.load(f)

        logger.info(
            f"Successfully loaded:"
            f" {len(chunked_data['train'])} training samples & "
            f" {len(chunked_data['test'])} test samples."
        )

    # --------------
    # Binary Dataset
    # --------------
    if BINARY_FLAG:
        # Note: training dataset contains no samples of sensitive posts/comments.
        train_df = pd.DataFrame(
            index=[
                chunked_data["train"][i]["metadata"]["id"]
                for i in range(len(chunked_data["train"]))
            ],
            data=np.array(
                [
                    chunked_data["train"][i]["embedding"]
                    for i in range(len(chunked_data["train"]))
                ]
            ),
        )
        train_df["author"] = [
            chunked_data["train"][i]["metadata"]["author"]
            for i in range(len(chunked_data["train"]))
        ]

        test_df = pd.DataFrame(
            index=[
                chunked_data["test"][i]["metadata"]["id"]
                for i in range(len(chunked_data["test"]))
            ],
            data=np.array(
                [
                    chunked_data["test"][i]["embedding"]
                    for i in range(len(chunked_data["test"]))
                ]
            ),
        )
        test_df["author"] = [
            chunked_data["test"][i]["metadata"]["author"]
            for i in range(len(chunked_data["test"]))
        ]

        logger.info(f"Successfully created train and test dataframes.")

        # Create pairs of samples from different authors
        different_author_pairs = []
        for author in tqdm(train_df["author"].unique()):
            temp = pd.DataFrame(
                data=itertools.product(
                    train_df[train_df["author"] != author].values,
                    train_df[train_df["author"] == author].values,
                ),
                columns=["author_1", "author_2"],
            )
            temp = temp.sample(50 if len(temp) > 50 else len(temp))
            different_author_pairs.extend(
                [
                    np.concatenate(
                        (
                            temp["author_1"].iloc[i],
                            np.array([0]),
                            temp["author_2"].iloc[i],
                        ),
                        axis=None,
                    ).tolist()
                    for i in range(len(temp))
                ]
            )

        different_authors_df = pd.DataFrame(
            data=different_author_pairs,
        )
        different_authors_df["label"] = 0
        logger.info(f"Successfully created different author dataset.")
        del different_author_pairs

        # Create pairs of samples from same author
        same_author_pairs = []
        for author in tqdm(train_df["author"].unique()):
            temp = pd.DataFrame(
                data=itertools.product(
                    train_df[train_df["author"] == author].values,
                    train_df[train_df["author"] == author].values,
                ),
                columns=["author_1", "author_2"],
            )
            temp = temp.sample(50 if len(temp) > 50 else len(temp))
            same_author_pairs.extend(
                [
                    np.concatenate(
                        (
                            temp["author_1"].iloc[i],
                            np.array([0]),
                            temp["author_2"].iloc[i],
                        ),
                        axis=None,
                    ).tolist()
                    for i in range(len(temp))
                ]
            )
        same_authors_df = pd.DataFrame(
            data=same_author_pairs,
        )
        same_authors_df["label"] = 1
        logger.info(f"Successfully created same author dataset.")
        del same_author_pairs

        for split, dataset in zip(["train", "test"], [train_df, test_df]):
            data_dump_path = os.path.join(
                os.path.dirname(__file__),
                "out",
                "reddit_chunked",
                f"reddit_{split}_pandas_df_{datetime}.pickle",
            )
            with open(data_dump_path, "wb") as f:
                pickle.dump(dataset, f, protocol=3)
                logger.info(
                    f"Successfully saved pandas version of data to {data_dump_path}."
                )
        for tag, dataset in zip(
            ["same_authors", "different_authors"],
            [same_authors_df, different_authors_df],
        ):
            data_dump_path = os.path.join(
                os.path.dirname(__file__),
                "out",
                "reddit_chunked",
                f"reddit_train_{tag}_df_{datetime}.pickle",
            )
            with open(data_dump_path, "wb") as f:
                pickle.dump(dataset, f, protocol=3)
                logger.info(
                    f"Successfully saved pandas version of {tag} data to {data_dump_path}."
                )
    else:
        logger.info("Loading predefined binary dataset components...")
        files = [
            files[2]
            for files in os.walk(
                os.path.join(os.path.dirname(__file__), "out", "reddit_chunked")
            )
        ][0]

        same_authors_df = pd.read_pickle(
            os.path.join(
                os.path.dirname(__file__),
                "out",
                "reddit_chunked",
                [file for file in files if "same_authors" in file][0],
            )
        )
        different_authors_df = pd.read_pickle(
            os.path.join(
                os.path.dirname(__file__),
                "out",
                "reddit_chunked",
                [file for file in files if "different_authors" in file][0],
            )
        )

    # Remove the author column
    same_authors_df = same_authors_df.drop(columns=[3074])
    different_authors_df = different_authors_df.drop(columns=[3074])

    # Create binary classification dataset
    binary_train_df = pd.concat([same_authors_df, different_authors_df])
    logger.info(f"Successfully created binary classification dataset.")

    # --------------
    # Classification
    # --------------
    logger.info(f"Fitting predictor...")
    predictor = TabularPredictor(label="label", problem_type="binary")
    predictor.fit(
        binary_train_df,
        time_limit=60 * 10,
        presets="medium_quality",
        ag_args_fit={"num_gpus": 1},
    )
    # print the in-sample score
    logger.info(
        f"Predictor performance on classification: "
        f"{predictor.evaluate(binary_train_df, silent=False)}"
    )

    """
    Preliminary results:
    Evaluations on test data (AutoGluon internal):
    {
        "accuracy": 0.9591845493562232,
        "balanced_accuracy": 0.9591845493562232,
        "mcc": 0.918374786025882,
        "roc_auc": 0.9962794636114131,
        "f1": 0.9591126015735844,
        "precision": 0.9608062709966405,
        "recall": 0.9574248927038627
    }
    """
    pass
