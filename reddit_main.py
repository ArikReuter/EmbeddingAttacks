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

from autogluon.tabular.models import TextPredictorModel
from autogluon.multimodal import MultiModalPredictor
from tqdm import tqdm

# Data
from data import data_utils

# Embedding
from Embeddings.EmbeddingModelOpenAI import EmbeddingModelOpenAI

from globals import (
    EMBEDDING_FLAG,
    CREATE_BINARY_DATASET_FLAG,
    FIT_FLAG,
    CHUNKING_FLAG,
    NUM_DESIRED_AUTHORS,
    NUM_DESIRED_SAMPLES,
    MODEL_LOAD_NAME,
)

# Classification
from autogluon.tabular import TabularPredictor

import logging

logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    nl = "\n"
    datetime = time.strftime("%Y%m%d%H%M%S")
    if not os.path.exists("out/reddit_chunked"):
        os.makedirs("out/reddit_chunked")

    # --------------------
    # Loading and Chunking
    # --------------------
    if CHUNKING_FLAG:
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
                f"reddit_{split}_chunks{NUM_DESIRED_AUTHORS}_AUTH_{NUM_DESIRED_SAMPLES}_SAMPLE_{datetime}.pickle",
            )

            with open(data_dump_path, "wb") as f:
                pickle.dump(dict_chunk_list, f, protocol=3)
                logger.info(f"Successfully saved chunked data to {data_dump_path}.")

        logger.info(
            f"Successfully chunked:"
            f" {len(chunked_data['train'])} training samples & "
            f" {len(chunked_data['test'])} test samples."
        )
    else:
        logger.info("Loading predefined chunked data...")
        chunked_data = {}

        files = [
            files[2]
            for files in os.walk(
                os.path.join(os.path.dirname(__file__), "out", "reddit_chunked")
            )
        ][0]

        for split in ["train", "test"]:
            file_list = [file for file in files if split in file and "chunks" in file]
            file = sorted(file_list, key=lambda x: x.split("_")[-1].split(".")[0])[-1]

            with open(
                os.path.join(os.path.dirname(__file__), "out", "reddit_chunked", file),
                "rb",
            ) as f:
                chunked_data[split] = pickle.load(f)
            logger.info(f"Successfully loaded file with name: {file}.")

        logger.info(
            f"Successfully loaded:"
            f" {len(chunked_data['train'])} training samples & "
            f" {len(chunked_data['test'])} test samples."
        )

    # Append the "SEP" token to the end of each chunk.
    for split, dict_chunk_list in chunked_data.items():
        for chunk_dict in dict_chunk_list:
            chunk_dict["text"] = chunk_dict["text"] + " [SEP]"
        logger.info(f"Successfully appended [SEP] token to {split} data.")

    # -----------------
    # Textual Embedding
    # -----------------
    if EMBEDDING_FLAG:
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
                f"reddit_{split}_embedd{NUM_DESIRED_AUTHORS}_AUTH_{NUM_DESIRED_SAMPLES}_SAMPLE_ings_{datetime}.pickle",
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
            file_list = [
                file for file in files if split in file and "embedding" in file
            ]
            file = sorted(file_list, key=lambda x: x.split("_")[-1].split(".")[0])[-1]

            with open(
                os.path.join(os.path.dirname(__file__), "out", "reddit_chunked", file),
                "rb",
            ) as f:
                chunked_data[split] = pickle.load(f)

            logger.info(f"Successfully loaded file with name: {file}.")

        logger.info(
            f"Successfully loaded:"
            f" {len(chunked_data['train'])} training samples & "
            f" {len(chunked_data['test'])} test samples."
        )

    # --------------
    # Binary Dataset
    # --------------
    if CREATE_BINARY_DATASET_FLAG:
        # --------------------------------
        # Create train and test dataframes
        # --------------------------------
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
        # train_df["author"] = [
        #     chunked_data["train"][i]["metadata"]["author"]
        #     for i in range(len(chunked_data["train"]))
        # ]
        for meta_data_key in chunked_data["train"][0]["metadata"].keys():
            train_df[meta_data_key] = [
                chunked_data["train"][i]["metadata"][meta_data_key]
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
        # test_df["author"] = [
        #     chunked_data["test"][i]["metadata"]["author"]
        #     for i in range(len(chunked_data["test"]))
        # ]
        for meta_data_key in chunked_data["test"][0]["metadata"].keys():
            test_df[meta_data_key] = [
                chunked_data["test"][i]["metadata"][meta_data_key]
                for i in range(len(chunked_data["test"]))
            ]
        logger.info(f"Successfully created train and test dataframes.")

        # -------------------------------------------------------------------
        # Create same and different author datasets for binary classification
        # -------------------------------------------------------------------
        same_and_diff_authors_data_dict = {
            "train": {"same_authors": None, "different_authors": None},
            "test": {"same_authors": None, "different_authors": None},
        }
        for split, split_df in zip(["train", "test"], [train_df, test_df]):
            # Create pairs of samples from different authors
            different_author_pairs = []
            for author in tqdm(split_df["author"].unique()[:NUM_DESIRED_AUTHORS]):
                temp = pd.DataFrame(
                    data=itertools.product(
                        split_df[split_df["author"] != author].values,
                        split_df[split_df["author"] == author].values,
                    ),
                    columns=["author_1", "author_2"],
                )
                temp = temp.sample(
                    NUM_DESIRED_SAMPLES
                    if len(temp) > NUM_DESIRED_SAMPLES
                    else len(temp),
                    replace=False,
                )
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
            same_and_diff_authors_data_dict[split][
                "different_authors"
            ] = different_authors_df
            logger.info(f"Successfully created different author dataset for {split}.")
            del different_author_pairs

            # Create pairs of samples from same author
            same_author_pairs = []
            for author in tqdm(split_df["author"].unique()[:NUM_DESIRED_AUTHORS]):
                temp = pd.DataFrame(
                    data=itertools.product(
                        split_df[split_df["author"] == author].values,
                        split_df[split_df["author"] == author].values,
                    ),
                    columns=["author_1", "author_2"],
                )
                temp = temp.sample(
                    NUM_DESIRED_SAMPLES
                    if len(temp) > NUM_DESIRED_SAMPLES
                    else len(temp),
                    replace=False,
                )
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
            same_and_diff_authors_data_dict[split]["same_authors"] = same_authors_df
            logger.info(f"Successfully created same author dataset for {split}.")
            del same_author_pairs

            # -----------------------------------
            # Save the original created datasets.
            # -----------------------------------
            data_dump_path = os.path.join(
                os.path.dirname(__file__),
                "out",
                "reddit_chunked",
                f"reddit_{split}_pandas{NUM_DESIRED_AUTHORS}_AUTH_{NUM_DESIRED_SAMPLES}_SAMPLE_df_{datetime}.pickle",
            )
            with open(data_dump_path, "wb") as f:
                pickle.dump(split_df, f, protocol=3)
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
                    f"reddit_{split}_{tag}_{NUM_DESIRED_AUTHORS}_AUTH_{NUM_DESIRED_SAMPLES}_SAMPLE_df_{datetime}.pickle",
                )
                with open(data_dump_path, "wb") as f:
                    pickle.dump(dataset, f, protocol=3)
                    logger.info(
                        f"Successfully saved pandas version of "
                        f"{split}-{tag} data to {data_dump_path}."
                    )
    else:
        logger.info("Loading predefined binary dataset components...")
        files = [
            files[2]
            for files in os.walk(
                os.path.join(os.path.dirname(__file__), "out", "reddit_chunked")
            )
        ][0]

        same_and_diff_authors_data_dict = {
            "train": {"same_authors": None, "different_authors": None},
            "test": {"same_authors": None, "different_authors": None},
        }
        for split in ["train", "test"]:
            all_file_names = [
                file for file in files if "same_authors" in file and split in file
            ]
            file_name = sorted(
                all_file_names, key=lambda x: x.split("_")[-1].split(".")[0]
            )[-1]
            same_and_diff_authors_data_dict[split]["same_authors"] = pd.read_pickle(
                os.path.join(
                    os.path.dirname(__file__),
                    "out",
                    "reddit_chunked",
                    file_name,
                )
            )
            logger.info(f"Successfully loaded file with name: {file_name}")

            all_file_names = [
                file for file in files if "different_authors" in file and split in file
            ]
            file_name = sorted(
                all_file_names, key=lambda x: x.split("_")[-1].split(".")[0]
            )[-1]
            same_and_diff_authors_data_dict[split][
                "different_authors"
            ] = pd.read_pickle(
                os.path.join(
                    os.path.dirname(__file__),
                    "out",
                    "reddit_chunked",
                    file_name,
                )
            )
            logger.info(f"Successfully loaded file with name: {file_name}")

    # Remove the author column to avoid data leakage.
    for split in ["train", "test"]:
        same_and_diff_authors_data_dict[split][
            "same_authors"
        ] = same_and_diff_authors_data_dict[split]["same_authors"].drop(
            columns=[
                # 3074, 1536
                3083,
                1538,
            ]
        )
        same_and_diff_authors_data_dict[split][
            "different_authors"
        ] = same_and_diff_authors_data_dict[split]["different_authors"].drop(
            columns=[
                # 3074, 1536
                3083,
                1538,
            ]
        )

    # Combine the "same_authors" and "different_authors" datasets.

    # combined_same_author_df = pd.concat(
    #     [
    #         same_and_diff_authors_data_dict["train"]["same_authors"],
    #         same_and_diff_authors_data_dict["test"]["same_authors"],
    #     ]
    # )
    # combined_different_author_df = pd.concat(
    #     [
    #         same_and_diff_authors_data_dict["train"]["different_authors"],
    #         same_and_diff_authors_data_dict["test"]["different_authors"],
    #     ]
    # )
    # binary_train_df = pd.concat(
    #     [
    #         combined_same_author_df.sample(frac=0.5, replace=False, random_state=42),
    #         combined_different_author_df.sample(
    #             frac=0.5, replace=False, random_state=42
    #         ),
    #     ]
    # )
    # combined_same_author_df = combined_same_author_df.drop(binary_train_df.index)
    # combined_different_author_df = combined_different_author_df.drop(
    #     binary_train_df.index
    # )
    # binary_test_df = pd.concat(
    #     [
    #         combined_same_author_df.sample(frac=1, replace=False, random_state=42),
    #         combined_different_author_df.sample(frac=1, replace=False, random_state=42),
    #     ]
    # )

    # Create binary classification dataset
    binary_train_df = pd.concat(
        [
            same_and_diff_authors_data_dict["train"]["same_authors"],
            same_and_diff_authors_data_dict["train"]["different_authors"],
        ]
    )
    binary_test_df = pd.concat(
        [
            same_and_diff_authors_data_dict["test"]["same_authors"],
            same_and_diff_authors_data_dict["test"]["different_authors"],
        ]
    )

    logger.info(f"Successfully created binary classification dataset.")

    # --------------
    # Classification
    # --------------
    if FIT_FLAG:
        logger.info(f"Fitting predictor...")
        predictor = TabularPredictor(label="label", problem_type="binary", verbosity=3)
        predictor.fit(
            binary_train_df,
            time_limit=60 * 60 * 1/8,
            presets="medium_quality",
            # ag_args_fit={"num_gpus": 1},
        )
    else:
        # Load the last saved Autogloun model from AutogluonModels folder.
        model_path = os.path.join(
            os.path.dirname(__file__), "AutogluonModels", MODEL_LOAD_NAME
        )
        predictor = TabularPredictor.load(model_path)
        logger.info(f"Successfully loaded predictor from {model_path}.")

    # print which models are used in the predictor.
    logger.info(f"Predictor contains the following models: {predictor.model_names()}")
    logger.info(f"Predictor summary: {nl}{predictor.fit_summary()}")
    predictor_results = {
        "train": predictor.evaluate(binary_train_df, silent=False),
        "test": predictor.evaluate(binary_test_df, silent=False),
    }
    logger.info(
        f"Predictor train performance on classification: {predictor_results['train']}"
    )
    logger.info(
        f"Predictor test performance on classification: {predictor_results['test']}"
    )

    pass
