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
    CALIBRATE_FLAG,
    BINARY,
    MAX_DESIRED_SAMPLES_PER_COMMENT,
)

# Classification
from autogluon.tabular import TabularPredictor

import logging

logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def get_embedding(x: str, df: pd.DataFrame):
    lookup = df.loc[x]

    if isinstance(lookup, pd.Series):
        return lookup.values
    else:
        # sample a random index from the lookup
        return lookup.sample(
            MAX_DESIRED_SAMPLES_PER_COMMENT
            if len(lookup) > MAX_DESIRED_SAMPLES_PER_COMMENT
            else len(lookup),
            replace=False,
        ).values


def remove_data_leakage(split_df: pd.DataFrame, leakage_values):
    if split_df is not None:
        for col in split_df.columns:
            if split_df.dtypes[col] == "object":
                # print(split_df[col].unique())
                if split_df[col].isin(leakage_values).any():
                    logger.info(f"Column {col} is a data leak. Remove it.")
                    split_df = split_df.drop(columns=[col])
                else:
                    logger.info(f"Column {col} is not a data leak. Keep it.")
    else:
        logger.info(f"No data passed.")

    return split_df


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
                f"reddit_{split}_chunks_"
                f"{NUM_DESIRED_AUTHORS}_AUTH_"
                f"{MAX_DESIRED_SAMPLES_PER_COMMENT}_MAXCOMB_"
                f"{NUM_DESIRED_SAMPLES}_SAMPLE_"
                f"{datetime}.pickle",
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
                f"reddit_{split}_embedding_"
                f"{NUM_DESIRED_AUTHORS}_AUTH_"
                f"{MAX_DESIRED_SAMPLES_PER_COMMENT}_MAXCOMB_"
                f"{NUM_DESIRED_SAMPLES}_SAMPLE_"
                f"{datetime}.pickle",
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
    logger.info(
        f"Successfully created train and test dataframes with {NUM_DESIRED_AUTHORS} authors."
    )

    # Filter the train and test dataframes for the NUM_DESIRED_AUTHORS with the most samples
    authors_with_most_samples = (
        train_df["author"].value_counts()[:NUM_DESIRED_AUTHORS].index
    )
    train_df = train_df[train_df["author"].isin(authors_with_most_samples)]
    test_df = test_df[test_df["author"].isin(authors_with_most_samples)]

    np.testing.assert_array_equal(
        train_df["author"].unique(), test_df["author"].unique()
    )

    # --------------
    # Binary Dataset
    # --------------
    if BINARY:  # else MULTICLASS
        if CREATE_BINARY_DATASET_FLAG:
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
                for author in tqdm(split_df["author"].unique()):
                    # temp = pd.DataFrame(
                    #     data=itertools.product(
                    #         split_df[split_df["author"] != author].values,
                    #         split_df[split_df["author"] == author].values,
                    #     ),
                    #     columns=["author_1", "author_2"],
                    # )
                    # temp = temp.sample(
                    #     NUM_DESIRED_SAMPLES
                    #     if len(temp) > NUM_DESIRED_SAMPLES
                    #     else len(temp),
                    #     replace=False,
                    # )
                    # different_author_pairs.extend(
                    #     [
                    #         np.concatenate(
                    #             (
                    #                 temp["author_1"].iloc[i],
                    #                 np.array([0]),
                    #                 temp["author_2"].iloc[i],
                    #             ),
                    #             axis=None,
                    #         ).tolist()
                    #         for i in range(len(temp))
                    #     ]
                    # )
                    temp = pd.DataFrame(
                        data=itertools.product(
                            split_df[split_df["author"] != author].index,
                            split_df[split_df["author"] == author].index,
                        ),
                        columns=["author_1", "author_2"],
                    )
                    temp = temp.sample(
                        NUM_DESIRED_SAMPLES
                        if len(temp) > NUM_DESIRED_SAMPLES
                        else len(temp),
                        replace=False,
                    )
                    # Replace the author_1 and author_2 columns with the corresponding embeddings.
                    temp = temp.reset_index(drop=True)
                    temp["author_1"] = temp["author_1"].apply(
                        lambda x: get_embedding(x, split_df)
                    )
                    temp["author_2"] = temp["author_2"].apply(
                        lambda x: get_embedding(x, split_df)
                    )

                    # Go through the rows of temp and if an entry in either column is not one
                    # dimensional, split it into multiple rows.
                    for i in range(len(temp)):
                        if (
                            len(temp["author_1"].iloc[i].shape) > 1
                            and len(temp["author_2"].iloc[i].shape) > 1
                        ):
                            for j in range(temp["author_1"].iloc[i].shape[0]):
                                for k in range(temp["author_2"].iloc[i].shape[0]):
                                    different_author_pairs.append(
                                        np.concatenate(
                                            (
                                                temp["author_1"].iloc[i][j],
                                                temp["author_2"].iloc[i][k],
                                            ),
                                            axis=None,
                                        )
                                    )
                        elif len(temp["author_1"].iloc[i].shape) > 1:
                            for j in range(temp["author_1"].iloc[i].shape[0]):
                                different_author_pairs.append(
                                    np.concatenate(
                                        (
                                            temp["author_1"].iloc[i][j],
                                            temp["author_2"].iloc[i],
                                        ),
                                        axis=None,
                                    )
                                )
                        elif len(temp["author_2"].iloc[i].shape) > 1:
                            for j in range(temp["author_2"].iloc[i].shape[0]):
                                different_author_pairs.append(
                                    np.concatenate(
                                        (
                                            temp["author_1"].iloc[i],
                                            temp["author_2"].iloc[i][j],
                                        ),
                                        axis=None,
                                    )
                                )
                        else:
                            different_author_pairs.append(
                                np.concatenate(
                                    (
                                        temp["author_1"].iloc[i],
                                        temp["author_2"].iloc[i],
                                    ),
                                    axis=None,
                                )
                            )

                    # different_author_pairs.extend(
                    #     [
                    #         np.concatenate(
                    #             (
                    #                 temp["author_1"].iloc[i],
                    #                 temp["author_2"].iloc[i],
                    #             ),
                    #             axis=None,
                    #         )
                    #         for i in range(len(temp))
                    #     ]
                    # )

                different_authors_df = pd.DataFrame(
                    data=different_author_pairs,
                )
                different_authors_df["label"] = 0
                same_and_diff_authors_data_dict[split][
                    "different_authors"
                ] = different_authors_df
                logger.info(
                    f"Successfully created different author dataset for {split}ing."
                )
                del different_author_pairs

                # Create pairs of samples from same author
                same_author_pairs = []
                for author in tqdm(split_df["author"].unique()):
                    # temp = pd.DataFrame(
                    #     data=itertools.product(
                    #         split_df[split_df["author"] == author].values,
                    #         split_df[split_df["author"] == author].values,
                    #     ),
                    #     columns=["author_1", "author_2"],
                    # )
                    # temp = temp.sample(
                    #     NUM_DESIRED_SAMPLES
                    #     if len(temp) > NUM_DESIRED_SAMPLES
                    #     else len(temp),
                    #     replace=False,
                    # )
                    # same_author_pairs.extend(
                    #     [
                    #         np.concatenate(
                    #             (
                    #                 temp["author_1"].iloc[i],
                    #                 np.array([0]),
                    #                 temp["author_2"].iloc[i],
                    #             ),
                    #             axis=None,
                    #         ).tolist()
                    #         for i in range(len(temp))
                    #     ]
                    # )
                    temp = pd.DataFrame(
                        data=itertools.product(
                            split_df[split_df["author"] == author].index,
                            split_df[split_df["author"] == author].index,
                        ),
                        columns=["author_1", "author_2"],
                    )
                    temp = temp.sample(
                        NUM_DESIRED_SAMPLES
                        if len(temp) > NUM_DESIRED_SAMPLES
                        else len(temp),
                        replace=False,
                    )
                    # Replace the author_1 and author_2 columns with the corresponding embeddings.
                    temp = temp.reset_index(drop=True)
                    temp["author_1"] = temp["author_1"].apply(
                        lambda x: get_embedding(x, split_df)
                    )
                    temp["author_2"] = temp["author_2"].apply(
                        lambda x: get_embedding(x, split_df)
                    )

                    # Go through the rows of temp and if an entry in either column is not one
                    # dimensional, split it into multiple rows.
                    for i in range(len(temp)):
                        if (
                            len(temp["author_1"].iloc[i].shape) > 1
                            and len(temp["author_2"].iloc[i].shape) > 1
                        ):
                            for j in range(temp["author_1"].iloc[i].shape[0]):
                                for k in range(temp["author_2"].iloc[i].shape[0]):
                                    same_author_pairs.append(
                                        np.concatenate(
                                            (
                                                temp["author_1"].iloc[i][j],
                                                temp["author_2"].iloc[i][k],
                                            ),
                                            axis=None,
                                        )
                                    )
                        elif len(temp["author_1"].iloc[i].shape) > 1:
                            for j in range(temp["author_1"].iloc[i].shape[0]):
                                same_author_pairs.append(
                                    np.concatenate(
                                        (
                                            temp["author_1"].iloc[i][j],
                                            temp["author_2"].iloc[i],
                                        ),
                                        axis=None,
                                    )
                                )
                        elif len(temp["author_2"].iloc[i].shape) > 1:
                            for j in range(temp["author_2"].iloc[i].shape[0]):
                                same_author_pairs.append(
                                    np.concatenate(
                                        (
                                            temp["author_1"].iloc[i],
                                            temp["author_2"].iloc[i][j],
                                        ),
                                        axis=None,
                                    )
                                )
                        else:
                            same_author_pairs.append(
                                np.concatenate(
                                    (
                                        temp["author_1"].iloc[i],
                                        temp["author_2"].iloc[i],
                                    ),
                                    axis=None,
                                )
                            )

                    # same_author_pairs.extend(
                    #     [
                    #         np.concatenate(
                    #             (
                    #                 temp["author_1"].iloc[i],
                    #                 temp["author_2"].iloc[i],
                    #             ),
                    #             axis=None,
                    #         )
                    #         for i in range(len(temp))
                    #     ]
                    # )

                same_authors_df = pd.DataFrame(
                    data=same_author_pairs,
                )
                same_authors_df["label"] = 1
                same_and_diff_authors_data_dict[split]["same_authors"] = same_authors_df
                logger.info(f"Successfully created same author dataset for {split}ing.")
                del same_author_pairs

                # -----------------------------------
                # Save the original created datasets.
                # -----------------------------------
                data_dump_path = os.path.join(
                    os.path.dirname(__file__),
                    "out",
                    "reddit_chunked",
                    f"reddit_{split}_pandas_"
                    f"{NUM_DESIRED_AUTHORS}_AUTH_"
                    f"{MAX_DESIRED_SAMPLES_PER_COMMENT}_MAXCOMB_"
                    f"{NUM_DESIRED_SAMPLES}_SAMPLE_"
                    f"df_{datetime}.pickle",
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
                        f"reddit_{split}_{tag}_"
                        f"{NUM_DESIRED_AUTHORS}_AUTH_"
                        f"{MAX_DESIRED_SAMPLES_PER_COMMENT}_MAXCOMB_"
                        f"{NUM_DESIRED_SAMPLES}_SAMPLE_"
                        f"df_{datetime}.pickle",
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
                    file
                    for file in files
                    if "different_authors" in file and split in file
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

        final_training_df = binary_train_df.copy()
        final_testing_df = binary_test_df.copy()
        final_validation_df = None

        final_training_df = remove_data_leakage(
            split_df=final_training_df, leakage_values=train_df["author"].unique()
        )
        final_testing_df = remove_data_leakage(
            split_df=final_testing_df, leakage_values=train_df["author"].unique()
        )
        final_validation_df = remove_data_leakage(
            split_df=final_validation_df, leakage_values=train_df["author"].unique()
        )

        logger.info(f"Successfully created binary classification dataset.")

    else:  # MULTICLASS
        final_training_df = train_df.copy()
        final_training_df = final_training_df.rename(columns={"author": "label"})
        final_testing_df = test_df.copy()
        final_testing_df = final_testing_df.rename(columns={"author": "label"})

        final_validation_df = pd.DataFrame()
        for author in final_testing_df["label"].unique():
            validation_samples = final_testing_df[final_testing_df["label"] == author]
            validation_samples = validation_samples[~validation_samples["is_sensitive"]]
            final_testing_df = final_testing_df.drop(validation_samples.index)
            final_validation_df = pd.concat([final_validation_df, validation_samples])

            # Assert at least one sample from each author is present.
            # assert (
            #     len(final_validation_df["label"].unique())
            #     ==
            #     len(final_testing_df["label"].unique())
            # )

        final_validation_df = final_validation_df.reset_index(drop=True)

    # --------------
    # Classification
    # --------------
    if FIT_FLAG:
        logger.info(f"Fitting predictor...")
        predictor = TabularPredictor(
            label="label",
            problem_type="binary" if BINARY else None,
            verbosity=3,
        )
        predictor.fit(
            train_data=final_training_df,
            tuning_data=final_validation_df,
            dynamic_stacking=True,
            # included_model_types=[
            #     "FASTAI",
            #     "NN_TORCH",
            # ],
            excluded_model_types=[
                "KNN",
            ],
            time_limit=60 * 60 * 1 / 2,
            presets=["medium_quality", "optimize_for_deployment"],
            hyperparameters="very_light",
            # ag_args_fit={"num_gpus": 1},
        )
    else:
        # Load the last saved Autogloun model from AutogluonModels folder.
        model_path = os.path.join(
            os.path.dirname(__file__), "AutogluonModels", MODEL_LOAD_NAME
        )
        predictor = TabularPredictor.load(model_path)
        logger.info(f"Successfully loaded predictor from {model_path}.")

    logger.info(f"Predictor summary: {nl}{predictor.fit_summary()}")
    if CALIBRATE_FLAG:
        if not BINARY:
            raise ValueError(f"Calibration is only possible for the binary problem.")

        calib_data = final_testing_df.sample(
            int(len(final_testing_df) / 2), replace=False
        )

        sub_testing_df = final_testing_df.drop(calib_data.index)

        calibrated_decision_boundary = predictor.calibrate_decision_threshold(
            metric="mcc", data=calib_data
        )

        predictor.set_decision_threshold(calibrated_decision_boundary)

        predictor_results = {
            "calib_train": predictor.evaluate(final_training_df, silent=False),
            "calib_test": predictor.evaluate(sub_testing_df, silent=False),
        }

    if not BINARY:
        predictor_results = {
            "train": predictor.evaluate(final_training_df, silent=False),
            "test": predictor.evaluate(final_testing_df, silent=False),
        }

        k = 2
        probabilities = predictor.predict_proba(final_testing_df)
        # Select the top 5 most probable authors for each sample using pandas
        top_k_authors = pd.DataFrame(
            data=pd.DataFrame(probabilities, index=final_testing_df.index).apply(
                lambda x: x.nlargest(k).index.tolist(), axis=1
            )
        )
        # Get the true author for each sample and compare it with the top 5 authors.
        top_k_authors["true_author"] = final_testing_df["label"]
        top_k_authors[f"is_in_top_{k}"] = top_k_authors.apply(
            lambda x: x["true_author"] in x[0], axis=1
        )
        top_k_accuracy = top_k_authors[f"is_in_top_{k}"].mean()
        logger.info(f"Top {k} accuracy: {top_k_accuracy}")

    logger.info(
        predictor.leaderboard(final_testing_df, silent=False)
    )

    logger.info(
        f"Predictor train performance on classification: {predictor_results['train']}"
    )
    logger.info(
        f"Predictor test performance on classification: {predictor_results['test']}"
    )