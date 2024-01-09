import os
import pandas as pd
import random


def load_metadata():
    metadata = pd.read_csv(
        os.path.join(
            os.path.abspath(""),
            "data",
            "gutenberg",
            "SPGC_metadata_20180718.csv"
        ),
        index_col="id"
    )

    metadata = metadata[
        (metadata["language"] == "['en']")
    ][["title", "author"]].dropna()

    return metadata


def load_token_data():
    metadata = load_metadata()
    metadata = metadata[:1000]

    # Get all files in the directory and subdirectories that end with .txt
    files = [
        os.path.join(root, name)
        for root, dirs, files in os.walk(
            os.path.join(
                os.path.abspath(""),
                "data",
                "gutenberg",
                "SPGC_tokens_20180718"
            )
        )
        for name in files
        if name.endswith(".txt")
    ]

    texts = pd.DataFrame(
        [
            [
                os.path.basename(file).split(".")[0],
                open(file, "r", encoding="utf8").read().split("\n")
            ] for file in files if os.path.basename(file).split(".")[0].replace(
                "_tokens", ""
            ) in metadata.index
        ],
        columns=["id", "text"]
    )

    texts["id"] = texts["id"].apply(lambda x: x.replace("_tokens", ""))
    texts = texts.set_index("id")
    texts = texts.join(
        other=metadata["author"],
        how="inner"
    )

    return texts


def split_into_chunks(text: str, max_chunksize:int = 300, min_chunksize:int = 10) -> list[str]:
    """
    Splits a text into chunks where each has a size that is randomly chosen between min_chunksize and max_chunksize
    
    Args:
        text str: The text to split
        max_chunksize (int, optional): The maximum chunksize. Defaults to 300.
        min_chunksize (int, optional): The minimum chunksize. Defaults to 10.

    Returns:
        list[list[str]]: The text split into chunks
    """

    text = text.split(" ")

    chunk_lens = []
    
    while sum(chunk_lens) < len(text):
        chunk_lens.append(random.randint(min_chunksize, max_chunksize))

    chunks = []

    for i in range(len(chunk_lens)):
        words_to_append = text[sum(chunk_lens[:i]):sum(chunk_lens[:i+1])]
        chunks.append(" ".join(words_to_append))

    return chunks


def create_chunk_dict_list(text: str, metadata: dict, max_chunksize: int = 300, min_chunksize: int = 10) -> list[dict]:
    """
    Creates a list of dicts with the keys "text" and "metadata" from a text. The text is split into chunks where each has a size that is randomly chosen between min_chunksize and max_chunksize
    
    Args:
        text (str): The text to split
        metadata (dict): The metadata for the text
        max_chunksize (int, optional): The maximum chunksize. Defaults to 300.
        min_chunksize (int, optional): The minimum chunksize. Defaults to 10.

    Returns:
        list[dict]: The text split into chunks with metadata
    """

    chunks = split_into_chunks(text, max_chunksize, min_chunksize)

    chunk_dicts = []

    for chunk in chunks:
        chunk_dicts.append({"text": chunk, "metadata": metadata})

    return chunk_dicts