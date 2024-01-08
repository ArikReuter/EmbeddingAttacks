import os
import pandas as pd


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
