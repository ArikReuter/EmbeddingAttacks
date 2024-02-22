# EmbeddingAttacks

## Setup

```bash
Python-version: 3.10.11
```

## Raw datasets
Data can be downloaded from the following links: 

- https://paperswithcode.com/dataset/standardized-project-gutenberg-corpus
- https://zenodo.org/records/3608135

Save the files in following paths:

EmbeddingAttacks \
├── *data* \
│   ├── *gutenberg* \
│   │   ├── *SPGC_tokens_20180718* \
│   │   ├── *SPGC_metadata_20180718.csv* \
│   ├── *reddit* \
|   |   ├── *RS_2019-04.zst* \
|   |   ├── *RC_2019-04.zst* \
├── embeddings 

## Preprocessing the data

Requirements:

- Linux
- Data downloaded from https://zenodo.org/records/3608135
- zstd

```bash
cd ./data/reddit
source extract-users.sh
python split-and-select-users.py
source extract-sample-utterances.sh
python convert-njson-to-df-like.py
```

# Metadata

An *utterance* is any text produced by a user. It could be a *submission* (a new post) or
a *comment* to a submission.

Sensitivity is decided based on subreddit name and flag `nsfw`. The list of subreddit patterns 
is in `sensitive-subreddit-patterns.txt`.

The end result is `reddit_train_df.csv` and `reddit_test_df.csv`, uploaded at GDrive. Each contains
following columns:

- `author` - (txt) Username of author of utterance
- `utterance` - (txt) Text of submission or comment.
- `subreddit` - (txt) Name of subreddit (subforum) where the text was published.
- `thread_id` - (txt) ID of submission that started the thread / discussion.
- `id` - (txt) ID of utterance. If it was a submission - identical with `thread_id`.
- `is_submission` - (bool) Was the `utterance` a submission (TRUE) or a comment (FALSE)?
- `n_sentences` - (int) How many sentences does `utterance` have?
- `is_sensitive` - (bool) Is the utterance sensitive, as decided based on subreddit and nsfw flag.
