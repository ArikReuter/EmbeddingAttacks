# EmbeddingAttacks

This is the repository corresponding to the paper "Extracting Sensitive Information from Textual Embeddings".
Our work explores the predictive performance of machine learning when trained on embeddings generated by state-of-the-art (SOTA) embedding models, with an emphasis on the potential risks and consequences associated with the misuse of such predictive models. The classification ensemble employed in this study utilizes the cutting edge Autogluon library to predict whether two pieces of text share a common author. Notably, the training data used consists of non-sensitive data from Reddit, while the testing data comprises a carefully selected set of sensitive texts from the same platform. The embeddings, serving as a compact and information-rich representation of textual content, encapsulate not only the semantic meaning but also the various idiosyncrasies of individual authors. The conducted experiments showcase the ability of the model to discern authorship, and in so doing, extract subtle nuances in writing style from the training data. 

## Setup


```bash
Python-version: 3.10.11
```

## Data
Data can be downloaded from the following links: 

- https://paperswithcode.com/dataset/standardized-project-gutenberg-corpus
- https://zenodo.org/records/3608135

### Downloading and preprocessing the data

Requirements:

- Linux
- Data downloaded from https://zenodo.org/records/3608135
- zstd

```bash
source extract-users.sh
python split-and-select-users.py
source extract-sample-utterances.sh
python convert-njson-to-df-like.py
```

### Metadata

An *utterance* is any text produced by a user. It could be a *submission* (a new post) or
a *comment* to a submission.

Sensitivity is decided based on subreddit name and flag `nsfw`. The list of subreddit patterns 
is in `sensitive-subreddit-patterns.txt`.

The end result is `reddit_train_df.csv` and `reddit_test_df.csv`.

- `author` - (txt) Username of author of utterance
- `utterance` - (txt) Text of submission or comment.
- `subreddit` - (txt) Name of subreddit (subforum) where the text was published.
- `thread_id` - (txt) ID of submission that started the thread / discussion.
- `id` - (txt) ID of utterance. If it was a submission - identical with `thread_id`.
- `is_submission` - (bool) Was the `utterance` a submission (TRUE) or a comment (FALSE)?
- `n_sentences` - (int) How many sentences does `utterance` have?
- `is_sensitive` - (bool) Is the utterance sensitive, as decided based on subreddit and nsfw flag.

Also a result is a curated small dataset `showcase.csv` with 5 users, each with an innocent and sensitive utterance.


### Data Storing

EmbeddingAttacks \
├── *data* \
│   ├── *gutenberg* \
│   │   ├── *SPGC_tokens_20180718* \
│   │   ├── *SPGC_metadata_20180718.csv* \
│   ├── *reddit* \
|   |   ├── *RS_2019-04.zst* \
|   |   ├── *RC_2019-04.zst* \
├── embeddings 

## Reproducing the experiments

Given the files EmbeddingAttacks/data/reddit/reddit_test_df_v1_1.csv and EmbeddingAttacks/data/reddit/reddit_train_df_v1_1.csv, running the script reddit_main.py runs our entire experimental pipeline, including chunking and embedding of the texts and fitting the final prediction model. 
Several flags in globals.py can be used to change the parameters of the experiment. 

| Variable     | Description |
| -----------  | ----------- |
| CHUNKING_FLAG  | If True, use chunks from most recent previous run of the script.     |
| EMBEDDING_FLAG  | If True, use embeddings from most recent previous run of the script.      |
| BINARY      | If True, use a Binary classification task, otherwise multiclass classification |
| CREATE_BINARY_DATASET_FLAG      | If True, use a Binary classification task, otherwise multiclass classification |
| CALIBRATE_FLAG      | If True, calibrate the finally obtained model |
| NUM_DESIRED_SAMPLES      | Maximum number of samples per author |
| MAX_DESIRED_SAMPLES_PER_COMMENT      | Maximum number of samples of pairs per comment |
| NUM_DESIRED_AUTHORS      | Number of authors to use the data of |
| FIT_FLAG     | If True, fit a model, else just the data chunking and embedding is performer |
| MODEL_LOAD_NAME      | Name of a model to load. if "None", no model is loaded |
 




