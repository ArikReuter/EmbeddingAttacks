# EmbeddingAttacks

## Setup

```bash
Python-version: 3.10.11
```

## Data
Data can be downloaded from the following links: 

- https://paperswithcode.com/dataset/standardized-project-gutenberg-corpus
- https://zenodo.org/records/3608135

# Downloading the data

```bash
# 0. Download the data from https://zenodo.org/records/3608135
source extract_users.sh
python process-authors.py
source extract_sample_utterances.sh
python convert-njson-to-df-like.py
```

## Where to store the data
EmbeddingAttacks \
├── *data* \
│   ├── *gutenberg* \
│   │   ├── *SPGC_tokens_20180718* \
│   │   ├── *SPGC_metadata_20180718.csv* \
│   ├── *reddit* \
|   |   ├── *RS_2019-04.zst* \
|   |   ├── *RC_2019-04.zst* \
├── embeddings 

## Report draft
https://www.overleaf.com/5799997986nqctndwqgsjd#390f6b
