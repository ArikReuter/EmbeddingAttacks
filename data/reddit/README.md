# Workflow

Requirements:

- Linux
- Data downloaded from https://zenodo.org/records/3608135
- zstd

```bash
source extract_users.sh
python process-authors.py
source extract_sample_utterances.sh
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

Also a result is a curated small dataset `showcase.csv` with 5 users, each with an innocent and sensitive utterance.