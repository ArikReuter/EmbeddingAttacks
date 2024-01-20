import json
import numpy as np
import pandas as pd
from cleantext.clean import clean
import nltk
from itertools import chain
import re
from tqdm import tqdm

filename = "utterances.njson"
min_words = 5
min_train_sentences = 50
min_test_sentences = 50

with open("sensitive-subreddit-patterns.txt") as f:
    patterns = f.read().splitlines()
    SENSITIVE_PATTERNS = [re.compile(pattern) for pattern in patterns]

def preprocess(cmt):
    cmt = json.loads(cmt)
    if "link_id" in cmt.keys():
        body = cmt["body"]
        link_id = cmt["link_id"]
        is_submission = False
    else:
        body = cmt["title"]
        if cmt["selftext"] is not None:
            body += ". " + cmt["selftext"]
        link_id = cmt["id"]
        is_submission = True

    body_cleaned = clean(body,
                      fix_unicode=True,               # fix various unicode errors
                      to_ascii=True,                  # transliterate to closest ASCII representation
                      lower=False,                     # lowercase text
                      # fully strip line breaks as opposed to only normalizing them
                      no_line_breaks=False,
                      no_urls=True,                  # replace all URLs with a special token
                      no_emails=True,                # replace all email addresses with a special token
                      no_phone_numbers=True,         # replace all phone numbers with a special token
                      no_numbers=False,               # replace all numbers with a special token
                      no_digits=False,                # replace all digits with a special token
                      no_currency_symbols=True,      # replace all currency symbols with a special token
                      no_punct=False,                 # remove punctuations
                      replace_with_punct="",          # instead of removing punctuations you may replace them
                      replace_with_url="<URL>",
                      replace_with_email="<EMAIL>",
                      replace_with_phone_number="<PHONE>",
                      replace_with_number="<NUMBER>",
                      replace_with_digit="0",
                      replace_with_currency_symbol="<CUR>",
                      lang="en") 
    body_sentences = nltk.sent_tokenize(body)
    out = {
        "author": cmt["author"],
        "utterance": body,
        "subreddit": cmt["subreddit"],
        "thread_id": link_id,
        "id": cmt["id"],
        "is_submission": is_submission,
        "n_sentences":len(body_sentences)}
    return out

rng = np.random.default_rng()

with open("sensitive_utterances.njson") as f:
    comments_txt = f.read().splitlines()
df = pd.DataFrame.from_records(map(preprocess, comments_txt))
df["is_sensitive"] = True

with open("not_sensitive_utterances.njson") as f:
    comments_txt = f.read().splitlines()
new_df = pd.DataFrame.from_records(map(preprocess, comments_txt))
new_df["is_sensitive"] = False

df = pd.concat((df, new_df)).reset_index(drop=True)

filtered_df = df.groupby("author").filter(lambda x: sum(x.loc[~x["is_sensitive"], "n_sentences"]) > min_train_sentences)
train_dfs = []
test_dfs = [] 

for author in tqdm(filtered_df["author"].unique()):
    author_df = filtered_df[filtered_df["author"] == author].reset_index(drop=True)
    not_sensitive_df = author_df.loc[~author_df["is_sensitive"]]
    n = int(min_train_sentences / not_sensitive_df["n_sentences"].mean())
    to_train = rng.choice(not_sensitive_df.index, n, p=not_sensitive_df["n_sentences"]/not_sensitive_df["n_sentences"].sum())
    new_train_df = author_df.iloc[to_train]
    new_test_df = author_df.drop(to_train)
    if new_test_df.shape[0] < min_train_sentences: continue
    train_dfs.append(new_train_df)
    test_dfs.append(new_test_df)

train_df = pd.concat(train_dfs)
test_df = pd.concat(test_dfs) 

train_df.to_csv("train_df.csv")
test_df.to_csv("test_df.csv")