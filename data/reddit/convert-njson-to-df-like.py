import json
import pandas as pd
from cleantext import clean
import nltk
from itertools import chain

filename = "control_comments.njson"
min_words = 5
min_sentences = 10

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
    longer_body_sentences = [sent for sent in body_sentences if len(nltk.word_tokenize(sent)) > min_words]
    cmt_filtered = [{
        "author": cmt["author"],
        "utterance": sent,
        "subreddit": cmt["subreddit"],
        "thread_id": link_id,
        "id": cmt["id"],
        "is_submission": is_submission} for sent in longer_body_sentences]
    return cmt_filtered

# TODO: filter utterances s.t. they have a minimum amount of words and split into sentences
# And filter authors s.t. they have a minimum number of sentences
# For each author split the utterances into train and test s.t. the different threads end up in same split
# Also, all the sensitive content should be in test

def read_to_df(filename):
    with open(filename) as f:
        comments_txt = f.read().splitlines()
    df = pd.DataFrame.from_records(chain.from_iterable(map(preprocess, comments_txt)))
    filtered_df = df.groupby("author").filter(lambda x: len(x) > min_sentences)
    return filtered_df

df = read_to_df("control_users_utterances.njson")
df.to_csv("control_users_utterances.csv")

df = read_to_df("sensitive_users_utterances.njson")
df.to_csv("sensitive_users_utterances.csv")
