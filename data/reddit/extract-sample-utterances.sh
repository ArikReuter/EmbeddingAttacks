unzstd -c RC_2019-04.zst | grep -Fi -f users_sample_patterns.txt > utterances.njson
unzstd -c RS_2019-04.zst | grep -Fi -f users_sample_patterns.txt >> utterances.njson
grep -Fi -f sensitive-subreddit-patterns.txt utterances.njson > sensitive_utterances.njson
grep -Fiv -f sensitive-subreddit-patterns.txt utterances.njson > not_sensitive_utterances.njson
