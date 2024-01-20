REGEX='"author":"[^"]+"'
unzstd -c RS_2019-04.zst | tee >(grep -oE $REGEX > all_authors.txt) | grep -Fi -f sensitive-subreddit-patterns.txt | grep -oE $REGEX > sensitive_authors.txt