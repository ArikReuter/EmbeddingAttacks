REGEX='"author":"[^"]+"'
unzstd -c RS_2019-04.zst | tee >(grep -oE $REGEX > all-users.txt) | grep -Fi -f sensitive-subreddit-patterns.txt | grep -oE $REGEX > sensitive-users.txt