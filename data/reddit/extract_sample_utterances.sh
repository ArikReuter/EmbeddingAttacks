unzstd -c RC_2019-04.zst | grep -F -f control_users_sample_patterns.txt > control_users_utterances.njson
unzstd -c RC_2019-04.zst | grep -F -f sensitive_users_sample_patterns.txt > sensitive_users_utterances.njson
unzstd -c RS_2019-04.zst | grep -F -f control_users_sample_patterns.txt >> control_users_utterances.njson
unzstd -c RS_2019-04.zst | grep -F -f sensitive_users_sample_patterns.txt >> sensitive_users_utterances.njson