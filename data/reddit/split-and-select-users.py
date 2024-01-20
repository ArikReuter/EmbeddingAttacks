import re
import numpy as np

"""
Arguably everything here could be accomplished by bash script
However, we want to keep the lists of all authors and all sensitive authors
"""

all_authors_file = "all-users.txt"
sensitive_authors_file = "sensitive-users.txt"
control_file = "control_users_sample_patterns.txt"
trial_file = "sensitive_users_sample_patterns.txt"
sample_file = "users_sample_patterns.txt"
sample_size = 1000

def read_authors(filename):

    with open(filename) as f:
        authors = f.read().splitlines()
        authors = set(authors).difference({'"author":"[deleted]"'})
    return authors

def write_patterns(authors, filename):

    with open(filename, "w") as f:
        f.writelines([f'{author}\n' for author in authors])


all_authors = read_authors(all_authors_file)
sensitive_authors = read_authors(sensitive_authors_file)

control_group = list(all_authors.difference(sensitive_authors))
trial_group = list(all_authors.intersection(sensitive_authors))

rng = np.random.default_rng()
control_sample = rng.choice(control_group, sample_size)
trial_sample = rng.choice(trial_group, sample_size)

write_patterns(control_sample, control_file)
write_patterns(trial_sample, trial_file)
write_patterns(np.concatenate((control_sample + trial_sample)), sample_file)