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