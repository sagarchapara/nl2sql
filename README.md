# NL2SQL BIRD Dataset Utilities

This repository now includes helper scripts for downloading the [BIRD](https://bird-bench.github.io/) SQL dataset and verifying model predictions against the official references. The goal is to make it easy to obtain the training/evaluation data and to check whether a SQL query returns the same results as the gold annotation.

## Prerequisites

* Bash >= 4.0
* `curl` and `unzip`
* Python 3.9 or newer (the `sqlite3` module from the standard library is required)

## Downloading the BIRD dataset

Use the provided shell script to download and unpack the dataset:

```bash
./scripts/download_bird_dataset.sh            # downloads into data/bird by default
./scripts/download_bird_dataset.sh /tmp/bird  # optional custom destination
```

The script tries a small list of known mirrors. If all of them fail (for example due to a firewall), provide your own mirror via the `BIRD_URLS` environment variable:

```bash
BIRD_URLS="https://example.org/bird.zip" ./scripts/download_bird_dataset.sh
```

The archive is extracted without removing any of the original directory names. After extraction you should have a layout similar to:

```
data/bird/
├── BIRD/               # (depending on the archive the root folder name may vary)
│   ├── databases/
│   │   ├── airline/airline.sqlite
│   │   └── ...
│   ├── dev/bird_dev.json
│   ├── train/bird_train.json
│   └── test/bird_test.json
└── ... (other metadata files included in the release)
```

> **Note:** The script downloads the archive to a temporary directory and removes it after a successful extraction. If you want to keep a copy of the zip (for example to cache it across machines) you can set the `BIRD_URLS` variable to a local file URL before running the script.

## Verifying a single SQL query

The `scripts/bird_sql_evaluator.py` utility lets you check whether a prediction matches the official answer by executing both queries on the associated SQLite database and comparing the result sets. The command below locates an example by its `question_id` and prints the diff:

```bash
python scripts/bird_sql_evaluator.py \
  --dataset-root data/bird/BIRD \
  --split dev \
  verify \
  --question-id DEV_0001 \
  --prediction "SELECT COUNT(*) FROM table_name;"
```

Key options:

* `--dataset-root` – directory containing the extracted dataset. Point this to the folder that includes `databases/` and the JSON split files.
* `--split` – dataset split to search (`train`, `dev`, or `test`).
* `--question-id` or `--question` – locate examples either by identifier or the exact natural language question.
* `--prediction`/`--prediction-file` – provide the SQL to validate inline or from a file.
* `--show-results` – always print the executed result sets (otherwise they are shown only on mismatch).

The script exits with status code `0` when the prediction matches the reference results, `2` when the results differ, and raises an error for malformed inputs.

## Evaluating a file of predictions

To score many predictions at once, pass a JSON or JSONL file to the `evaluate` sub-command. Each entry should contain a question identifier and the predicted SQL. Examples of supported formats:

* JSON lines (`.jsonl`): one object per line with keys `question_id`/`id` and `sql`/`prediction`/`query`.
* JSON list: `[ {"question_id": "DEV_0001", "prediction": "..."}, ... ]`
* JSON object mapping identifiers to SQL strings.

Example usage:

```bash
python scripts/bird_sql_evaluator.py \
  --dataset-root data/bird/BIRD \
  --split dev \
  evaluate \
  --predictions outputs/dev_predictions.jsonl
```

The script reports exact-match accuracy and warns about any question IDs that were not found in the dataset split.

## Troubleshooting

* **Download mirrors fail** – Supply a different mirror via `BIRD_URLS`. The official download page lists up-to-date links.
* **Missing `sqlite3`** – Ensure you are using the system Python (3.9+) which includes the SQLite bindings. Some minimal Python builds omit SQLite support.
* **`question_id` not found** – Double-check the dataset split and the identifier in your prediction file. You can search for examples manually in the JSON files under the dataset root.

Feel free to extend the scripts as your workflow evolves (e.g., add metrics, integrate with your training loop, etc.).
