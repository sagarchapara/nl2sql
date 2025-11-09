#!/usr/bin/env python3
"""Utilities for validating SQL predictions on the BIRD dataset."""
from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

SPLIT_CANDIDATES = {
    "train": [
        "train.json",
        "bird_train.json",
        "train/train.json",
        "train/bird_train.json",
    ],
    "dev": [
        "dev.json",
        "bird_dev.json",
        "validation.json",
        "dev/dev.json",
        "dev/bird_dev.json",
    ],
    "test": [
        "test.json",
        "bird_test.json",
        "test/test.json",
        "test/bird_test.json",
    ],
}

DATABASE_DIR_NAMES = ["databases", "database"]
DB_EXTENSIONS = [".sqlite", ".db", ".sqlite3"]


@dataclass
class BirdExample:
    """Representation of an example from the BIRD dataset."""

    record: Dict
    db_path: Path

    @property
    def question_id(self) -> str:
        return extract_question_id(self.record)

    @property
    def question(self) -> str:
        return extract_question(self.record)

    @property
    def gold_sql(self) -> str:
        return extract_gold_sql(self.record)

    @property
    def db_id(self) -> str:
        return extract_db_id(self.record)


def find_split_file(dataset_root: Path, split: str) -> Path:
    candidates = SPLIT_CANDIDATES.get(split.lower())
    if not candidates:
        raise ValueError(f"Unknown split: {split}")

    for candidate in candidates:
        path = dataset_root / candidate
        if path.exists():
            return path
        # Allow nested directories such as split/candidate_name
        path = dataset_root / split / Path(candidate).name
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Unable to locate metadata file for split '{split}'. Checked: {candidates}"
    )


def extract_question_id(record: Dict) -> str:
    for key in ("question_id", "id", "qid"):
        value = record.get(key)
        if value is not None:
            return str(value)
    raise KeyError("No question identifier found in record")


def extract_question(record: Dict) -> str:
    for key in ("question", "utterance", "instruction"):
        value = record.get(key)
        if value:
            return str(value)
    raise KeyError("No natural language question found in record")


def extract_gold_sql(record: Dict) -> str:
    for key in ("query", "sql", "SQL"):
        value = record.get(key)
        if value:
            return str(value)
    raise KeyError("No reference SQL found in record")


def extract_db_id(record: Dict) -> str:
    for key in ("db_id", "database_id", "db"):
        value = record.get(key)
        if value:
            return str(value)
    raise KeyError("No database identifier found in record")


def iter_records(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    if isinstance(data, dict):
        if "data" in data and isinstance(data["data"], list):
            data = data["data"]
        else:
            raise ValueError(f"Unexpected JSON structure in {path}")

    if not isinstance(data, list):
        raise ValueError(f"Expected a list of examples in {path}")

    for record in data:
        if isinstance(record, dict):
            yield record
        else:
            raise ValueError("Each example must be a JSON object")


def find_database_root(dataset_root: Path) -> Path:
    for name in DATABASE_DIR_NAMES:
        candidate = dataset_root / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Unable to locate database directory (looked for {DATABASE_DIR_NAMES}) under {dataset_root}"
    )


def find_database_file(databases_root: Path, db_id: str) -> Path:
    db_dir = databases_root / db_id
    if not db_dir.exists():
        raise FileNotFoundError(f"Database directory not found: {db_dir}")

    for ext in DB_EXTENSIONS:
        candidate = db_dir / f"{db_id}{ext}"
        if candidate.exists():
            return candidate

    # Fallback: use the first file with a known extension inside the directory
    for child in db_dir.iterdir():
        if child.suffix in DB_EXTENSIONS:
            return child

    raise FileNotFoundError(
        f"No SQLite database file found for db_id '{db_id}' inside {db_dir}"
    )


def load_examples_by_id(dataset_root: Path, split: str) -> Dict[str, BirdExample]:
    """Load examples for a split and index them by question identifier."""
    split_file = find_split_file(dataset_root, split)
    databases_root = find_database_root(dataset_root)

    examples: Dict[str, BirdExample] = {}
    for record in iter_records(split_file):
        db_id = extract_db_id(record)
        example = BirdExample(record=record, db_path=find_database_file(databases_root, db_id))
        examples[example.question_id] = example
    return examples


def iter_examples(dataset_root: Path, split: str) -> Iterable[BirdExample]:
    """Yield examples from a dataset split without materializing the full mapping."""
    split_file = find_split_file(dataset_root, split)
    databases_root = find_database_root(dataset_root)

    for record in iter_records(split_file):
        db_id = extract_db_id(record)
        example = BirdExample(record=record, db_path=find_database_file(databases_root, db_id))
        yield example


def run_sql(db_path: Path, sql: str) -> Tuple[Tuple[str, ...], List[Tuple]]:
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cursor = connection.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        description = cursor.description or []
        header = tuple(column[0] for column in description)
        normalized_rows = sorted(tuple(row) for row in rows)
        return header, normalized_rows
    finally:
        connection.close()


def compare_queries(db_path: Path, gold_sql: str, predicted_sql: str) -> Tuple[bool, Tuple[Tuple[str, ...], List[Tuple]], Tuple[Tuple[str, ...], List[Tuple]]]:
    gold_result = run_sql(db_path, gold_sql)
    predicted_result = run_sql(db_path, predicted_sql)
    is_equal = gold_result == predicted_result
    return is_equal, gold_result, predicted_result


def run_verify_command(args: argparse.Namespace) -> int:
    """Entry point for the ``verify`` sub-command."""
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    examples = list(iter_examples(dataset_root, args.split))

    target_example: BirdExample | None = None
    if args.question_id:
        for example in examples:
            if example.question_id == args.question_id:
                target_example = example
                break
    elif args.question:
        lowered = args.question.strip().lower()
        for example in examples:
            if example.question.strip().lower() == lowered:
                target_example = example
                break
    else:
        raise ValueError("Either --question-id or --question must be provided")

    if target_example is None:
        raise ValueError("Unable to locate the requested example in the dataset")

    prediction = args.prediction
    if args.prediction_file:
        prediction = Path(args.prediction_file).read_text(encoding="utf-8").strip()
    if not prediction:
        raise ValueError("A SQL prediction is required")

    matches, gold_result, pred_result = compare_queries(
        target_example.db_path, target_example.gold_sql, prediction
    )

    print(f"Question: {target_example.question}")
    print(f"Database: {target_example.db_id}")
    print(f"Gold SQL: {target_example.gold_sql}")
    print(f"Predicted SQL: {prediction}")
    print(f"Match: {'YES' if matches else 'NO'}")
    if args.show_results or not matches:
        print("\nGold result header:", gold_result[0])
        print("Gold rows:")
        for row in gold_result[1]:
            print("  ", row)

        print("\nPredicted result header:", pred_result[0])
        print("Predicted rows:")
        for row in pred_result[1]:
            print("  ", row)
    return 0 if matches else 2


def parse_prediction_entry(record: Dict) -> Tuple[str, str]:
    """Normalize a prediction record to ``(question_id, sql)``."""
    question_id = None
    for key in ("question_id", "id", "qid"):
        if key in record:
            question_id = str(record[key])
            break
    if question_id is None:
        raise KeyError("Prediction entry missing question identifier")

    sql = None
    for key in ("predicted_sql", "sql", "query", "prediction"):
        if key in record and record[key]:
            sql = str(record[key])
            break
    if sql is None:
        raise KeyError("Prediction entry missing SQL text")

    return question_id, sql


def load_predictions(path: Path) -> Dict[str, str]:
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Treat as JSON lines
        predictions: Dict[str, str] = {}
        for line in text.splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            qid, sql = parse_prediction_entry(record)
            predictions[qid] = sql
        return predictions

    if isinstance(data, dict):
        if all(isinstance(v, str) for v in data.values()):
            return {str(k): str(v) for k, v in data.items()}
        if "data" in data and isinstance(data["data"], list):
            data = data["data"]

    if isinstance(data, list):
        predictions: Dict[str, str] = {}
        for record in data:
            if not isinstance(record, dict):
                raise ValueError("Prediction entries must be JSON objects")
            qid, sql = parse_prediction_entry(record)
            predictions[qid] = sql
        return predictions

    raise ValueError("Unsupported prediction file format")


def run_evaluate_command(args: argparse.Namespace) -> int:
    """Entry point for the ``evaluate`` sub-command."""
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    predictions_path = Path(args.predictions)
    predictions = load_predictions(predictions_path)

    examples = load_examples_by_id(dataset_root, args.split)

    total = 0
    correct = 0
    missing = []
    for question_id, prediction in predictions.items():
        example = examples.get(question_id)
        if example is None:
            missing.append(question_id)
            continue
        matches, _, _ = compare_queries(example.db_path, example.gold_sql, prediction)
        total += 1
        if matches:
            correct += 1

    if total == 0:
        print("No overlapping predictions were found between the file and the dataset")
        if missing:
            print(f"Skipped {len(missing)} predictions with unknown question ids")
        return 1

    accuracy = correct / total
    print(f"Evaluated {total} predictions")
    print(f"Exact match accuracy: {accuracy:.4f} ({correct}/{total})")
    if missing:
        print(f"Warning: {len(missing)} predictions did not match any question id in the dataset")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate SQL predictions against the BIRD dataset")
    parser.add_argument(
        "--dataset-root",
        default="data/bird",
        help="Location of the extracted BIRD dataset",
    )
    parser.add_argument(
        "--split",
        default="dev",
        choices=["train", "dev", "test"],
        help="Dataset split to operate on",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    verify_parser = subparsers.add_parser(
        "verify", help="Check a single SQL prediction against the reference answer"
    )
    verify_parser.add_argument("--question-id", help="Question identifier to evaluate")
    verify_parser.add_argument(
        "--question", help="Exact natural language question text to search for"
    )
    verify_parser.add_argument(
        "--prediction",
        help="SQL prediction to validate (can also be supplied via --prediction-file)",
    )
    verify_parser.add_argument(
        "--prediction-file",
        help="Path to a file containing the SQL prediction",
    )
    verify_parser.add_argument(
        "--show-results",
        action="store_true",
        help="Display the executed result sets in addition to the match outcome",
    )
    verify_parser.set_defaults(func=run_verify_command)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a batch of predictions from a JSON/JSONL file",
    )
    evaluate_parser.add_argument(
        "--predictions",
        required=True,
        help="Path to the predictions file",
    )
    evaluate_parser.set_defaults(func=run_evaluate_command)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
