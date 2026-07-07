from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path(
    "results/baseline/gqa_predictions.jsonl"
)

DEFAULT_OUTPUT = Path(
    "results/baseline/gqa_error_analysis.json"
)

YES_NO_ANSWERS = {
    "да",
    "нет",
}

LATIN_RE = re.compile(r"[A-Za-z]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    return parser.parse_args()


def load_records(
    path: Path,
) -> list[dict[str, Any]]:
    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(
                    json.loads(line)
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line "
                    f"{line_number}: {exc}"
                ) from exc

    return records


def accuracy(
    records: list[dict[str, Any]],
) -> float:
    if not records:
        return 0.0

    correct = sum(
        int(record["correct"])
        for record in records
    )

    return correct / len(records)


def group_accuracy(
    records: list[dict[str, Any]],
    field: str,
) -> list[dict[str, Any]]:
    grouped: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for record in records:
        grouped[str(record[field])].append(
            record
        )

    result = []

    for name, group in grouped.items():
        correct = sum(
            int(record["correct"])
            for record in group
        )

        result.append(
            {
                "name": name,
                "total": len(group),
                "correct": correct,
                "accuracy": correct / len(group),
            }
        )

    result.sort(
        key=lambda item: (
            -item["total"],
            item["accuracy"],
            item["name"],
        )
    )

    return result


def count_words(text: str) -> int:
    return len(
        text.strip().split()
    )


def make_example(
    record: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": record["id"],
        "question": record["question"],
        "gold": record["gold"],
        "prediction": record["prediction"],
        "detailed_type": record["detailed_type"],
        "semantic_type": record["semantic_type"],
        "structural_type": record["structural_type"],
    }


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Predictions file not found: "
            f"{args.input}"
        )

    records = load_records(
        args.input
    )

    if not records:
        raise ValueError(
            "Predictions file is empty."
        )

    wrong = [
        record
        for record in records
        if not record["correct"]
    ]

    yes_no = [
        record
        for record in records
        if record["gold_normalized"]
        in YES_NO_ANSWERS
    ]

    yes_no_wrong = [
        record
        for record in yes_no
        if not record["correct"]
    ]

    polarity_flips = [
        record
        for record in yes_no_wrong
        if (
            record["prediction_normalized"]
            in YES_NO_ANSWERS
            and record["prediction_normalized"]
            != record["gold_normalized"]
        )
    ]

    verbose_predictions = [
        record
        for record in records
        if count_words(
            record["prediction"]
        ) > 3
    ]

    latin_predictions = [
        record
        for record in records
        if LATIN_RE.search(
            record["prediction"]
        )
    ]

    exact_gold_frequency = Counter(
        record["gold_normalized"]
        for record in records
    )

    gold_stats = []

    for gold_answer, total in (
        exact_gold_frequency.most_common()
    ):
        group = [
            record
            for record in records
            if (
                record["gold_normalized"]
                == gold_answer
            )
        ]

        gold_stats.append(
            {
                "gold": gold_answer,
                "total": total,
                "correct": sum(
                    int(record["correct"])
                    for record in group
                ),
                "accuracy": accuracy(group),
            }
        )

    report = {
        "overall": {
            "total": len(records),
            "correct": sum(
                int(record["correct"])
                for record in records
            ),
            "accuracy": accuracy(records),
            "wrong": len(wrong),
        },
        "yes_no": {
            "total": len(yes_no),
            "correct": sum(
                int(record["correct"])
                for record in yes_no
            ),
            "accuracy": accuracy(yes_no),
            "wrong": len(yes_no_wrong),
            "polarity_flips": len(
                polarity_flips
            ),
        },
        "format_behavior": {
            "predictions_longer_than_3_words": len(
                verbose_predictions
            ),
            "predictions_with_latin_letters": len(
                latin_predictions
            ),
        },
        "by_detailed_type": group_accuracy(
            records,
            "detailed_type",
        ),
        "by_semantic_type": group_accuracy(
            records,
            "semantic_type",
        ),
        "by_structural_type": group_accuracy(
            records,
            "structural_type",
        ),
        "gold_answer_stats": gold_stats,
        "example_errors": {
            "general": [
                make_example(record)
                for record in wrong[:20]
            ],
            "yes_no_polarity_flips": [
                make_example(record)
                for record in polarity_flips[:20]
            ],
            "verbose_predictions": [
                make_example(record)
                for record in verbose_predictions[:20]
            ],
            "latin_predictions": [
                make_example(record)
                for record in latin_predictions[:20]
            ],
        },
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("GQA BASELINE ERROR ANALYSIS")
    print("=" * 80)

    print(
        f"Overall accuracy:      "
        f"{report['overall']['accuracy']:.4f}"
    )

    print(
        f"Wrong answers:         "
        f"{report['overall']['wrong']}"
    )

    print()
    print("Yes/No subset:")

    print(
        f"  Examples:            "
        f"{report['yes_no']['total']}"
    )

    print(
        f"  Accuracy:            "
        f"{report['yes_no']['accuracy']:.4f}"
    )

    print(
        f"  Polarity flips:      "
        f"{report['yes_no']['polarity_flips']}"
    )

    print()
    print("Format behavior:")

    print(
        f"  >3 word predictions: "
        f"{report['format_behavior']['predictions_longer_than_3_words']}"
    )

    print(
        f"  Latin predictions:   "
        f"{report['format_behavior']['predictions_with_latin_letters']}"
    )

    print()
    print("Most common detailed types:")

    for item in report["by_detailed_type"][:10]:
        print(
            f"  {item['name']:<30} "
            f"n={item['total']:<4} "
            f"acc={item['accuracy']:.3f}"
        )

    print()
    print("Most common semantic types:")

    for item in report["by_semantic_type"][:10]:
        print(
            f"  {item['name']:<30} "
            f"n={item['total']:<4} "
            f"acc={item['accuracy']:.3f}"
        )

    print()
    print(
        f"Full report: {args.output}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
