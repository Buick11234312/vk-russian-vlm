from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from datasets import Image, load_dataset


GQA_DATASET_ID = "deepvk/GQA-ru"
MMBENCH_DATASET_ID = "deepvk/MMBench-ru"

SEED = 42
VALIDATION_FRACTION = 0.05

RESULTS_PATH = Path("results/data_audit.json")
SPLIT_PATH = Path("data/splits/gqa_image_split.json")


def top_counts(values: Iterable[Any], n: int = 20) -> list[dict[str, Any]]:
    counter = Counter(str(value) for value in values)

    return [
        {
            "value": value,
            "count": count,
        }
        for value, count in counter.most_common(n)
    ]


def count_exact_duplicates(
    rows: Iterable[tuple[str, str, str]],
) -> int:
    counter = Counter(rows)

    return sum(
        count - 1
        for count in counter.values()
        if count > 1
    )


def is_missing_option(value: Any) -> bool:
    if value is None:
        return True

    text = str(value).strip().lower()

    return text in {
        "",
        "nan",
        "none",
        "null",
    }


def create_image_level_split(
    image_ids: set[str],
    validation_fraction: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    sorted_ids = sorted(image_ids)

    rng = random.Random(seed)
    rng.shuffle(sorted_ids)

    n_validation = math.ceil(
        len(sorted_ids) * validation_fraction
    )

    validation_ids = sorted(
        sorted_ids[:n_validation]
    )

    train_ids = sorted(
        sorted_ids[n_validation:]
    )

    return train_ids, validation_ids


def audit_gqa() -> tuple[dict[str, Any], dict[str, Any]]:
    print("Loading GQA-ru train instructions...")

    train = load_dataset(
        GQA_DATASET_ID,
        "train_balanced_instructions",
        split="train",
    )

    print("Loading GQA-ru testdev instructions...")

    testdev = load_dataset(
        GQA_DATASET_ID,
        "testdev_balanced_instructions",
        split="testdev",
    )

    train_image_ids = set(train["imageId"])
    testdev_image_ids = set(testdev["imageId"])

    image_overlap = train_image_ids & testdev_image_ids

    train_question_ids = train["id"]
    testdev_question_ids = testdev["id"]

    train_duplicate_rows = count_exact_duplicates(
        zip(
            train["imageId"],
            train["question"],
            train["answer"],
        )
    )

    testdev_duplicate_rows = count_exact_duplicates(
        zip(
            testdev["imageId"],
            testdev["question"],
            testdev["answer"],
        )
    )

    train_image_split, validation_image_split = (
        create_image_level_split(
            image_ids=train_image_ids,
            validation_fraction=VALIDATION_FRACTION,
            seed=SEED,
        )
    )

    train_split_set = set(train_image_split)
    validation_split_set = set(validation_image_split)

    n_train_questions = sum(
        image_id in train_split_set
        for image_id in train["imageId"]
    )

    n_validation_questions = sum(
        image_id in validation_split_set
        for image_id in train["imageId"]
    )

    detailed_types = [
        row["types"]["detailed"]
        for row in train
    ]

    semantic_types = [
        row["types"]["semantic"]
        for row in train
    ]

    structural_types = [
        row["types"]["structural"]
        for row in train
    ]

    report = {
        "train": {
            "rows": len(train),
            "unique_question_ids": len(set(train_question_ids)),
            "unique_image_ids": len(train_image_ids),
            "exact_duplicate_rows": train_duplicate_rows,
            "top_answers": top_counts(
                train["answer"],
                n=20,
            ),
            "top_detailed_types": top_counts(
                detailed_types,
                n=20,
            ),
            "top_semantic_types": top_counts(
                semantic_types,
                n=20,
            ),
            "top_structural_types": top_counts(
                structural_types,
                n=20,
            ),
        },
        "testdev": {
            "rows": len(testdev),
            "unique_question_ids": len(set(testdev_question_ids)),
            "unique_image_ids": len(testdev_image_ids),
            "exact_duplicate_rows": testdev_duplicate_rows,
            "top_answers": top_counts(
                testdev["answer"],
                n=20,
            ),
        },
        "leakage_checks": {
            "train_testdev_image_overlap_count": len(
                image_overlap
            ),
            "train_testdev_question_id_overlap_count": len(
                set(train_question_ids)
                & set(testdev_question_ids)
            ),
        },
        "generated_split": {
            "seed": SEED,
            "validation_fraction": VALIDATION_FRACTION,
            "train_images": len(train_image_split),
            "validation_images": len(validation_image_split),
            "train_questions": n_train_questions,
            "validation_questions": n_validation_questions,
            "image_overlap_between_generated_splits": len(
                train_split_set & validation_split_set
            ),
        },
    }

    split = {
        "seed": SEED,
        "validation_fraction": VALIDATION_FRACTION,
        "train_image_ids": train_image_split,
        "validation_image_ids": validation_image_split,
    }

    return report, split


def audit_mmbench() -> dict[str, Any]:
    print("Loading MMBench-ru dev...")

    dataset = load_dataset(
        MMBENCH_DATASET_ID,
        split="dev",
    )

    # We do not need to decode images for metadata audit.
    dataset = dataset.cast_column(
        "image",
        Image(decode=False),
    )

    valid_letters = {"A", "B", "C", "D"}

    invalid_answers = [
        answer
        for answer in dataset["answer"]
        if str(answer).strip() not in valid_letters
    ]

    available_option_counts = []

    for row in dataset:
        options = [
            row["A"],
            row["B"],
            row["C"],
            row["D"],
        ]

        available = sum(
            not is_missing_option(option)
            for option in options
        )

        available_option_counts.append(available)

    return {
        "rows": len(dataset),
        "unique_indices": len(set(dataset["index"])),
        "invalid_answer_count": len(invalid_answers),
        "answer_distribution": top_counts(
            dataset["answer"],
            n=10,
        ),
        "available_options_distribution": top_counts(
            available_option_counts,
            n=10,
        ),
        "top_categories": top_counts(
            dataset["category"],
            n=30,
        ),
        "top_l2_categories": top_counts(
            dataset["l2-category"],
            n=30,
        ),
        "top_sources": top_counts(
            dataset["source"],
            n=30,
        ),
    }


def main() -> None:
    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SPLIT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    gqa_report, gqa_split = audit_gqa()
    mmbench_report = audit_mmbench()

    full_report = {
        "gqa_ru": gqa_report,
        "mmbench_ru": mmbench_report,
    }

    RESULTS_PATH.write_text(
        json.dumps(
            full_report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    SPLIT_PATH.write_text(
        json.dumps(
            gqa_split,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)

    print(
        f"Report saved to: {RESULTS_PATH}"
    )

    print(
        f"Split saved to: {SPLIT_PATH}"
    )

    print()
    print("Key GQA checks:")

    print(
        "  Train rows:",
        gqa_report["train"]["rows"],
    )

    print(
        "  Testdev rows:",
        gqa_report["testdev"]["rows"],
    )

    print(
        "  Train/testdev image overlap:",
        gqa_report["leakage_checks"][
            "train_testdev_image_overlap_count"
        ],
    )

    print(
        "  Generated train questions:",
        gqa_report["generated_split"][
            "train_questions"
        ],
    )

    print(
        "  Generated validation questions:",
        gqa_report["generated_split"][
            "validation_questions"
        ],
    )

    print(
        "  Generated split image overlap:",
        gqa_report["generated_split"][
            "image_overlap_between_generated_splits"
        ],
    )

    print()
    print("Key MMBench checks:")

    print(
        "  Rows:",
        mmbench_report["rows"],
    )

    print(
        "  Invalid answers:",
        mmbench_report["invalid_answer_count"],
    )

    print(
        "  Available options distribution:",
        mmbench_report[
            "available_options_distribution"
        ],
    )


if __name__ == "__main__":
    main()
