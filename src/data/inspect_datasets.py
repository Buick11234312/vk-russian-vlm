from __future__ import annotations

from typing import Any

from datasets import (
    get_dataset_config_names,
    get_dataset_split_names,
    load_dataset,
)


DATASET_IDS = [
    "deepvk/GQA-ru",
    "deepvk/MMBench-ru",
]


def summarize_value(value: Any, max_text_length: int = 160) -> str:
    """Return a compact human-readable description of a dataset value."""

    if value is None:
        return "None"

    # PIL image-like object
    if hasattr(value, "size") and hasattr(value, "mode"):
        return (
            f"{type(value).__name__}"
            f"(size={value.size}, mode={value.mode})"
        )

    if isinstance(value, str):
        text = value.replace("\n", "\\n")
        if len(text) > max_text_length:
            text = text[:max_text_length] + "..."
        return repr(text)

    if isinstance(value, dict):
        keys = list(value.keys())
        return f"dict(keys={keys})"

    if isinstance(value, (list, tuple)):
        return (
            f"{type(value).__name__}"
            f"(len={len(value)}, preview={value[:3]})"
        )

    return repr(value)


def inspect_dataset(dataset_id: str) -> None:
    print("\n" + "=" * 88)
    print(f"DATASET: {dataset_id}")
    print("=" * 88)

    configs = get_dataset_config_names(dataset_id)
    print(f"Configs: {configs}")

    for config_name in configs:
        print("\n" + "-" * 88)
        print(f"Config: {config_name}")

        splits = get_dataset_split_names(
            dataset_id,
            config_name=config_name,
        )
        print(f"Splits: {splits}")

        for split_name in splits:
            print(f"\n  Split: {split_name}")

            dataset = load_dataset(
                dataset_id,
                name=config_name,
                split=split_name,
                streaming=True,
            )

            print(f"  Features: {dataset.features}")

            try:
                sample = next(iter(dataset))
            except StopIteration:
                print("  Dataset split is empty.")
                continue

            print("  First sample:")
            for key, value in sample.items():
                print(
                    f"    {key}: "
                    f"{summarize_value(value)} "
                    f"[type={type(value).__name__}]"
                )


def main() -> None:
    for dataset_id in DATASET_IDS:
        try:
            inspect_dataset(dataset_id)
        except Exception as exc:
            print("\n" + "!" * 88)
            print(f"FAILED: {dataset_id}")
            print(f"{type(exc).__name__}: {exc}")
            print("!" * 88)


if __name__ == "__main__":
    main()
