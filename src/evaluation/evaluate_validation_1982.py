from __future__ import annotations

import argparse
import json
import re
import string
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DATASET_ID = "deepvk/GQA-ru"

SAMPLE_PATH = Path("data/splits/gqa_train_2000_sample.json")
ADAPTER_DIR = Path("outputs/train_2000_completion_only")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["base", "adapter"], required=True)
    return parser.parse_args()


def normalize_exact_match(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"[«»„“”–—…]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compute_group_accuracy(
    records: list[dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    grouped = defaultdict(lambda: {"correct": 0, "total": 0})

    for record in records:
        key = str(record[field])
        grouped[key]["total"] += 1
        grouped[key]["correct"] += int(record["correct"])

    return {
        key: {
            "correct": values["correct"],
            "total": values["total"],
            "accuracy": values["correct"] / values["total"],
        }
        for key, values in sorted(grouped.items())
    }


def load_validation_instructions() -> list[dict[str, Any]]:
    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(f"Sample file not found: {SAMPLE_PATH}")

    sample_data = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    question_ids = sample_data["validation_question_ids"]

    print(
        f"Loading fixed validation sample of "
        f"{len(question_ids)} question IDs..."
    )

    dataset = load_dataset(
        DATASET_ID,
        "train_balanced_instructions",
        split="train",
    )
    row_by_id = {row["id"]: dict(row) for row in dataset}

    missing = [qid for qid in question_ids if qid not in row_by_id]
    if missing:
        raise RuntimeError(f"Missing question IDs: {missing[:10]}")

    return [row_by_id[qid] for qid in question_ids]


def load_images(
    instructions: list[dict[str, Any]],
) -> dict[str, Any]:
    target_image_ids = {row["imageId"] for row in instructions}
    print(f"Need {len(target_image_ids)} unique images.")

    images_stream = load_dataset(
        DATASET_ID,
        "train_balanced_images",
        split="train",
        streaming=True,
    )

    image_by_id: dict[str, Any] = {}

    for row in images_stream:
        image_id = row["id"]

        if image_id in target_image_ids:
            image_by_id[image_id] = row["image"]
            found = len(image_by_id)
            total = len(target_image_ids)

            if found == 1 or found % 100 == 0 or found == total:
                print(
                    f"\rFound images: {found}/{total}",
                    end="",
                    flush=True,
                )

            if found == total:
                break

    print()

    missing = target_image_ids - set(image_by_id)
    if missing:
        raise RuntimeError(f"Missing image IDs: {sorted(missing)[:10]}")

    return image_by_id


def load_model(
    mode: str,
) -> tuple[torch.nn.Module, AutoProcessor]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this evaluation.")

    print(f"Loading processor: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    print(f"Loading base model: {MODEL_ID}")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    if mode == "adapter":
        if not ADAPTER_DIR.exists():
            raise FileNotFoundError(
                f"Adapter directory not found: {ADAPTER_DIR}"
            )

        print(f"Loading adapter: {ADAPTER_DIR}")
        model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    else:
        model = base_model

    model.to("cuda")
    model.eval()

    return model, processor


def generate_answer(
    model: torch.nn.Module,
    processor: AutoProcessor,
    image: Any,
    question: str,
) -> str:
    prompt = f"{question}\nОтветь одним словом."

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )

    inputs = {
        key: value.to("cuda")
        if isinstance(value, torch.Tensor)
        else value
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            num_beams=1,
        )

    prompt_length = inputs["input_ids"].shape[1]
    generated_only = generated_ids[:, prompt_length:]

    return processor.batch_decode(
        generated_only,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def main() -> None:
    args = parse_args()

    output_dir = Path("results") / "validation_1982" / args.mode
    predictions_path = output_dir / "gqa_predictions.jsonl"
    summary_path = output_dir / "gqa_summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"GQA HELD-OUT VALIDATION EVALUATION | mode={args.mode}")
    print("=" * 80)

    instructions = load_validation_instructions()
    image_by_id = load_images(instructions)
    model, processor = load_model(args.mode)

    records: list[dict[str, Any]] = []
    started_at = time.perf_counter()

    with predictions_path.open("w", encoding="utf-8") as output_file:
        for index, row in enumerate(instructions, start=1):
            example_started_at = time.perf_counter()

            prediction = generate_answer(
                model=model,
                processor=processor,
                image=image_by_id[row["imageId"]],
                question=row["question"],
            )

            gold_normalized = normalize_exact_match(row["answer"])
            prediction_normalized = normalize_exact_match(prediction)
            is_correct = gold_normalized == prediction_normalized
            latency = time.perf_counter() - example_started_at

            record = {
                "id": row["id"],
                "image_id": row["imageId"],
                "question": row["question"],
                "gold": row["answer"],
                "prediction": prediction,
                "gold_normalized": gold_normalized,
                "prediction_normalized": prediction_normalized,
                "correct": is_correct,
                "detailed_type": row["types"]["detailed"],
                "semantic_type": row["types"]["semantic"],
                "structural_type": row["types"]["structural"],
                "latency_seconds": latency,
                "mode": args.mode,
                "model_id": MODEL_ID,
                "adapter_dir": (
                    str(ADAPTER_DIR)
                    if args.mode == "adapter"
                    else None
                ),
            }

            records.append(record)
            output_file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

            if index == 1 or index % 25 == 0 or index == len(instructions):
                correct_so_far = sum(
                    int(item["correct"]) for item in records
                )
                accuracy_so_far = correct_so_far / len(records)

                print(
                    f"[{index:>4}/{len(instructions)}] "
                    f"acc={accuracy_so_far:.4f} "
                    f"time={latency:.2f}s"
                )

    total_elapsed = time.perf_counter() - started_at
    total = len(records)
    correct = sum(int(record["correct"]) for record in records)
    accuracy = correct / total

    summary = {
        "mode": args.mode,
        "model_id": MODEL_ID,
        "adapter_dir": (
            str(ADAPTER_DIR)
            if args.mode == "adapter"
            else None
        ),
        "sample_path": str(SAMPLE_PATH),
        "examples": total,
        "correct": correct,
        "accuracy": accuracy,
        "total_time_seconds": total_elapsed,
        "mean_latency_seconds": total_elapsed / total,
        "by_detailed_type": compute_group_accuracy(records, "detailed_type"),
        "by_semantic_type": compute_group_accuracy(records, "semantic_type"),
        "by_structural_type": compute_group_accuracy(records, "structural_type"),
    }

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("GQA HELD-OUT VALIDATION EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Mode:          {args.mode}")
    print(f"Examples:      {total}")
    print(f"Correct:       {correct}")
    print(f"Accuracy:      {accuracy:.4f}")
    print(f"Total time:    {total_elapsed:.2f}s")
    print(f"Mean latency:  {total_elapsed / total:.2f}s/example")
    print(f"Predictions:   {predictions_path}")
    print(f"Summary:       {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
