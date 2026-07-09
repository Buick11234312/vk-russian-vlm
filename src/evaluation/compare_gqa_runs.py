from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

BASE_PATH = Path("results/baseline/gqa_predictions.jsonl")
NEW_PATH = Path("results/completion_only/gqa_predictions.jsonl")


def load_jsonl(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows[row["id"]] = row
    return rows


def exact_mcnemar_p(base_only: int, new_only: int) -> float:
    n = base_only + new_only
    if n == 0:
        return 1.0

    k = min(base_only, new_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def main() -> None:
    base = load_jsonl(BASE_PATH)
    new = load_jsonl(NEW_PATH)

    if set(base) != set(new):
        raise RuntimeError("Prediction files contain different example IDs.")

    ids = list(base)

    both_correct = 0
    base_only = 0
    new_only = 0
    both_wrong = 0

    by_semantic = defaultdict(
        lambda: {"n": 0, "base_correct": 0, "new_correct": 0}
    )

    gains = []
    losses = []

    for qid in ids:
        b = base[qid]
        n = new[qid]

        b_ok = bool(b["correct"])
        n_ok = bool(n["correct"])

        if b_ok and n_ok:
            both_correct += 1
        elif b_ok and not n_ok:
            base_only += 1
            losses.append((b, n))
        elif not b_ok and n_ok:
            new_only += 1
            gains.append((b, n))
        else:
            both_wrong += 1

        semantic = str(n["semantic_type"])
        by_semantic[semantic]["n"] += 1
        by_semantic[semantic]["base_correct"] += int(b_ok)
        by_semantic[semantic]["new_correct"] += int(n_ok)

    total = len(ids)
    base_correct = both_correct + base_only
    new_correct = both_correct + new_only
    p_value = exact_mcnemar_p(base_only, new_only)

    print("=" * 80)
    print("PAIRED GQA COMPARISON")
    print("=" * 80)
    print(f"Examples:        {total}")
    print(f"Base correct:    {base_correct} ({base_correct / total:.4f})")
    print(f"New correct:     {new_correct} ({new_correct / total:.4f})")
    print(f"Delta:           {(new_correct - base_correct) / total:+.4f}")
    print()
    print(f"Both correct:    {both_correct}")
    print(f"Base only:       {base_only}")
    print(f"New only:        {new_only}")
    print(f"Both wrong:      {both_wrong}")
    print(f"McNemar exact p: {p_value:.6f}")

    print()
    print("BY SEMANTIC TYPE")
    print("-" * 80)
    for semantic, stats in sorted(
        by_semantic.items(),
        key=lambda item: item[1]["n"],
        reverse=True,
    ):
        n = stats["n"]
        base_acc = stats["base_correct"] / n
        new_acc = stats["new_correct"] / n
        print(
            f"{semantic:>10}  n={n:>3}  "
            f"base={base_acc:.3f}  "
            f"new={new_acc:.3f}  "
            f"delta={new_acc - base_acc:+.3f}"
        )

    print()
    print("FIRST 10 GAINS")
    print("-" * 80)
    for b, n in gains[:10]:
        print(
            f"Q: {n['question']}\n"
            f"Gold: {n['gold']!r}\n"
            f"Base: {b['prediction']!r}\n"
            f"New:  {n['prediction']!r}\n"
        )

    print("FIRST 10 LOSSES")
    print("-" * 80)
    for b, n in losses[:10]:
        print(
            f"Q: {n['question']}\n"
            f"Gold: {n['gold']!r}\n"
            f"Base: {b['prediction']!r}\n"
            f"New:  {n['prediction']!r}\n"
        )


if __name__ == "__main__":
    main()
