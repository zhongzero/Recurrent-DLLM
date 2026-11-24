"""Score one prediction per sanitized MBPP test using the benchmark's assertions."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("samples", type=Path)
    a = p.parse_args()
    import evaluate
    from datasets import load_dataset

    predictions = {}
    for line in a.samples.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        key = str(row["task_id"])
        if key in predictions:
            raise ValueError(f"Duplicate task {key}")
        predictions[key] = [row["solution"]]
    tasks = list(
        load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    )
    ids = [str(r["task_id"]) for r in tasks]
    if set(ids) != set(predictions):
        raise ValueError(
            "Provide exactly one solution for every sanitized MBPP test task."
        )
    references = [
        "\n".join(
            [
                *(r.get("test_imports") or []),
                r.get("test_setup_code") or "",
                *r["test_list"],
            ]
        )
        for r in tasks
    ]
    scores, _ = evaluate.load("code_eval").compute(
        references=references, predictions=[predictions[i] for i in ids], k=[1]
    )
    print(json.dumps(scores))


if __name__ == "__main__":
    main()
