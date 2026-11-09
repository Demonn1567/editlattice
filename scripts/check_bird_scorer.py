"""DEC-18 check: the streaming BIRD scorer reproduces every stored outcome.

Every stored BIRD run record holds the answer and the outcome that the earlier scorer (full fetch,
then set comparison) gave it. This rescores every distinct (task, answer) pair with the current
scorer and reports any disagreement.

Usage: python scripts/check_bird_scorer.py runs/gpu_pilot/gain_pilot*_bird/records.jsonl
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from editlattice.benchmarks.bird import BirdBenchmark  # noqa: E402


def main() -> None:
    benchmark = BirdBenchmark()
    tasks = {task["task_id"]: task for task in benchmark.load_all_tasks()}
    stored: dict[tuple[str, str], int] = {}
    conflicts = 0
    for path in sys.argv[1:]:
        for line in Path(path).read_text().splitlines():
            record = json.loads(line)
            if record.get("error") or not isinstance(record.get("answer"), str):
                continue
            key = (record["task_id"], record["answer"])
            if key in stored and stored[key] != record["outcome"]:
                conflicts += 1
            stored.setdefault(key, record["outcome"])
    disagree = []
    for (task_id, answer), outcome in stored.items():
        if benchmark.score(tasks[task_id], answer)[0] != outcome:
            disagree.append((task_id, answer[:80], outcome))
    print(json.dumps({"pairs": len(stored), "correct_pairs": sum(stored.values()),
                      "stored_conflicts": conflicts, "disagreements": len(disagree), "examples": disagree[:5]}))


if __name__ == "__main__":
    main()
