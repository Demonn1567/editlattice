"""Create the frozen, stratified search/validation/held-out splits (D-007, D-008)."""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from editlattice.benchmarks.bird import BirdBenchmark  # noqa: E402
from editlattice.benchmarks.gsmhard import GsmHardBenchmark  # noqa: E402
from editlattice.benchmarks.livecodebench import LiveCodeBenchmark  # noqa: E402

SEED = 20261017
FRACTIONS = {"search": 0.4, "validation": 0.2, "heldout": 0.4}


def stratified_split(task_ids_by_stratum: dict[str, list[str]], seed: int) -> dict[str, list[str]]:
    """Split each stratum 40/20/40 with largest-remainder rounding; shuffle each split."""
    rng = random.Random(seed)
    splits: dict[str, list[str]] = {"search": [], "validation": [], "heldout": []}
    for stratum in sorted(task_ids_by_stratum):
        ids = sorted(task_ids_by_stratum[stratum])
        rng.shuffle(ids)
        n = len(ids)
        n_search = round(n * FRACTIONS["search"])
        n_validation = round(n * FRACTIONS["validation"])
        splits["search"].extend(ids[:n_search])
        splits["validation"].extend(ids[n_search:n_search + n_validation])
        splits["heldout"].extend(ids[n_search + n_validation:])
    for name in splits:
        splits[name].sort()
        rng.shuffle(splits[name])
    return splits


def bird_splits() -> dict:
    """BIRD: stratify by (db_id, difficulty); drop gold failures."""
    benchmark = BirdBenchmark()
    tasks = benchmark.load_all_tasks()
    benchmark.ensure_gold(tasks)
    excluded = []
    strata: dict[str, list[str]] = defaultdict(list)
    for task in tasks:
        if not benchmark.gold(task["task_id"])["ok"]:
            excluded.append(task["task_id"])
            continue
        strata[f"{task['db_id']}|{task['difficulty']}"].append(task["task_id"])
    splits = stratified_split(strata, SEED)
    return {"benchmark": "bird", "seed": SEED, "excluded": sorted(excluded), "splits": splits}


def lcb_splits() -> dict:
    """LiveCodeBench: stratify by difficulty."""
    tasks = LiveCodeBenchmark().load_all_tasks()
    strata: dict[str, list[str]] = defaultdict(list)
    for task in tasks:
        strata[task["difficulty"]].append(task["task_id"])
    splits = stratified_split(strata, SEED)
    return {"benchmark": "lcb", "seed": SEED, "excluded": [], "splits": splits}


def gsmhard_splits() -> dict:
    """GSM-Hard: stratify by whether the target is plausible, so label noise is spread evenly."""
    tasks = GsmHardBenchmark().load_all_tasks()
    strata: dict[str, list[str]] = defaultdict(list)
    for task in tasks:
        strata["plausible" if task["plausible"] else "implausible"].append(task["task_id"])
    splits = stratified_split(strata, SEED)
    return {"benchmark": "gsmhard", "seed": SEED, "excluded": [], "splits": splits}


def main() -> None:
    """Write configs/splits/{bird,lcb,gsmhard}.json, refusing to overwrite a different split."""
    out_dir = ROOT / "configs" / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    for make in (bird_splits, lcb_splits, gsmhard_splits):
        result = make()
        path = out_dir / f"{result['benchmark']}.json"
        text = json.dumps(result, indent=1)
        if path.exists() and path.read_text() != text:
            raise SystemExit(f"{path} exists and differs; splits are frozen")
        path.write_text(text)
        sizes = {name: len(ids) for name, ids in result["splits"].items()}
        everything = [i for ids in result["splits"].values() for i in ids]
        assert len(everything) == len(set(everything)), "splits overlap"
        print(result["benchmark"], sizes, "excluded:", len(result["excluded"]))


if __name__ == "__main__":
    main()
