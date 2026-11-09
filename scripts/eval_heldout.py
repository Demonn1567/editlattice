"""RQ5: evaluate final harnesses, the seed and TTS baselines on the held-out split.

Usage: python scripts/eval_heldout.py configs/<heldout>.yaml

Harnesses: the seed; best-of-k TTS baselines built from the seed; every
runs/<rq5 experiment>/*.final.json harness. All share CRN streams per (task, replicate)
(namespace ``heldout``) so pairwise differences are lower-variance; each run's marginal is
that of a fresh run. Never used for tuning or gating (D-007).
Output: results/<experiment>.json with accuracy and a task-cluster bootstrap CI per harness.
"""

import argparse
import glob
import json
import random
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from pilot import load_benchmark, progress_printer, select_tasks  # noqa: E402

from editlattice.harness_runtime.runner import HarnessVersion, RecordStore, RunRequest, run_many  # noqa: E402
from editlattice.llm.cache import ResponseCache  # noqa: E402
from editlattice.llm.client import LLMClient, OpenAICompatibleBackend  # noqa: E402
from editlattice.llm.fake import FakeBackend, react_sql_responder  # noqa: E402
from editlattice.llm.settings import backend_settings  # noqa: E402
from editlattice.seed_harnesses.tts import make_tts_harness  # noqa: E402


def bootstrap_ci(per_task: list[float], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap over tasks."""
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        total = 0.0
        for _ in range(len(per_task)):
            total += per_task[rng.randrange(len(per_task))]
        means.append(total / len(per_task))
    means.sort()
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def main() -> None:
    """Collect harnesses, run them, summarise."""
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    backend_cfg = backend_settings(config)
    benchmark = load_benchmark(config["benchmark"])
    tasks = select_tasks(benchmark, "heldout", config["n_heldout"])
    store_dir = ROOT / "runs" / "harnesses"
    seed_source = (ROOT / config["seed_harness"]).read_text()
    harnesses: dict[str, HarnessVersion] = {"seed": HarnessVersion.from_source(seed_source, store_dir)}
    for k in config.get("tts_k", []):
        harnesses[f"tts_k{k}"] = HarnessVersion.from_source(make_tts_harness(seed_source, k, config["benchmark"]), store_dir)
    for pattern in config.get("final_globs", []):
        for path in sorted(glob.glob(str(ROOT / pattern))):
            final = json.loads(Path(path).read_text())
            name = Path(path).name.replace(".final.json", "")
            harnesses[f"{Path(path).parent.name}/{name}"] = HarnessVersion.from_file(store_dir / final["final_hash"] / "harness.py", store_dir)
    # Same per-run LLM-call cap as during evolution for every ordinary harness; a best-of-k
    # TTS harness gets k times that cap (k complete attempts).
    groups: dict[int, list[RunRequest]] = {}
    for name, version in harnesses.items():
        cap = config["max_llm_calls"]
        if name.startswith("tts_k"):
            cap = config["max_llm_calls"] * int(name[len("tts_k"):])
        for task in tasks:
            for replicate in range(config["replicates"]):
                groups.setdefault(cap, []).append(RunRequest(task, version, replicate, True, False, "heldout"))
    requests = [request for group in groups.values() for request in group]
    if backend_cfg["name"].startswith("fake"):
        backend = FakeBackend(react_sql_responder, name=backend_cfg["name"])
    else:
        backend = OpenAICompatibleBackend(backend_cfg["base_url"], backend_cfg["name"])
    client = LLMClient(backend=backend, model=backend_cfg["model"], cache=ResponseCache(ROOT / "runs" / config.get("cache_dir", "cache")), seed_mode=config.get("seed_mode", "request"))
    out_dir = ROOT / "runs" / config["experiment"]
    store = RecordStore(out_dir / "records.jsonl")
    print(f"{len(harnesses)} harnesses x {len(tasks)} tasks x {config['replicates']} reps = {len(requests)} runs", flush=True)
    records = []
    for cap, group in groups.items():
        records.extend(run_many(benchmark, group, client, config["params"], store, config["concurrency"], cap, progress_printer(len(group))))
    by_hash: dict[str, dict[str, list[int]]] = {}
    for record in records:
        by_hash.setdefault(record["harness_hash"], {}).setdefault(record["task_id"], []).append(record["outcome"])
    summary = {}
    for name, version in harnesses.items():
        per_task = [sum(outcomes) / len(outcomes) for outcomes in by_hash[version.harness_hash].values()]
        low, high = bootstrap_ci(per_task)
        summary[name] = {"hash": version.harness_hash, "accuracy": sum(per_task) / len(per_task), "ci95": [low, high], "n_tasks": len(per_task), "replicates": config["replicates"]}
        print(name, json.dumps(summary[name]), flush=True)
    (ROOT / "results" / f"{config['experiment']}.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
