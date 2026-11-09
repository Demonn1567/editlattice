"""Run several evolution loops concurrently (RQ1 edit corpus, RQ5 end-to-end).

Usage: python scripts/run_evolution.py configs/<evolution>.yaml

Each job = (proposer style, gate, seed). Jobs run in parallel threads so that a vLLM
server sees many concurrent requests (batching); each job is itself sequential.
Per-candidate logs: runs/<experiment>/<job>.log.jsonl. Resumable at run level (finished
runs and LLM responses are cached), not at loop level: a restarted job replays its earlier
iterations from cache at zero cost and continues.
"""

import argparse
import json
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from pilot import load_benchmark, select_tasks  # noqa: E402

from editlattice.evolution.live_pool import Executor  # noqa: E402
from editlattice.evolution.loop import LoopConfig, evolve  # noqa: E402
from editlattice.gates import baselines, ours  # noqa: E402
from editlattice.gates.judge import JudgeGate  # noqa: E402
from editlattice.harness_runtime.runner import HarnessVersion, RecordStore  # noqa: E402
from editlattice.llm.cache import ResponseCache  # noqa: E402
from editlattice.llm.settings import backend_settings, proposer_settings  # noqa: E402
from editlattice.llm.client import LLMClient, OpenAICompatibleBackend  # noqa: E402
from editlattice.llm.fake import FakeBackend, react_sql_responder  # noqa: E402
from editlattice.proposers.proposers import MetaHarnessStyleProposer, SelfHarnessStyleProposer, describe_environment  # noqa: E402

PROPOSERS = {"metaharness": MetaHarnessStyleProposer, "selfharness": SelfHarnessStyleProposer}


def make_gate(spec: dict, n_tasks: int, client: LLMClient, questions: dict[str, str]):
    """Gate callable from a config entry such as {name: O4, budget_runs: 160}."""
    name = spec["name"]
    params = {key: value for key, value in spec.items() if key != "name"}
    table = {
        "B1": baselines.gate_greedy,
        "B2": baselines.gate_best_so_far_tokens,
        "B3": baselines.gate_self_harness,
        "B3p": baselines.gate_strict_non_regression,
        "B4": baselines.gate_avg_at_k,
        "B5": baselines.gate_mcnemar_full,
        "B6": baselines.gate_paired_t,
        "B7": baselines.gate_fresh_validation,
        "B9": baselines.gate_random_subsample,
        "B10a": baselines.gate_historical_caching,
        "B10b": baselines.gate_stratified_subset,
        "B11": baselines.gate_pace,
        "O1": ours.gate_impact_mcnemar,
        "O4": ours.gate_impact_sequential,
    }
    if name == "B8":
        return partial(baselines.gate_two_gate, params=baselines.TwoGateParameters(**params))
    if name == "B12":
        return JudgeGate(client, questions, **params)
    return partial(table[name], **params)


def run_job(job: dict, config: dict, executor: Executor, search: dict, validation: dict, client: LLMClient, proposer_client: LLMClient) -> str:
    """One evolution loop."""
    job_id = f"{job['proposer']}_{job['gate']['name']}_s{job['seed']}"
    out_dir = ROOT / "runs" / config["experiment"]
    questions = {task_id: task.get("question", "") for task_id, task in search.items()}
    proposer = PROPOSERS[job["proposer"]](client=proposer_client, params=config["proposer_params"], stream_prefix=f"{config['experiment']}:{job_id}",
                                          prompt_version=config.get("proposer_prompt_version", 1), env_doc=config.get("_env_doc", ""),
                                          accept_diff=config.get("proposer_accept_diff", False),
                                          strip_thinking=config.get("proposer_strip_thinking", False))
    gate = make_gate(job["gate"], len(search), client, questions)
    seed = HarnessVersion.from_file(ROOT / config["seed_harness"], ROOT / "runs" / "harnesses")
    loop_config = LoopConfig(
        run_id=f"{config['experiment']}:{job_id}",
        total_budget_runs=config["total_budget_runs"],
        n_streams=config["n_streams"],
        k_char=config["k_char"],
        max_iterations=config["max_iterations"],
        seed=job["seed"],
    )
    log_path = out_dir / f"{job_id}.log.jsonl"
    final_path = out_dir / f"{job_id}.final.json"
    if final_path.exists():
        # A finished loop is never re-run: after a state purge its replay could re-execute
        # purged runs and take a different trajectory than the one already logged.
        return job_id + " (already finished)"
    # A restarted loop replays its earlier iterations from cache and rewrites every entry,
    # so the old log is removed first (otherwise entries would be duplicated).
    if log_path.exists():
        log_path.unlink()
    final = evolve(loop_config, executor, search, validation, seed, ROOT / "runs" / "harnesses", proposer, gate, log_path)
    final_path.write_text(json.dumps({"job": job, "final_hash": final.harness_hash}))
    return job_id


def main() -> None:
    """Expand the job grid and run it."""
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    backend_cfg = backend_settings(config)
    benchmark = load_benchmark(config["benchmark"])
    search = {task["task_id"]: task for task in select_tasks(benchmark, "search", config["n_search"])}
    validation = {task["task_id"]: task for task in select_tasks(benchmark, "validation", config["n_validation"])}
    if config.get("proposer_prompt_version", 1) >= 2:
        # Phase A2: the environment description is generated by one procedure for every benchmark.
        first = next(iter(search.values()))
        env = benchmark.make_env(first)
        try:
            config["_env_doc"] = describe_environment(env, benchmark.public_view(first))
        finally:
            benchmark.close_env(env)
        print(config["_env_doc"], flush=True)
    if backend_cfg["name"].startswith("fake"):
        backend = FakeBackend(react_sql_responder, name=backend_cfg["name"])
    else:
        backend = OpenAICompatibleBackend(backend_cfg["base_url"], backend_cfg["name"])
    client = LLMClient(backend=backend, model=backend_cfg["model"], cache=ResponseCache(ROOT / "runs" / config.get("cache_dir", "cache")), seed_mode=config.get("seed_mode", "request"))
    # The proposer may be a different (stronger) model than the solver; same response cache,
    # whose keys include the model name.
    proposer_client = client
    proposer_cfg = proposer_settings(config)
    if proposer_cfg is not None:
        proposer_backend = OpenAICompatibleBackend(proposer_cfg["base_url"], proposer_cfg["name"],
                                                   timeout_seconds=float(proposer_cfg.get("timeout_seconds", 1800.0)),
                                                   max_in_flight=proposer_cfg.get("max_in_flight"))
        proposer_client = LLMClient(backend=proposer_backend, model=proposer_cfg["model"], cache=client.cache, seed_mode=config.get("seed_mode", "request"))
    out_dir = ROOT / "runs" / config["experiment"]
    out_dir.mkdir(parents=True, exist_ok=True)
    executor = Executor(benchmark, client, RecordStore(out_dir / "records.jsonl"), config["params"], config["max_llm_calls"])
    jobs = []
    if "pairs" in config:
        # Each proposer with its own method's acceptance rule (faithful lineages, RQ1).
        for proposer, gate in config["pairs"]:
            for seed in config["seeds"]:
                jobs.append({"proposer": proposer, "gate": gate, "seed": seed})
    else:
        for proposer in config["proposers"]:
            for gate in config["gates"]:
                for seed in config["seeds"]:
                    jobs.append({"proposer": proposer, "gate": gate, "seed": seed})
    print(f"{len(jobs)} jobs", flush=True)
    with ThreadPoolExecutor(max_workers=config["parallel_jobs"]) as pool:
        futures = {pool.submit(run_job, job, config, executor, search, validation, client, proposer_client): job for job in jobs}
        for future in as_completed(futures):
            try:
                print("finished", future.result(), flush=True)
            except Exception:  # noqa: BLE001 - one failed job must not stop the others
                print("job failed", futures[future], traceback.format_exc(), flush=True)


if __name__ == "__main__":
    main()
