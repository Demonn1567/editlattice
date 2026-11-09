"""Proposer probe (search-side only, no evolution, no held-out data): how often does a proposer model
produce an applicable edit, and does it use the environment, under prompt v1 and v2?

Usage: python scripts/probes/proposer_probe.py <base_url> <model> <out.json> [--n 4]
Builds each prompt from the seed harness and its logged search-set runs of the Phase A pilot
(runs/gpu_pilot/gain_pilot_<benchmark>/records.jsonl), exactly as the loop's first iteration would.
"""

import argparse
import json
import random
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from pilot import load_benchmark, select_tasks  # noqa: E402

from editlattice.harness_runtime.runner import HarnessVersion  # noqa: E402
from editlattice.llm.cache import ResponseCache  # noqa: E402
from editlattice.llm.client import LLMClient, OpenAICompatibleBackend  # noqa: E402
from editlattice.proposers.proposers import MetaHarnessStyleProposer, describe_environment  # noqa: E402

SEEDS = {"lcb": "lcb_oneshot.py", "gsmhard": "math_direct.py"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_url")
    parser.add_argument("model")
    parser.add_argument("out")
    parser.add_argument("--n", type=int, default=4)
    args = parser.parse_args()
    client = LLMClient(backend=OpenAICompatibleBackend(args.base_url, "probe"), model=args.model,
                       cache=ResponseCache(Path(tempfile.mkdtemp()) / "cache"), seed_mode="request")
    params = {"temperature": 0.7, "top_p": 0.95, "max_tokens": 3072, "chat_template_kwargs": {"enable_thinking": False}}
    rows = []
    for name, seed_file in SEEDS.items():
        benchmark = load_benchmark(name)
        search = {t["task_id"]: t for t in select_tasks(benchmark, "search", 60)}
        source = (ROOT / "src/editlattice/seed_harnesses" / seed_file).read_text()
        seed_hash = HarnessVersion.from_source(source, Path(tempfile.mkdtemp())).harness_hash
        records = [json.loads(line) for line in open(ROOT / f"runs/gpu_pilot/gain_pilot_{name}/records.jsonl")]
        records = [r for r in records if r["harness_hash"] == seed_hash and r["task_id"] in search][:240]
        first = next(iter(search.values()))
        env = benchmark.make_env(first)
        env_doc = describe_environment(env, benchmark.public_view(first))
        benchmark.close_env(env)
        questions = {task_id: task.get("question", "") for task_id, task in search.items()}
        for version in (1, 2):
            proposer = MetaHarnessStyleProposer(client=client, params=params, stream_prefix=f"probe:{name}:v{version}",
                                                prompt_version=version, env_doc=env_doc, accept_diff=version == 2)
            for k in range(args.n):
                started = time.time()
                result = proposer.propose(k, source, records, questions, [], random.Random(k))
                child = result.child_source or ""
                rows.append({"benchmark": name, "version": version, "k": k, "valid": result.valid, "error": result.error[:120],
                             # More environment calls than the seed has (the seed's docstring mentions the API).
                             "calls_env": bool(child) and sum(child.count(c) for c in ("env.run_python(", "env.execute(")) > sum(source.count(c) for c in ("env.run_python(", "env.execute(")),
                             "seconds": round(time.time() - started, 1), "tokens_out": result.tokens_out, "records_used": len(records)})
                print(json.dumps(rows[-1]), flush=True)
    Path(args.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
