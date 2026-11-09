"""End-to-end runs of the experiment scripts on a fake model (no GPU, temporary folders)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from editlattice.harness_runtime.runner import HarnessVersion

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "src" / "editlattice" / "seed_harnesses" / "math_direct.py"
FINAL = ROOT / "examples" / "gsm_demo_final.py"
# Every test here runs the scripts on GSM-Hard tasks, which scripts/fetch_datasets.sh downloads.
pytestmark = pytest.mark.skipif(not (ROOT / "datasets" / "gsmhard" / "gsmhardv2.jsonl").exists(), reason="GSM-Hard data not downloaded")


def run(args, tmp_path: Path) -> str:
    env = {**os.environ, "EDITLATTICE_RUNS_DIR": str(tmp_path / "runs"), "EDITLATTICE_RESULTS_DIR": str(tmp_path / "results")}
    completed = subprocess.run([sys.executable, *args], cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert completed.returncode == 0, completed.stderr[-2000:]
    return completed.stdout


def fake_config(tmp_path: Path) -> Path:
    path = tmp_path / "fake.yaml"
    path.write_text("backend: {name: fake, base_url: none, model: fake}\nparams: {}\nconcurrency: 4\nmax_llm_calls: 5\n")
    return path


def write_loop_log(directory: Path, store: Path, name: str, final_source: str) -> None:
    seed = HarnessVersion.from_source(SEED.read_text(), store)
    final = HarnessVersion.from_source(final_source, store)
    directory.mkdir(parents=True, exist_ok=True)
    entries = [{"parent_hash": seed.harness_hash, "accepted": True, "child_hash": final.harness_hash, "rationale": "test"}]
    (directory / f"{name}.log.jsonl").write_text("".join(json.dumps(entry) + "\n" for entry in entries))


def test_scripts_end_to_end(tmp_path: Path) -> None:
    config = fake_config(tmp_path)
    assert "accuracy" in run(["scripts/seed_eval.py", str(config), str(SEED), "gsmhard", "--n-tasks", "5"], tmp_path)
    out = run(["scripts/run_lattice.py", str(config), "gsmhard", "--seed", str(SEED), "--final", str(FINAL), "--name", "demo",
               "--n-tasks", "4", "--replicates", "2", "--brute", "--workers", "2", "--statement-level"], tmp_path)
    assert "mismatches 0" in out
    lattice = tmp_path / "runs" / "lattice" / "demo_stmt_search.json"
    data = json.loads(lattice.read_text())
    assert data["namespace"] == "demo" and len(data["cells"]) == 8
    audit = run(["scripts/a2_audit.py", str(config), "gsmhard", str(lattice), "--fraction", "0.5", "--workers", "2"], tmp_path)
    assert json.loads(audit.strip().splitlines()[-1])["pairs"] > 0
    report = run(["scripts/analyze_lattice.py", str(lattice), "--n-boot", "50"], tmp_path)
    assert "unit 0" in report and (tmp_path / "results" / "attribution_demo_stmt_search.json").exists()
    store = tmp_path / "runs" / "harnesses"
    evolution = tmp_path / "runs" / "pilot"
    write_loop_log(evolution, store, "loop_a", FINAL.read_text())
    write_loop_log(evolution, store, "loop_b", SEED.read_text().replace("Think step by step", "Think carefully"))
    gain = run(["scripts/gain_check.py", str(config), "gsmhard", str(evolution), "--n-tasks", "4", "--replicates", "2", "--workers", "2"], tmp_path)
    assert "real-gain loops" in gain
    result = json.loads((tmp_path / "results" / "gain_pilot.json").read_text())
    assert [loop["loop"] for loop in result["loops"]] == ["loop_a", "loop_b"] and all("gain" in loop for loop in result["loops"])
    pooled = run(["scripts/run_lattice.py", str(config), "gsmhard", "--pool", str(evolution / "loop_a.log.jsonl"), str(evolution / "loop_b.log.jsonl"),
                  "--harness-dir", str(store), "--name", "pooled", "--n-tasks", "3", "--replicates", "1", "--brute", "--workers", "2"], tmp_path)
    assert "mismatches 0" in pooled
    pooled_data = json.loads((tmp_path / "runs" / "lattice" / "pooled_search.json").read_text())
    assert pooled_data["conflicts"] and pooled_data["origin"]
    for split in ("search", "heldout"):
        run(["scripts/run_lattice.py", str(config), "gsmhard", "--pool", str(evolution / "loop_a.log.jsonl"), str(evolution / "loop_b.log.jsonl"),
             "--harness-dir", str(store), "--name", "pooled", "--split", split, "--n-tasks", "6", "--replicates", "2", "--workers", "2"], tmp_path)
    for split in ("heldout",):
        run(["scripts/run_lattice.py", str(config), "gsmhard", "--seed", str(SEED), "--final", str(FINAL), "--name", "demo",
             "--split", split, "--n-tasks", "4", "--replicates", "2", "--workers", "2", "--statement-level"], tmp_path)
    run(["scripts/a2_audit.py", str(config), "gsmhard", str(tmp_path / "runs/lattice/demo_stmt_heldout.json"), "--fraction", "1.0", "--workers", "2"], tmp_path)
    rq2 = run(["scripts/rq2_estimation.py", str(tmp_path / "runs/lattice/demo_stmt_heldout.json"), str(tmp_path / "results/a2_demo_stmt_heldout.json"), "--repetitions", "3"], tmp_path)
    assert "coupled" in rq2 and "independent" in rq2
    single = run(["scripts/rq_report.py", str(tmp_path / "runs/lattice/demo_stmt_search.json"), str(tmp_path / "runs/lattice/demo_stmt_heldout.json"), "--n-boot", "50"], tmp_path)
    assert "rq4" in single
    pooled_report = run(["scripts/rq_report.py", str(tmp_path / "runs/lattice/pooled_search.json"), str(tmp_path / "runs/lattice/pooled_heldout.json"), "--n-boot", "50"], tmp_path)
    data = json.loads((tmp_path / "results" / "report_pooled.json").read_text())
    assert data["rq5"] is not None and "superior_certified" in data["rq5"]


def test_tree_mode_matches_exhaustive_lattice(tmp_path: Path) -> None:
    """run_lattice --tree gives the same per-product outcomes as the exhaustive explorer, with leaves stored."""
    config = fake_config(tmp_path)
    common = ["gsmhard", "--seed", str(SEED), "--final", str(FINAL), "--n-tasks", "4", "--replicates", "2", "--workers", "2", "--statement-level", "--namespace", "shared"]
    run(["scripts/run_lattice.py", str(config), *common, "--name", "plain"], tmp_path)
    out = run(["scripts/run_lattice.py", str(config), *common, "--name", "treed", "--tree", "--brute"], tmp_path)
    assert "mismatches 0" in out
    plain = json.loads((tmp_path / "runs" / "lattice" / "plain_stmt_search.json").read_text())
    treed = json.loads((tmp_path / "runs" / "lattice" / "treed_stmt_search.json").read_text())
    assert treed["tree"] and all(cell["leaves"] for cell in treed["cells"])
    for a, b in zip(plain["cells"], treed["cells"]):
        assert (a["task_id"], a["replicate"]) == (b["task_id"], b["replicate"])
        assert a["outcomes"] == b["outcomes"] and len(a["executions"]) == len(b["executions"])
