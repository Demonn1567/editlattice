"""Compute used by the study, from the stored lattices and stage logs (never estimated by hand).

Lattices: wall-clock seconds, executions, generated and nominal tokens (nominal = what the runs would
have generated without replay). Evolution: wall-clock time per pilot from the stage logs' start/exit
lines. Usage: python scripts/compute_summary.py   (on the GPU machine and on the Mac; writes
results/compute_<host>.json).
"""

import json
import re
import socket
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def lattice_totals() -> dict:
    """Sum over every lattice file except quarantined ones, grouped by the solver model that ran it
    (each machine reports the models it served; copies of the other machine's files are ignored there)."""
    by_model: dict[str, dict] = {}
    for path in sorted((ROOT / "runs" / "lattice").glob("*.json")):
        if path.name.startswith(("invalid_", "buggy_", "dup_")):
            continue
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        cells = data.get("cells", [])
        executions = [e for cell in cells for e in cell.get("executions", [])]
        model = str(data.get("model", "unknown"))
        local = model.startswith("mlx-community") == (not (ROOT / ".venv-gpu").exists())
        if not local:
            continue
        totals = by_model.setdefault(model, {"files": 0, "seconds": 0.0, "executions": 0, "generated_tokens": 0, "nominal_tokens": 0})
        totals["files"] += 1
        totals["seconds"] += float(data.get("seconds", 0.0))
        totals["executions"] += len(executions)
        totals["generated_tokens"] += sum(int(e.get("tokens_out", 0)) for e in executions)
        totals["nominal_tokens"] += sum(int(e.get("nominal_tokens_out", 0)) for e in executions)
    return by_model


def evolution_hours() -> dict:
    """Hours between each pilot's 'start' and 'exit' lines in the stage logs."""
    hours = {}
    pattern = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (A\d? )?evolution (\w+) (start|exit)")
    for log in sorted((ROOT / "runs" / "logs").glob("stage*.log")):
        starts = {}
        for line in log.read_text(errors="replace").splitlines():
            match = pattern.match(line)
            if not match:
                continue
            stamp, phase, bench, event = match.groups()
            key = f"{(phase or 'A ').strip()}:{bench}"
            when = datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S")
            if event == "start":
                starts[key] = when
            elif key in starts:
                hours[f"{log.stem}:{key}"] = round((when - starts.pop(key)).total_seconds() / 3600, 2)
    return hours


def main() -> None:
    host = socket.gethostname().split(".")[0]
    summary = {"host": host, "lattices": lattice_totals(), "evolution_hours": evolution_hours()}
    out = ROOT / "results" / f"compute_{'gpu' if (ROOT / '.venv-gpu').exists() else 'mac'}.json"
    out.write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
