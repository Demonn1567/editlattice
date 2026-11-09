"""Re-check the statement-level marking of every stored lattice under the current add_markers rules.

Usage: python scripts/check_markers.py [lattice files ...]   (default: runs/lattice/*_stmt_*.json)
Prints, per lattice, the units it marked and the units the current rules mark; a lattice is affected
if it marked a unit the current rules refuse (its shared runs may be wrong for those units).
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from a2_audit import unit_set_from_lattice  # noqa: E402
from editlattice.units import add_markers  # noqa: E402
from editlattice.variational import ReadModel  # noqa: E402


def main() -> None:
    files = [Path(arg) for arg in sys.argv[1:]] or sorted((ROOT / "runs" / "lattice").glob("*_stmt_*.json"))
    affected = []
    for path in files:
        data = json.loads(path.read_text())
        if not data.get("statement_level"):
            continue
        unit_set = unit_set_from_lattice(data, data["seed_source"])
        coarse = ReadModel.build(unit_set)
        alive = frozenset(u for units in coarse.depends.values() for u in units) | coarse.global_units
        _, marked = add_markers(unit_set, eligible=alive)
        stored = set(data.get("marked_units", []))
        bad = sorted(stored - set(marked))
        if bad:
            affected.append(path.name)
        print(f"{path.name}: stored {sorted(stored)} now {sorted(marked)}{'  AFFECTED ' + str(bad) if bad else ''}")
    print("affected:", affected)


if __name__ == "__main__":
    main()
