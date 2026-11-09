"""Number of conflict-free products of the pooled lattice of the given loop logs (DEC-14 size rule).

Usage: python scripts/pool_size.py <harness_dir> <log> [<log> ...]
"""

import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from editlattice.lineage import load_loop_lineage  # noqa: E402
from editlattice.units import combine_lineages  # noqa: E402

lineages = [load_loop_lineage(log, sys.argv[1]) for log in sys.argv[2:]]
finals = [lineage.product(frozenset(edit.index for edit in lineage.edits)) for lineage in lineages]
pooled, _ = combine_lineages(lineages[0].seed, finals)
m = pooled.m
count = sum(1 for r in range(m + 1) for c in itertools.combinations(range(m), r) if not any(p <= frozenset(c) for p in pooled.conflicts)) if m <= 22 else 2 ** m
print(count)
