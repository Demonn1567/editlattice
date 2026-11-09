"""Test-wide settings: scores of test benchmarks never go to the real on-disk score cache."""

import os
import tempfile

os.environ.setdefault("EDITLATTICE_SCORE_CACHE", tempfile.mkdtemp(prefix="editlattice-score-cache-"))
