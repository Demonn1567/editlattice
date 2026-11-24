"""Highlights for the Knowledge-Based Systems submission (3 to 5 bullets, each at most 85 characters with spaces).

Every number is read from paper/macros.tex, which scripts/paper_numbers.py generates, so the highlights cannot drift
from the paper. Writes paper/highlights.txt.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def macros() -> dict[str, str]:
    text = (ROOT / "paper" / "macros.tex").read_text()
    return {k: v.replace("\\%", "%") for k, v in re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", text)}


def main() -> None:
    m = macros()
    bullets = [
        "Variational replay evaluates every subset of an agent's accepted edits exactly",
        "A task costs at most 2^w runs when each run reads at most w of the edits",
        f"Only {m['PilotRealGain']} of {m['PilotLoops']} evolution loops gave a real held-out gain",
        f"{m['NullInertPct']} of the edits kept by loops without a real gain never changed an outcome",
        f"Shared random streams cut Shapley estimation error by {m['RqTwoRedMin']} to {m['RqTwoRedMax']} at equal cost",
    ]
    assert 3 <= len(bullets) <= 5
    for b in bullets:
        assert len(b) <= 85, (len(b), b)
        assert "\\" not in b and "{" not in b, b
    (ROOT / "paper" / "highlights.txt").write_text("Highlights\n\n" + "\n".join("• " + b for b in bullets) + "\n")
    for b in bullets:
        print(len(b), b)


if __name__ == "__main__":
    main()
