"""Figures of the paper, drawn only from stored result files (never from typed numbers).

Usage: python scripts/make_figures.py   (writes paper/figures/*.pdf)
Colours: a validated categorical palette (blue, orange, aqua, yellow on white); every series is also told
apart by a line style or marker and a legend, never by colour alone.
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "paper" / "figures"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6, "pdf.fonttype": 42,
})


def rq2() -> None:
    """Mean absolute error of sampled Shapley estimates against the exact lattice, coupled vs independent streams."""
    panels = [("Planted (GSM-Hard)", RESULTS / "rq2_planted_stmt_heldout.json")] + [
        (f"Constructed {name}", RESULTS / "phaseC" / f"rq2_constructed_{key}_stmt_heldout.json")
        for key, name in (("gsmhard", "GSM-Hard"), ("lcb", "LiveCodeBench"), ("bird", "BIRD"))]
    fig, axes = plt.subplots(1, 4, figsize=(6.9, 1.85), sharey=False)
    styles = {("permutation", "coupled"): (BLUE, "-", "o"), ("permutation", "independent"): (ORANGE, "-", "o"),
              ("kernelshap", "coupled"): (BLUE, "--", "s"), ("kernelshap", "independent"): (ORANGE, "--", "s")}
    names = {"permutation": "permutation", "kernelshap": "KernelSHAP"}
    for ax, (title, path) in zip(axes, panels):
        data = json.loads(path.read_text())
        for (estimator, regime), (colour, line, marker) in styles.items():
            rows = sorted((r for r in data["rows"] if r["estimator"] == estimator and r["regime"] == regime), key=lambda r: r["budget"])
            ax.plot([r["budget"] for r in rows], [100 * r["mae"] for r in rows], color=colour, linestyle=line, marker=marker,
                    markersize=3.2, markeredgecolor="white", markeredgewidth=0.6, label=f"{names[estimator]}, {regime}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        budgets = data["budgets"]
        ax.set_xticks([budgets[0], budgets[len(budgets) // 2], budgets[-1]])
        ax.set_xticklabels([f"{b:,}" for b in (budgets[0], budgets[len(budgets) // 2], budgets[-1])])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_title(title)
        ax.set_xlabel("budget (task-runs)")
    axes[0].set_ylabel("mean absolute error (points)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(w_pad=0.6)
    fig.savefig(OUT / "rq2_error.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)


def evolved() -> None:
    """Held-out and search Shapley values with 95% intervals for the three evolved real-gain lineages."""
    lineages = [
        ("LiveCodeBench B1", "report_phaseA3_lcb_metaharness_B1_s2_stmt.json",
         ["U0 code extraction", "U1 examples in prompt", "U2 test and repair"]),
        ("BIRD B1", "report_phaseA3_bird_metaharness_B1_s1_stmt.json", ["U0 SQL guidance", "U1 quoted columns"]),
        ("BIRD B3", "report_phaseA3_bird_metaharness_B3_s1_stmt.json",
         ["U0 SQL guidance", "U1 docstring", "U2 table descriptions"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 1.9), sharex=True)
    for ax, (title, file, labels) in zip(axes, lineages):
        report = json.loads((RESULTS / "gpu" / file).read_text())["rq3"]
        units = sorted(report["heldout"]["shapley"], key=int)
        assert len(units) == len(labels)
        for offset, split, colour, marker in ((-0.14, "search", ORANGE, "s"), (0.14, "heldout", BLUE, "o")):
            for i, unit in enumerate(units):
                entry = report[split]["shapley"][unit]
                y = len(units) - 1 - i + offset
                ax.plot([100 * entry["low"], 100 * entry["high"]], [y, y], color=colour, linewidth=1.4, solid_capstyle="round")
                ax.plot(100 * entry["value"], y, marker=marker, color=colour, markersize=4.5, markeredgecolor="white",
                        markeredgewidth=0.7, linestyle="none", label=("held-out" if split == "heldout" else "search") if i == 0 else None)
        ax.axvline(0, color=MUTED, linewidth=0.8)
        ax.set_yticks(range(len(units)))
        ax.set_yticklabels(list(reversed(labels)))
        ax.set_ylim(-0.6, len(units) - 0.4)
        ax.set_title(title)
        ax.set_xlabel("Shapley value (points)")
        ax.grid(axis="y", visible=False)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles[::-1], names[::-1], loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(w_pad=0.8)
    fig.savefig(OUT / "evolved_attribution.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)


def scaling() -> None:
    """Executions per cell against the number of units, for routes of 2, 4 and 8 steps, against brute force."""
    rows = json.loads((RESULTS / "scaling.json").read_text())["rows"]
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    for length, colour, marker in ((2, BLUE, "o"), (4, ORANGE, "s"), (8, AQUA, "^")):
        sub = [r for r in rows if r["length"] == length and r["m"] >= length and not (r["length"] == r["m"] and r["m"] > 8)]
        sub.sort(key=lambda r: r["m"])
        ax.plot([r["m"] for r in sub], [r["executions"] / r["cells"] for r in sub], color=colour, marker=marker, markersize=3.5,
                markeredgecolor="white", markeredgewidth=0.6, label=f"route of {length} steps")
    dense = sorted((r for r in rows if r["length"] == r["m"] and r["m"] >= 10), key=lambda r: r["m"])
    ax.plot([r["m"] for r in dense], [r["executions"] / r["cells"] for r in dense], color=YELLOW, marker="D", markersize=3.5,
            markeredgecolor="white", markeredgewidth=0.6, label="every run reads every unit")
    ms = sorted({r["m"] for r in rows})
    ax.plot(ms, [2 ** m for m in ms], color=MUTED, linestyle=":", linewidth=1.2, label="brute force ($2^m$)")
    ax.set_yscale("log")
    ax.set_xlabel("number of units $m$")
    ax.set_ylabel("executions per cell")
    ax.set_ylim(1, 3e5)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "scaling.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)


def null_group() -> None:
    """Accepted units of the evolved lineages without a real gain, by phase and held-out effect."""
    units = json.loads((RESULTS / "null_summary.json").read_text())["units"]
    phases = [("A", "phaseA_"), ("A2", "phaseA2_"), ("A3", "phaseA3_"), ("A4", "phaseA4_")]
    classes = [("exactly inert", lambda u: u["inert"], BLUE), ("no detectable effect", lambda u: not u["inert"] and not u["matters"], AQUA),
               ("helps (Holm)", lambda u: u["matters"] and not u["harmful"], YELLOW), ("harms (Holm)", lambda u: u["matters"] and u["harmful"], ORANGE)]
    counts = {name: [sum(1 for u in units if u["lineage"].startswith(prefix) and rule(u)) for _, prefix in phases] for name, rule, _ in classes}
    assert sum(sum(v) for v in counts.values()) == len(units)
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    left = [0] * len(phases)
    for name, _, colour in classes:
        values = counts[name]
        bars = ax.barh([f"Phase {p}" for p, _ in phases], values, left=left, color=colour, edgecolor="white", linewidth=1.2, height=0.62, label=name)
        for bar, value in zip(bars, values):
            if value >= 3:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, str(value), ha="center", va="center",
                        fontsize=6.5, color=INK)
        left = [a + b for a, b in zip(left, values)]
    ax.invert_yaxis()
    ax.set_xlabel("accepted units")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.45, -0.32))
    fig.tight_layout()
    fig.savefig(OUT / "null_units.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for figure in (rq2, evolved, scaling, null_group):
        figure()
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
