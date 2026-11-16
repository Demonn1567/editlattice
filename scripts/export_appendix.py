"""Appendix material generated from code and result files: the proposer prompt (built from the constants the
proposer uses), copies of the seed harnesses, and the full per-loop and per-lineage result tables.

Usage: python scripts/export_appendix.py   (writes paper/appendix/*)
"""

import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from editlattice.proposers.proposers import COPY_RULE_V2, EDIT_FORMAT_HELP, PROPOSER_SYSTEM  # noqa: E402

RESULTS = ROOT / "results"
OUT = ROOT / "paper" / "appendix"
BENCH = {"gsmhard": "GSM-Hard", "lcb": "LiveCodeBench", "bird": "BIRD"}


def points(x: float) -> str:
    return f"{100 * x:+.1f}".replace("-", "$-$")


def prompt() -> None:
    """The Phase A2 to A4 proposer prompt with placeholders where run-specific text goes (prompt version 2)."""
    user = (
        "<description of the environment: the task fields and the env methods the harness may call>\n\n"
        "Current harness (search-set accuracy <score>):\n```python\n<current harness source>\n```\n\n"
        "Earlier candidate edits and their results:\n"
        "Candidate <n> (<accepted|rejected>; parent score <s>, candidate score <s>):\n"
        "(changed <k> line(s); new lines:)\n<new lines>\n  ... (up to six earlier candidates)\n\n"
        "Failing runs:\n<four failing runs: task, final answer, outcome, error, last model input and reply>\n\n"
        "Passing runs:\n<two passing runs, same format>\n\n"
        "Propose one edit to the harness that improves accuracy. Briefly explain your reasoning first.\n\n"
        + EDIT_FORMAT_HELP + "\n" + COPY_RULE_V2)
    (OUT / "prompt.txt").write_text(f"[system]\n{PROPOSER_SYSTEM}\n\n[user]\n{user}\n")


def seeds() -> None:
    for name in ("math_direct", "lcb_oneshot", "bird_oneshot"):
        shutil.copy(ROOT / "src" / "editlattice" / "seed_harnesses" / f"{name}.py", OUT / f"{name}.py")


def loops_table() -> None:
    """Every evolution loop: proposals, validity, accepted edits, net units and held-out gain."""
    loops = json.loads((RESULTS / "pilot_summary.json").read_text())["loops"]
    lines = ["\\begin{longtable}{llllrrrrl}",
             "\\caption{Every evolution loop. Gain: held-out gain of the final harness over the seed with its 95\\% "
             "task-bootstrap interval, in points; $^\\dagger$: a real gain under the registered criterion; none: no "
             "accepted edit.}\\label{tab:all-loops}\\\\",
             "\\toprule", "Phase & Benchmark & Gate & Seed & proposals & valid & accepted & units & gain \\\\", "\\midrule", "\\endfirsthead",
             "\\toprule", "Phase & Benchmark & Gate & Seed & proposals & valid & accepted & units & gain \\\\", "\\midrule", "\\endhead",
             "\\bottomrule", "\\endfoot"]
    for row in loops:
        gate, seed = row["loop"].replace("metaharness_", "").split("_s")
        gain = "none" if row["gain"] is None else f"{points(row['gain'])} [{points(row['low'])}, {points(row['high'])}]" + ("$^\\dagger$" if row["real_gain"] else "")
        lines.append(f"{row['phase']} & {BENCH[row['benchmark']]} & {gate} & {seed} & {row['proposals']} & {row['valid']} & "
                     f"{row['accepted']} & {row['net_units']} & {gain} \\\\")
    lines.append("\\end{longtable}")
    (OUT / "all_loops.tex").write_text("\n".join(lines) + "\n")


def lineages_table() -> None:
    """Every evolved lineage without a real gain: unit effects, executions, exactness and fresh-run audits."""
    summary = json.loads((RESULTS / "null_summary.json").read_text())
    units = defaultdict(list)
    for unit in summary["units"]:
        units[unit["lineage"]].append(unit)
    lines = ["{\\footnotesize\\setlength{\\tabcolsep}{4pt}", "\\begin{longtable}{lllrrrrrrl}",
             "\\caption{Every evolved lineage without a real gain. Units: net units; inert: change no held-out outcome; "
             "helps and harms: held-out Shapley interval excludes 0 after Holm correction; overfit: search interval "
             "above 0, held-out not. Executions: what the search and held-out lattices ran, next to the brute-force "
             "count. Audit: how many (product, cell) pairs were compared with runs alone, and how many disagreed. "
             "Fresh: pairs rerun on private streams, and products with Fisher $p<0.05$.}\\label{tab:all-lineages}\\\\",
             "\\toprule", "Phase & Benchmark & Loop & units & inert & helps & harms & overfit & executions & audit; fresh \\\\",
             "\\midrule", "\\endfirsthead",
             "\\toprule", "Phase & Benchmark & Loop & units & inert & helps & harms & overfit & executions & audit; fresh \\\\",
             "\\midrule", "\\endhead", "\\bottomrule", "\\endfoot"]
    def order(name: str) -> tuple:
        phase, bench, *rest = name.split("_", 2)
        return (phase, list(BENCH).index(bench), rest)
    for name in sorted(units, key=order):
        phase, bench, loop = name.split("_", 2)
        group = units[name]
        cost = summary["costs"][name]
        executions = sum(cost[s]["executions"] for s in cost)
        brute = sum(cost[s]["brute_force"] for s in cost)
        audit = summary["audits"].get(name, {})
        fresh = summary["fresh"].get(name, {})
        gate, seed = loop.replace("metaharness_", "").split("_s")
        lines.append(
            f"{phase.replace('phase', '')} & {BENCH[bench]} & {gate} s{seed} & {len(group)} & {sum(u['inert'] for u in group)} & "
            f"{sum(u['matters'] and not u['harmful'] for u in group)} & {sum(u['harmful'] for u in group)} & "
            f"{sum(u['overfitting'] for u in group)} & {executions:,}/{brute:,} & "
            f"{audit.get('checked', 0)}/{audit.get('mismatches', 0)}; {fresh.get('pairs', 0)}/{fresh.get('products_with_p_below_0.05', 0)} \\\\")
    lines.append("\\end{longtable}}")
    (OUT / "all_lineages.tex").write_text("\n".join(lines).replace(",", "{,}").replace("{,}/", "/") + "\n")


def settings_table() -> None:
    """Per-phase settings, read from the BIRD config of each phase (the three benchmarks differ only in data)."""
    import yaml
    phases = [("A", "gain_pilot_bird"), ("A2", "gain_pilot2_bird"), ("A3", "gain_pilot3_bird"), ("A4", "gain_pilot4_bird")]
    configs = {phase: yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text()) for phase, name in phases}
    def comparable(config: dict) -> dict:
        """Settings that must agree across benchmarks: all but data, seed, the proposer timeout and the solver budget."""
        out = {k: v for k, v in config.items() if k not in ("experiment", "benchmark", "seed_harness")}
        out["proposer_backend"] = {k: v for k, v in out["proposer_backend"].items() if k != "timeout_seconds"}
        out["params"] = {k: v for k, v in out["params"].items() if k != "max_tokens"}
        return out
    budget = {}
    for bench in ("gsmhard", "lcb", "bird"):
        for phase, name in phases:
            other = yaml.safe_load((ROOT / "configs" / f"{name.replace('bird', bench)}.yaml").read_text())
            assert comparable(other) == comparable(configs[phase]), (phase, bench)
            budget.setdefault(phase, {})[bench] = other["params"]["max_tokens"]
    # The text states these budgets (DEC-16); fail loudly if the configs say otherwise.
    assert all(budget[p] == {"gsmhard": 768, "lcb": 2048, "bird": 768} for p in budget), budget
    rows = [
        ("Solver", lambda c: c["backend"]["model"].split("/")[-1]),
        ("Solver temperature, top-$p$", lambda c: f"{c['params']['temperature']}, {c['params']['top_p']}"),
        ("Solver max tokens in evolution", lambda c: f"{c['params']['max_tokens']} (LiveCodeBench 2{{,}}048)"),
        ("Proposer", lambda c: c["proposer_backend"]["model"].split("/")[-1] + (" (MLX)" if "mlx" in c["proposer_backend"]["model"] else "")),
        ("Proposer reasoning mode", lambda c: "on" if c["proposer_params"].get("chat_template_kwargs", {}).get("enable_thinking") else "off"),
        ("Proposer temperature, max tokens", lambda c: f"{c['proposer_params']['temperature']}, {c['proposer_params']['max_tokens']}"),
        ("Prompt version", lambda c: str(c.get("proposer_prompt_version", 1))),
        ("Gates; seeds", lambda c: ", ".join(g["name"] for g in c["gates"]) + "; " + ", ".join(map(str, c["seeds"]))),
        ("Iterations; task-runs per loop", lambda c: f"{c['max_iterations']}; {c['total_budget_runs']:,}".replace(",", "{,}")),
        ("Search; validation tasks", lambda c: f"{c['n_search']}; {c['n_validation']}"),
        ("Model calls per run (limit)", lambda c: str(c["max_llm_calls"])),
    ]
    lines = ["\\begin{table}[ht]", "\\centering", "\\caption{Settings of the evolution phases, read from the configuration files. "
             "The three benchmarks of a phase share every setting except the data, the seed harness, the proposer request "
             "timeout (an operational setting) and the solver's output budget on LiveCodeBench. Every held-out gain check "
             "and lattice uses 768 solver tokens on every benchmark (Appendix~\\ref{app:deviations}).}", "\\label{tab:settings}",
             "\\small", "\\resizebox{\\textwidth}{!}{\\begin{tabular}{lllll}", "\\toprule", "Setting & Phase A & Phase A2 & Phase A3 & Phase A4 \\\\", "\\midrule"]
    for label, get in rows:
        lines.append(f"{label} & " + " & ".join(get(configs[p]).replace("_", "\\_") for p, _ in phases) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    (OUT / "settings.tex").write_text("\n".join(lines) + "\n")


def sensitivity_table() -> None:
    """DEC-16: every LiveCodeBench loop's held-out gain at the registered 768 and at 2,048 solver tokens."""
    rows = ["\\begin{table}[ht]", "\\centering", "\\caption{LiveCodeBench gain checks with 768 solver output tokens "
            "(registered, as in every evaluation) and with 2{,}048 (the evolution budget). Gain of the final harness "
            "over the seed with its 95\\% interval, in points; $^\\dagger$: a real gain; none: no accepted edit.}",
            "\\label{tab:sens}", "\\small", "\\begin{tabular}{lllll}", "\\toprule", "Phase & Gate & Seed & 768 tokens & 2{,}048 tokens \\\\", "\\midrule"]
    def text(l):
        if l.get("gain") is None:
            return "none"
        return f"{points(l['gain'])} [{points(l['low'])}, {points(l['high'])}]" + ("$^\\dagger$" if l["real_gain"] else "")
    for phase, prefix in (("A", "gain_pilot"), ("A2", "gain_pilot2"), ("A3", "gain_pilot3"), ("A4", "gain_pilot4")):
        reg = {l["loop"]: l for l in json.loads((RESULTS / "gpu" / f"gain_{prefix}_lcb.json").read_text())["loops"]}
        new = {l["loop"]: l for l in json.loads((RESULTS / "sens_lcb2048" / f"gain_{prefix}_lcb.json").read_text())["loops"]}
        for loop in sorted(reg):
            gate, seed = loop.replace("metaharness_", "").split("_s")
            rows.append(f"{phase} & {gate} & {seed} & {text(reg[loop])} & {text(new[loop])} \\\\")
    rows += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (OUT / "sensitivity.tex").write_text("\n".join(rows) + "\n")


def worked_example() -> None:
    """The whole lattice of the evolved BIRD B3 lineage: v(S) for every product on both splits, and every attribution
    the paper uses (Shapley, leave-one-out, leave-one-in, pairwise interaction, Mobius dividends), on held-out cells."""
    from editlattice.attribution import leave_one_in, leave_one_out, mobius, shapley, shapley_interaction
    labels = {0: "U0 SQL guidance", 1: "U1 docstring", 2: "U2 table descriptions"}
    values, cost = {}, {}
    for split in ("search", "heldout"):
        data = json.loads((RESULTS / "lattices" / f"phaseA3_bird_metaharness_B3_s1_stmt_{split}.json").read_text())
        totals: dict = {}
        for cell in data["cells"]:
            for product, outcome in cell["outcomes"].items():
                totals[product] = totals.get(product, 0.0) + outcome
        values[split] = {frozenset(int(u) for u in k.split(",") if u): v / len(data["cells"]) for k, v in totals.items()}
        cost[split] = (len(data["cells"]), sum(len(c["executions"]) for c in data["cells"]), len(totals) * len(data["cells"]))
    players = sorted(labels)
    def name(subset):
        return "seed" if not subset else ", ".join(f"U{u}" for u in sorted(subset))
    def pts(x):
        return f"{100 * x:+.1f}".replace("-", "$-$")
    def num(value):
        return f"{value:,}".replace(",", "{,}")
    rows = ["\\begin{table}[ht]", "\\centering",
            "\\caption{Worked example: the complete lattice of the second evolved BIRD lineage (U0 SQL guidance in the system "
            "prompt, U1 a docstring, U2 a description of each table in the schema text). Accuracy of every product on the "
            f"{cost['search'][0]} search cells and the {cost['heldout'][0]} held-out cells, and the attributions on held-out "
            f"cells, in points. The search lattice took {num(cost['search'][1])} executions and the held-out lattice "
            f"{num(cost['heldout'][1])}, against {num(cost['search'][2])} and {num(cost['heldout'][2])} under brute force.}}",
            "\\label{tab:worked}", "\\small", "\\begin{tabular}{lrr@{\\hspace{2em}}lrrr}", "\\toprule",
            "Product & search & held-out & Unit & Shapley & leave-one-out & leave-one-in \\\\", "\\midrule"]
    held = values["heldout"]
    phi, loo, loi = shapley(held, players), leave_one_out(held, players), leave_one_in(held, players)
    products = sorted(held, key=lambda s: (len(s), sorted(s)))
    right = [f"{labels[u]} & {pts(phi[u])} & {pts(loo[u])} & {pts(loi[u])}" for u in players]
    inter = shapley_interaction(held, players)
    right += ["", "Pair & interaction & & "] + [f"U{a}, U{b} & {pts(v)} & &" for (a, b), v in sorted(inter.items())]
    for i, subset in enumerate(products):
        left = f"{name(subset)} & {100 * values['search'][subset]:.1f}\\% & {100 * held[subset]:.1f}\\%"
        rows.append(left + " & " + (right[i] if i < len(right) else "& & &") + " \\\\")
    dividends = mobius(held, players)
    rows[2] = rows[2][:-1] + (" M\\\"obius dividends on held-out cells, in points: " +
               "; ".join(f"{name(s)} {pts(v)}" for s, v in sorted(dividends.items(), key=lambda kv: (len(kv[0]), sorted(kv[0]))) if s) + ".}")
    rows += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    # The Shapley values add up to the gain of the full harness on the same cells (efficiency), and each equals the
    # unit's own dividend plus an equal share of every joint dividend it belongs to (as the appendix text explains).
    assert abs(sum(phi.values()) - (held[frozenset(players)] - held[frozenset()])) < 1e-12
    for u in players:
        assert abs(phi[u] - sum(v / len(s) for s, v in dividends.items() if u in s)) < 1e-12
    # Stated in words: U1 changes nothing; U0 hurts alone, helps next to U2, so leave-one-out and leave-one-in disagree.
    assert phi[1] == loo[1] == loi[1] == 0 and loo[0] > 0 > loi[0] and inter[(0, 2)] > 0
    (OUT / "worked_example.tex").write_text("\n".join(rows) + "\n")


def predictions_table() -> None:
    """Every prediction fixed in notes/preregistration.md, with its outcome computed here from the stored results.
    The expected outcome of each row is asserted, so the table cannot drift from the data."""
    from editlattice.attribution import mobius
    pts = points
    def ci(e):
        return f"{pts(e['value'])} [{pts(e['low'])}, {pts(e['high'])}]"
    def num(value):
        return f"{value:,}".replace(",", "{,}")
    rows, outcomes = [], []
    def add(pid, text, held, evidence, expected):
        assert held == expected, (pid, held)
        outcomes.append(held)
        rows.append(f"{pid} & {text} & {'held' if held else 'failed'} & {evidence} \\\\")

    # Planted lineage (addendum 2026-10-27). Exactness and reads from the stored tree lattices, values from the report.
    report = json.loads((RESULTS / "report_planted_stmt.json").read_text())["rq3"]
    splits = ("search", "heldout")
    lattice, values = {}, {}
    for split in splits:
        lattice[split] = json.loads((RESULTS / f"planted_tree_stmt_{split}.json").read_text())["cells"]
        totals: dict = {}
        for cell in lattice[split]:
            for product, outcome in cell["outcomes"].items():
                totals[product] = totals.get(product, 0.0) + outcome
        values[split] = {frozenset(int(u) for u in k.split(",") if u): v / len(lattice[split]) for k, v in totals.items()}
    checked = sum(c["n_checked"] for s in splits for c in lattice[s])
    mismatches = sum(len(c["mismatches"]) for s in splits for c in lattice[s])
    add("P1", "planted: 0 mismatches", mismatches == 0, f"{mismatches} of {num(checked)} brute-force checks", True)
    runs = {s: [e for c in lattice[s] for e in c["executions"]] for s in splits}
    reads = {s: {u: sum(u in e["reads"] for e in runs[s]) for u in range(4)} for s in splits}
    dividend = max(abs(v) for s in splits for k, v in mobius(values[s], [0, 1, 2, 3]).items() if 3 in k)
    p2 = dividend < 1e-15 and all(reads[s][3] == 0 and report[s]["shapley"]["3"]["value"] == 0 and
                                  all(e["value"] == 0 for k, e in report[s]["interaction"].items() if "3" in k.split(","))
                                  for s in splits)
    add("P2", "planted: no run ever calls the helper U3, so its Shapley value, interactions and M\\\"obius terms are 0", p2,
        f"read by 0 of {num(len(runs['search']))} and 0 of {num(len(runs['heldout']))} runs; dividends below $10^{{-15}}$", True)
    add("P3", "planted: I(U0,U2) $>0$", all(report[s]["interaction"]["0,2"]["value"] > 0 for s in splits),
        f"{ci(report['search']['interaction']['0,2'])} search, {ci(report['heldout']['interaction']['0,2'])} held-out", True)
    add("P4", "planted: $|$LOI(U2)$|<$ LOO(U2)", all(abs(report[s]["loi"]["2"]) < report[s]["loo"]["2"] for s in splits),
        f"{pts(report['search']['loi']['2'])} vs {pts(report['search']['loo']['2'])} search; "
        f"{pts(report['heldout']['loi']['2'])} vs {pts(report['heldout']['loo']['2'])} held-out", True)
    add("P5", "planted: Shapley(U1) $\\le 0$; fewer runs read U1 than U2",
        all(report[s]["shapley"]["1"]["value"] <= 0 and reads[s][1] < reads[s][2] for s in splits),
        f"{pts(report['search']['shapley']['1']['value'])} and {pts(report['heldout']['shapley']['1']['value'])}; "
        f"reads {reads['search'][1]} vs {reads['search'][2]} and {reads['heldout'][1]} vs {reads['heldout'][2]}", True)
    add("P6", "planted: I(U1,U2) $\\ge 0$", all(report[s]["interaction"]["1,2"]["value"] >= 0 for s in splits),
        f"{pts(report['search']['interaction']['1,2']['value'])} search, {pts(report['heldout']['interaction']['1,2']['value'])} held-out", True)
    rows.append("\\midrule")

    # Synthetic scaling study (addendum 2026-10-27, D11).
    scale = json.loads((RESULTS / "scaling.json").read_text())["rows"]
    add("S1", "scaling: 0 mismatches", sum(r["mismatches"] for r in scale) == 0,
        f"0 of {num(sum(r['checked'] for r in scale))} checks over {len(scale)} configurations", True)
    add("S2", "scaling: at most $2^w$ executions per cell, and $w\\le L$",
        all(r["max_leaves_per_cell"] <= 2 ** r["read_width"] and r["read_width"] <= r["length"] for r in scale),
        f"all {len(scale)} configurations", True)
    sparse = [r for r in scale if r["length"] < r["m"]]
    span = {L: (min(r["executions"] for r in sparse if r["length"] == L), max(r["executions"] for r in sparse if r["length"] == L))
            for L in (2, 4, 8)}
    add("S3", "scaling: executions grow with $L$, not with $m$", span[2][1] < span[4][0] and span[4][1] < span[8][0],
        "; ".join(f"$L={L}$: {num(a)} to {num(b)}" for L, (a, b) in span.items()) + f" executions on 48 cells, for every $m$ up to {max(r['m'] for r in sparse)}", True)
    top, base = max(r["m"] for r in scale), 16
    def mean_static(m):
        times = [r["t_static"] for r in sparse if r["m"] == m]
        return sum(times) / len(times)
    exponent = math.log(mean_static(top) / mean_static(base)) / math.log(top / base)
    add("S4", "scaling: static analysis time roughly linear in $m$", exponent < 1.5,
        f"grows as $m^{{{exponent:.1f}}}$ between $m={base}$ and $m={top}$", False)
    diff = max(max(r.get("max_diff_shapley", 0), r.get("max_diff_interaction", 0)) for r in scale)
    mantissa, power = f"{diff:.1e}".split("e")
    assert sum("max_diff_shapley" in r for r in scale) == sum(r["m"] <= 10 for r in scale)
    add("S5", "scaling: closed forms equal the exhaustive formulas to $10^{-12}$", diff <= 1e-12, f"largest difference ${mantissa}\\cdot10^{{{int(power)}}}$", True)
    slow = max((r for r in scale if r.get("t_milp_best") is not None), key=lambda r: r["t_milp_best"])
    add("S6", "scaling: MILP optimum within seconds for every $m$", slow["t_milp_best"] < 60,
        f"{slow['t_milp_best'] / 60:.0f} minutes at $m=L={slow['m']}$; enumeration used for larger dense rows", False)
    rows.append("\\midrule")

    # Constructed lineages (addendum 2026-10-29).
    base_c = RESULTS / "phaseC"
    gain = {b: json.loads((base_c / f"gain_lattice_constructed_{b}_stmt_heldout.json").read_text()) for b in BENCH}
    rep = {b: json.loads((base_c / f"report_constructed_{b}_stmt.json").read_text())["rq3"]["heldout"] for b in BENCH}
    distill = {b: json.loads((base_c / f"report_constructed_{b}_stmt.json").read_text())["rq4"] for b in BENCH}
    def gtext(b):
        g = gain[b]
        return f"{BENCH[b]} {pts(g['gain'])} [{pts(g['low'])}, {pts(g['high'])}]"
    assert gain["bird"]["real_gain"] and not gain["gsmhard"]["real_gain"] and not gain["lcb"]["real_gain"]
    # Stated in the limitations: under the evolution setting (AWQ, T=0.7), the planted lineage has no real gain either.
    assert not json.loads((RESULTS / "planted_gain" / "gain_lattice_planted_gpu_stmt_heldout.json").read_text())["real_gain"]
    add("C1", "constructed: GSM-Hard and LiveCodeBench lineages are real-gain", gain["gsmhard"]["real_gain"] and gain["lcb"]["real_gain"],
        f"{gtext('gsmhard')}; {gtext('lcb')}; BIRD (left open) real", False)
    shap = {b: {u: e["value"] for u, e in rep[b]["shapley"].items()} for b in BENCH}
    add("C2", "constructed: the technique unit U2 has the largest Shapley value", all(max(s, key=s.get) == "2" for s in shap.values()),
        "; ".join(f"{BENCH[b]} {pts(shap[b]['2'])}" for b in BENCH), True)
    add("C3", "constructed: GSM-Hard $|$Shapley(U1)$|\\le 1$ point", abs(shap["gsmhard"]["1"]) <= 0.01, "exactly 0", True)
    assert shap["gsmhard"]["1"] == 0
    inter = rep["lcb"]["interaction"]
    add("C4", "constructed: in LiveCodeBench, neither I(U0,U2) nor I(U1,U2) has an interval above 0",
        all(not rep["lcb"]["interacts"][k] and inter[k]["low"] <= 0 for k in ("0,2", "1,2")),
        f"{ci(inter['0,2'])}; {ci(inter['1,2'])}", True)
    real = [b for b in BENCH if gain[b]["real_gain"]]
    kept = {b: next(r for r in distill[b] if r["method"] == "exact" and abs(r["delta"] - 0.02) < 1e-9) for b in real}
    add("C5", "constructed: the distilled product at $\\delta=2$ points keeps U2 (real-gain lineages)",
        all(2 in kept[b]["product"] for b in real),
        "; ".join(f"{BENCH[b]}: \\{{" + ",".join(f"U{u}" for u in kept[b]["product"]) + "\\}" for b in real), True)
    transfer = json.loads((RESULTS / "transfer.json").read_text())
    registered = ("Qwen3-4B AWQ T1.0", "Qwen3-4B bf16 T0.7")
    every = all(row["top_unit"] == 2 for b in BENCH for row in transfer[f"constructed_{b}"]["settings"].values())
    assert every
    add("C6", "constructed: U2 ranks first under every transfer setting",
        all(transfer[f"constructed_{b}"]["settings"][s]["top_unit"] == 2 for b in BENCH for s in registered),
        f"all {len(BENCH) * len(registered)} (lineage, registered setting) pairs, and both exploratory Llama builds", True)
    rows.append("\\midrule")

    # Phase A4 (addendum 2026-11-02).
    totals = [t for t in json.loads((RESULTS / "pilot_summary.json").read_text())["totals"] if t["phase"] == "A4"]
    real_a4 = sum(t["real_gain_loops"] for t in totals)
    add("A4", "Phase A4 has at least one real-gain loop", real_a4 >= 1, f"{real_a4} of {sum(t['loops'] for t in totals)} loops", False)
    rows.append("\\midrule")

    # Branched lineages (addendum 2026-11-09, RQ1c).
    branched = json.loads((RESULTS / "branched.json").read_text())
    names = {"bird": "BIRD", "lcb": "LiveCodeBench"}
    width = {"bird": 4, "lcb": 3}
    add("R1", "branched: the exhaustive audits find no mismatch", all(b["audit"]["mismatches"] == 0 for b in branched.values()),
        "; ".join(f"{names[k]} {b['audit']['mismatches']} of {num(b['audit']['checked'])}" for k, b in branched.items()), True)
    add("R2", "branched: read width at most 4 (BIRD) and 3 (LiveCodeBench), so at most 16 and 8 executions per cell",
        all(b[s]["max_read_width"] <= width[k] and b[s]["max_leaves_per_cell"] <= 2 ** width[k] for k, b in branched.items() for s in ("search", "heldout")),
        "; ".join(f"{names[k]} width {max(b[s]['max_read_width'] for s in ('search', 'heldout'))}, "
                  f"at most {max(b[s]['max_leaves_per_cell'] for s in ('search', 'heldout'))} per cell" for k, b in branched.items()), True)
    add("R3", "branched: fewer than 1\\% of the brute-force executions on held-out cells",
        all(b["heldout"]["executions"] < 0.01 * b["heldout"]["brute_force"] for b in branched.values()),
        "; ".join(f"{names[k]} {num(b['heldout']['executions'])} of {num(b['heldout']['brute_force'])}" for k, b in branched.items()), True)
    add("R4", "branched: replayed minus fresh accuracy has an interval containing 0",
        all(b["fresh_audit"]["low"] <= 0 <= b["fresh_audit"]["high"] for b in branched.values()),
        "; ".join(f"{names[k]} {pts(b['fresh_audit']['difference'])} [{pts(b['fresh_audit']['low'])}, {pts(b['fresh_audit']['high'])}]"
                  for k, b in branched.items()), True)
    unread = {k: b["unread_units"] for k, b in branched.items()}
    add("R5", "branched: every unit that no held-out run reads has Shapley value, interactions and M\\\"obius terms 0",
        all(b["unread_max_dividend"] < 1e-12 for b in branched.values()),
        "; ".join(f"{names[k]}: {len(u)} such units" + (", every dividend exactly 0" if u and branched[k]["unread_max_dividend"] == 0 else "") for k, u in unread.items()), True)
    rows.append("\\midrule")

    # Pooled lattices (addendum 2026-11-09).
    pools = json.loads((RESULTS / "pool_audits.json").read_text())
    add("Q1", "pooled lattices: 0 mismatches in every brute-force audit", all(a["mismatches"] == 0 for a in pools.values()),
        f"{sum(a['mismatches'] for a in pools.values())} of {num(sum(a['checked'] for a in pools.values()))} checks over {len(pools)} pools", True)

    table = ["{\\small\\setlength{\\LTcapwidth}{\\textwidth}",
             "\\begin{longtable}{l>{\\raggedright\\arraybackslash}p{0.34\\textwidth}l>{\\raggedright\\arraybackslash}p{0.42\\textwidth}}",
             f"\\caption{{Registered predictions and what happened ({sum(outcomes)} of {len(outcomes)} held). Evidence is in "
             "points where it is a value; intervals are 95\\% bootstrap intervals. P: planted lineage, both splits; S: synthetic "
             "scaling study; C: constructed lineages, held-out cells; A4: the reasoning proposer at 8B; R: branched lineages (RQ1c); Q: pooled lattices.}",
             "\\label{tab:predictions}\\\\", "\\toprule", "ID & Prediction & Outcome & Evidence \\\\", "\\midrule", "\\endfirsthead",
             "\\toprule", "ID & Prediction & Outcome & Evidence \\\\", "\\midrule", "\\endhead", "\\bottomrule", "\\endfoot"]
    table += rows + ["\\end{longtable}", "}"]
    (OUT / "predictions.tex").write_text("\n".join(table) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prompt()
    seeds()
    loops_table()
    lineages_table()
    settings_table()
    sensitivity_table()
    worked_example()
    predictions_table()
    print("appendix material written to", OUT)


if __name__ == "__main__":
    main()
