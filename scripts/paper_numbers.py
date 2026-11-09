"""Every number and table in the paper, generated from results/*.json (never typed by hand).

Usage: python scripts/paper_numbers.py
Writes paper/macros.tex and paper/tables/*.tex for the results that exist.
"""

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PAPER = ROOT / "paper"


def macro(name: str, value) -> str:
    """One \\newcommand."""
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def fmt(value: float, digits: int = 3) -> str:
    """Signed or plain number; LaTeX minus."""
    return f"{value:.{digits}f}".replace("-", "$-$")


def signed(value: float, digits: int = 3) -> str:
    """Always signed."""
    return f"{value:+.{digits}f}".replace("-", "$-$")


def rq1_prior(lines: list[str]) -> None:
    """RQ1 on prior (null-gain) lineages: exactness and executions."""
    path = RESULTS / "rq1_prior_summary.json"
    if not path.exists():
        return
    rows = json.loads(path.read_text())["rows"]
    name = sum(r["name_level"]["executions"] for r in rows)
    statement = sum(r["statement_level"]["executions"] for r in rows if "statement_level" in r)
    brute = sum(r["name_level"]["brute_force"] for r in rows)
    checked = sum(r["name_level"]["checked"] + r.get("statement_level", {}).get("checked", 0) for r in rows)
    mismatches = sum(r["name_level"]["mismatches"] + r.get("statement_level", {}).get("mismatches", 0) for r in rows)
    lines += [macro("PriorLineages", len(rows)), macro("PriorChecked", f"{checked:,}".replace(",", "{,}")),
              macro("PriorMismatches", mismatches), macro("PriorBrute", f"{brute:,}".replace(",", "{,}")),
              macro("PriorExecName", f"{name:,}".replace(",", "{,}")), macro("PriorExecStmt", f"{statement:,}".replace(",", "{,}")),
              macro("PriorSaveName", f"{100 * (1 - name / brute):.0f}\\%"), macro("PriorSaveStmt", f"{100 * (1 - statement / brute):.0f}\\%")]


def planted(lines: list[str]) -> None:
    """Planted-structure validation: Shapley values, interactions, LOO/LOI."""
    path = RESULTS / "report_planted_stmt.json"
    if not path.exists():
        return
    report = json.loads(path.read_text())
    names = {"0": "A", "1": "B", "2": "C", "3": "D"}
    for split, tag in (("search", "S"), ("heldout", "H")):
        block = report["rq3"][split]
        lines += [macro(f"Planted{tag}Seed", f"{100 * block['seed']:.1f}\\%"), macro(f"Planted{tag}Full", f"{100 * block['full']:.1f}\\%")]
        for unit, entry in block["shapley"].items():
            u = names[unit]
            lines += [macro(f"Planted{tag}Shap{u}", pct(entry["value"])),
                      macro(f"Planted{tag}Shap{u}CI", f"[{pct(entry['low'])}, {pct(entry['high'])}]"),
                      macro(f"Planted{tag}Loo{u}", pct(block["loo"][unit])), macro(f"Planted{tag}Loi{u}", pct(block["loi"][unit]))]
        for pair, entry in block["interaction"].items():
            a, b = pair.split(",")
            lines += [macro(f"Planted{tag}Int{names[a]}{names[b]}", pct(entry["value"])),
                      macro(f"Planted{tag}Int{names[a]}{names[b]}CI", f"[{pct(entry['low'])}, {pct(entry['high'])}]")]


def rq2(lines: list[str]) -> None:
    """RQ2 table: MAE per estimator, regime and budget."""
    path = RESULTS / "rq2_planted_stmt_heldout.json"
    if not path.exists():
        return
    data = json.loads(path.read_text())
    rows = data["rows"]
    budgets = data["budgets"]
    estimators = [("permutation", "Permutation sampling"), ("kernelshap", "KernelSHAP"), ("plugin_exact", "Plug-in exact"),
                  ("loo", "Leave-one-out"), ("loi", "Leave-one-in")]
    table = ["\\begin{table}[t]", "\\centering", "\\caption{RQ2: mean absolute error of Shapley estimates against the exact lattice "
             "(planted lineage, held-out cells, 20 repetitions), in points, coupled (replayed streams) / independent streams, at "
             "equal budgets of task-runs.}", "\\label{tab:rq2}", "\\small", "\\begin{tabular}{l" + "c" * len(budgets) + "}", "\\toprule",
             "Estimator & " + " & ".join(f"{b:,}".replace(",", "{,}") for b in budgets) + " \\\\", "\\midrule"]
    by = {(r["estimator"], r["regime"], r["budget"]): r["mae"] for r in rows}
    for key, label in estimators:
        cells = [f"{100 * by[(key, 'coupled', b)]:.1f} / {100 * by[(key, 'independent', b)]:.1f}" for b in budgets]
        table.append(f"{label} & " + " & ".join(cells) + " \\\\")
    table += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (PAPER / "tables").mkdir(parents=True, exist_ok=True)
    (PAPER / "tables" / "rq2.tex").write_text("\n".join(table) + "\n")
    top = budgets[-1]
    lines += [macro("RqTwoTopBudget", f"{top:,}".replace(",", "{,}")),
              macro("RqTwoPermCoupled", f"{100 * by[('permutation', 'coupled', top)]:.1f}"), macro("RqTwoPermInd", f"{100 * by[('permutation', 'independent', top)]:.1f}")]
    # Error reduction from coupling at the full budget, over all lineages with an RQ2 file (quoted in the text).
    reductions = []
    for other in [path, *sorted((RESULTS / "phaseC").glob("rq2_*_stmt_heldout.json"))]:
        other_data = json.loads(other.read_text())
        other_by = {(r["estimator"], r["regime"], r["budget"]): r["mae"] for r in other_data["rows"]}
        full = other_data["budgets"][-1]
        reductions += [1 - other_by[(key, "coupled", full)] / other_by[(key, "independent", full)]
                       for key in ("permutation", "kernelshap", "plugin_exact")]
    lines += [macro("RqTwoRedMin", f"{100 * min(reductions):.0f}\\%"), macro("RqTwoRedMax", f"{100 * max(reductions):.0f}\\%")]
    audit = RESULTS / "a2_planted_stmt_heldout.json"
    if audit.exists():
        summary = json.loads(audit.read_text())["summary"]
        lines += [macro("AtwoPairs", f"{summary['pairs']:,}".replace(",", "{,}")), macro("AtwoReplay", f"{100 * summary['replayed_accuracy']:.1f}\\%"),
                  macro("AtwoFresh", f"{100 * summary['fresh_accuracy']:.1f}\\%"), macro("AtwoSignificant", summary["products_with_p_below_0.05"]),
                  macro("AtwoProducts", summary["products"])]


def scaling(lines: list[str]) -> None:
    """RQ1b: tree exploration cost, exactness and solver times on the synthetic family."""
    path = RESULTS / "scaling.json"
    if not path.exists():
        return
    rows = json.loads(path.read_text())["rows"]
    table = ["\\begin{table}[t]", "\\centering",
             "\\caption{RQ1b: tree exploration on a synthetic family with $m$ units and routes of $L$ steps, over "
             "48 cells. Columns: executions, brute-force runs ($48\\cdot 2^m$) and read width $w$. Times are in seconds, "
             "for the static analysis, the closed-form attribution and the exact optimisation. The optimisation uses the "
             "node MILP, or enumeration ($^e$) for the dense rows with $m>10$. Mismatches against plain runs: 0 in every row.}",
             "\\label{tab:scaling}", "\\small", "\\begin{tabular}{rrrrrrrr}", "\\toprule",
             "$m$ & $L$ & executions & brute force & $w$ & static & attribution & optimum \\\\", "\\midrule"]
    for r in rows:
        brute = r["brute_force_runs"]
        brute_text = f"{brute:,}".replace(",", "{,}") if brute < 10 ** 7 else f"$48\\cdot2^{{{r['m']}}}$"
        optimum = r.get("t_milp_best")
        optimum_text = f"{optimum:.2f}" if optimum is not None and (r["length"] < r["m"] or r["m"] <= 10) else f"{r['t_enumerate_both']:.2f}$^e$"
        executions = f"{r['executions']:,}".replace(",", "{,}")
        table.append(f"{r['m']} & {r['length']} & {executions} & {brute_text} & {r['read_width']} & "
                     f"{r['t_static']:.1f} & {r['t_attribution']:.2f} & {optimum_text} \\\\")
    table += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (PAPER / "tables").mkdir(parents=True, exist_ok=True)
    (PAPER / "tables" / "scaling.tex").write_text("\n".join(table) + "\n")
    checked = sum(r["checked"] for r in rows)
    max_diff = max(max(r.get("max_diff_shapley", 0), r.get("max_diff_interaction", 0)) for r in rows)
    mantissa, power = f"{max_diff:.1e}".split("e")
    mismatches = sum(r["mismatches"] for r in rows)
    lines += [macro("ScaleConfigs", len(rows)), macro("ScaleChecked", f"{checked:,}".replace(",", "{,}")), macro("ScaleMismatches", mismatches),
              macro("ScaleMaxM", max(r["m"] for r in rows)),
              macro("ScaleMaxDiff", f"${mantissa}\\cdot10^{{{int(power)}}}$")]
    # Numbers quoted in the RQ1b text.
    four = [r["executions"] / 48 for r in rows if r["length"] == 4 and r["m"] >= 8]
    top_m = max(r["m"] for r in rows)
    static_top = [r["t_static"] for r in rows if r["m"] == top_m]
    static_16 = [r["t_static"] for r in rows if r["m"] == 16 and r["length"] < r["m"]]
    exponent = math.log(sum(static_top) / len(static_top) / (sum(static_16) / len(static_16))) / math.log(top_m / 16)
    sparse_milp = [r["t_milp_best"] for r in rows if r["read_width"] <= 8 and r.get("t_milp_best") is not None]
    dense_milp = max((r for r in rows if r["length"] == r["m"] and r.get("t_milp_best") is not None), key=lambda r: r["m"])
    lines += [macro("ScaleFourLow", f"{min(four):.0f}"), macro("ScaleFourHigh", f"{max(four):.0f}"),
              macro("ScaleAttrMaxTwelve", f"{max(r['t_attribution'] for r in rows if r['read_width'] <= 12):.2f}"),
              macro("ScaleStaticExponent", f"{exponent:.1f}"),
              macro("ScaleStaticTopMin", f"{min(static_top) / 60:.0f}"), macro("ScaleStaticTopMax", f"{max(static_top) / 60:.1f}"),
              macro("ScaleMilpSparseMax", f"{max(sparse_milp):.0f}"),
              macro("ScaleMilpDenseM", dense_milp["m"]), macro("ScaleMilpDenseMin", f"{dense_milp['t_milp_best'] / 60:.0f}")]


def pilots(lines: list[str]) -> None:
    """Gain pilots (Phases A, A2, A3 and A4): per phase and benchmark, from results/pilot_summary.json."""
    path = RESULTS / "pilot_summary.json"
    if not path.exists():
        return
    totals = json.loads(path.read_text())["totals"]
    names = {"gsmhard": "GSM-Hard", "lcb": "LiveCodeBench", "bird": "BIRD"}
    table = ["\\begin{table}[t]", "\\centering",
             "\\caption{Gain pilots: eight loops per phase and benchmark (gates B1 and B3, four seeds). Valid: proposals "
             "that applied to the harness; accepted: edits the gate kept; units: net change units of the final harnesses; "
             "real gain: held-out gain with 95\\% CI above 0 and at least 3 points (pre-registered). Exactly 0: final harness "
             "and seed agree on every held-out cell. Best: largest held-out gain [95\\% CI], in points.}",
             "\\label{tab:pilots}", "\\small", "\\resizebox{\\textwidth}{!}{\\begin{tabular}{llrrrrrrrl}", "\\toprule",
             "Phase & Benchmark & proposals & valid & accepted & units & real gain & harmful & exactly 0 & best \\\\", "\\midrule"]
    for t in totals:
        best = t["best_gain"]
        best_text = "n/a" if best is None else f"{100 * best['gain']:+.1f} [{100 * best['low']:+.1f}, {100 * best['high']:+.1f}]".replace("-", "$-$")
        table.append(f"{t['phase']} & {names[t['benchmark']]} & {t['proposals']} & {t['valid']} & {t['accepted']} & {t['net_units']} & "
                     f"{t['real_gain_loops']}/{t['loops']} & {t['harmful_loops']} & {t['exactly_zero_loops']} & {best_text} \\\\")
    table += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    (PAPER / "tables").mkdir(parents=True, exist_ok=True)
    (PAPER / "tables" / "pilots.tex").write_text("\n".join(table) + "\n")
    loops = sum(t["loops"] for t in totals)
    lines += [macro("PilotLoops", loops), macro("PilotRealGain", sum(t["real_gain_loops"] for t in totals)),
              macro("PilotProposals", f"{sum(t['proposals'] for t in totals):,}".replace(",", "{,}")),
              macro("PilotAccepted", sum(t["accepted"] for t in totals)),
              macro("PilotHarmful", sum(t["harmful_loops"] for t in totals)), macro("PilotExactZero", sum(t["exactly_zero_loops"] for t in totals)),
              macro("PilotNetUnits", sum(t["net_units"] for t in totals)),
              macro("PilotProposerPhases", len({t["phase"] for t in totals}))]
    from scipy.stats import beta
    real = sum(t["real_gain_loops"] for t in totals)
    upper = beta.ppf(0.95, real + 1, loops - real) if real < loops else 1.0  # one-sided 95% Clopper-Pearson
    rows = json.loads(path.read_text())["loops"]
    widths = {}
    for row in rows:
        if row["gain"] is not None and row["high"] != row["low"]:
            widths.setdefault(row["benchmark"], []).append((row["high"] - row["low"]) / 2)
    names = {"gsmhard": "Gsm", "lcb": "Lcb", "bird": "Bird"}
    lines += [macro("PilotRealGainUpper", f"{100 * upper:.1f}\\%")]
    # Lineage sizes, quoted in the appendix of reader questions.
    lines += [macro("PilotMaxUnits", max(r["net_units"] for r in rows)), macro("PilotNoUnits", sum(r["net_units"] == 0 for r in rows))]
    low2 = beta.ppf(0.025, real, loops - real + 1) if real > 0 else 0.0  # two-sided 95% Clopper-Pearson
    high2 = beta.ppf(0.975, real + 1, loops - real) if real < loops else 1.0
    lines += [macro("PilotRateLow", f"{100 * low2:.2g}\\%"), macro("PilotRateHigh", f"{100 * high2:.1f}\\%")]
    for bench, values in widths.items():
        lines += [macro(f"PilotHalfWidth{names[bench]}", f"{100 * sum(values) / len(values):.1f}")]
        # Stated in the text: gains of three to four points could go undetected on LiveCodeBench and BIRD, not on GSM-Hard.
        assert (sum(values) / len(values) > 0.03) == (bench != "gsmhard") and sum(values) / len(values) < 0.04
    for phase, tag in (("A", "One"), ("A2", "Two"), ("A3", "Three"), ("A4", "Four")):
        sub = [t for t in totals if t["phase"] == phase]
        if not sub:
            continue
        lines += [macro(f"Pilot{tag}Loops", sum(t["loops"] for t in sub)), macro(f"Pilot{tag}RealGain", sum(t["real_gain_loops"] for t in sub))]
        lines += [macro(f"Pilot{tag}Valid", f"{100 * sum(t['valid'] for t in sub) / sum(t['proposals'] for t in sub):.0f}\\%")]


def pct(value: float, digits: int = 1) -> str:
    """Points (x100), signed, LaTeX minus."""
    return f"{100 * value:+.{digits}f}".replace("-", "$-$")


def null_group(lines: list[str]) -> None:
    """Phase B on every evolved lineage with accepted edits (results/null_summary.json)."""
    path = RESULTS / "null_summary.json"
    if not path.exists():
        return
    data = json.loads(path.read_text())
    summary = data["summary"]
    def num(value):
        return f"{value:,}".replace(",", "{,}")
    # The text says the overfitting units are among the units without a detectable held-out effect.
    assert not any(u["overfitting"] and u["matters"] for u in json.loads((RESULTS / "null_summary.json").read_text())["units"])
    lines += [macro("NullLineages", summary["lineages"]), macro("NullUnits", summary["units"]), macro("NullInert", summary["inert"]),
              macro("NullInertPct", f"{100 * summary['inert'] / summary['units']:.0f}\\%"), macro("NullMatters", summary["matters"]), macro("NullNoEffect", summary["units"] - summary["matters"]),
              macro("NullNoEffectPct", f"{100 * (summary['units'] - summary['matters']) / summary['units']:.0f}\\%"),
              macro("NullHarmful", summary["harmful"]), macro("NullOverfit", summary["overfitting"]),
              macro("NullAuditChecked", num(summary["audit_checked"])), macro("NullAuditMismatches", summary["audit_mismatches"]),
              macro("NullExec", num(summary["lattice_executions"])), macro("NullBrute", num(summary["lattice_brute_force"])),
              macro("NullFreshPairs", num(summary["fresh_pairs"])), macro("NullFreshProducts", summary["fresh_products"]),
              macro("NullFreshSignificant", summary["fresh_significant"]),
              macro("NullFreshReplayed", f"{100 * summary['fresh_replayed']:.1f}\\%"), macro("NullFreshFresh", f"{100 * summary['fresh_fresh']:.1f}\\%")]
    names = {"gsmhard": "GSM-Hard", "lcb": "LiveCodeBench", "bird": "BIRD"}
    sizes = json.loads((RESULTS / "pool_sizes.json").read_text())
    run = {name.replace("pooled", "") for name in data["pools"]}
    not_run = sorted((k for k in sizes if k not in run), key=lambda k: (len(k.split("_")[0]), k))
    # Every pool not run exceeds the registered size limit (500 products; Phase A LCB predates the rule).
    assert all(sizes[k]["products"] > 500 for k in not_run), not_run
    skipped = " and ".join(f"Phase {k.split('_')[0]} {names[k.split('_')[1]]} ({num(sizes[k]['products'])} products)" for k in not_run)
    lines += [macro("PoolALcbProducts", num(sizes["A_lcb"]["products"])), macro("PoolALcbUnits", sizes["A_lcb"]["units"]),
              macro("PoolsNotRun", len(not_run)),
              # brute force over the 60 search and 85 held-out LiveCodeBench tasks, four replicates each
              macro("PoolALcbBrute", num(sizes["A_lcb"]["products"] * (60 + 85) * 4)),
              macro("PoolAFourBirdProducts", num(sizes["A4_bird"]["products"]))]
    table = ["\\begin{table}[t]", "\\centering",
             "\\caption{Recombination (RQ5) over every loop of a phase on a benchmark. Columns: conflict-free products of the "
             "pool, executions over search and held-out cells, the product chosen on search cells, and its held-out difference "
             "to the best single final harness. Differences are in points, with 95\\% task-bootstrap intervals. The registered "
             "certification (anytime-valid superiority test, $\\alpha=0.05$) succeeds for none. Where the chosen product is the "
             "seed, a positive difference means that the best single final harness was worse than the seed. Not run, because "
             f"they exceed the size limit: {skipped}.}}",
             "\\label{tab:pools}", "\\small", "\\begin{tabular}{llrrll}", "\\toprule",
             "Phase & Benchmark & products & executions & chosen & difference \\\\", "\\midrule"]
    for name, pool in sorted(data["pools"].items(), key=lambda kv: (len(kv[0].split("_")[0]), kv[0])):
        phase, bench = name.replace("pooled", "").split("_", 1)
        rq5 = pool["rq5"]
        executions = sum(pool[s]["executions"] for s in ("search", "heldout") if s in pool)
        products = pool["search"]["brute_force"] // 240 if "search" in pool else "n/a"
        assert products == sizes[f"{phase}_{bench}"]["products"], (name, products)
        chosen = "seed" if not rq5["best_pooled_product"] else "\\{" + ",".join(map(str, rq5["best_pooled_product"])) + "\\}"
        table.append(f"{phase} & {names[bench]} & {products} & {num(executions)} & {chosen} & "
                     f"{pct(rq5['difference'])} [{pct(rq5['ci'][0])}, {pct(rq5['ci'][1])}] \\\\")
    table += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (PAPER / "tables" / "pools.tex").write_text("\n".join(table) + "\n")


def constructed(lines: list[str]) -> None:
    """Phase C: gains, Shapley values with CIs, distillation, RQ2 (results/phaseC/)."""
    base = RESULTS / "phaseC"
    if not base.exists():
        return
    names = {"gsmhard": "GSM-Hard", "lcb": "LiveCodeBench", "bird": "BIRD"}
    table = ["\\begin{table}[t]", "\\centering",
             "\\caption{Constructed lineages (Phase C, registered before any lattice): held-out gain of the full lineage over the "
             "seed and held-out Shapley values with 95\\% CIs, in points (U0 prompt, U1 auxiliary change, U2 technique). "
             "$^\\ast$: interval excludes 0 after Holm correction. Distilled: exact smallest product within $\\delta=2$ points "
             "on search cells, and whether it was certified non-inferior on held-out cells.}",
             "\\label{tab:constructed}", "\\small", "\\resizebox{\\textwidth}{!}{\\begin{tabular}{lllllll}", "\\toprule",
             "Benchmark & gain & U0 & U1 & U2 & distilled & certified \\\\", "\\midrule"]
    for bench in ("gsmhard", "lcb", "bird"):
        gain = json.loads((base / f"gain_lattice_constructed_{bench}_stmt_heldout.json").read_text())
        report = json.loads((base / f"report_constructed_{bench}_stmt.json").read_text())
        held = report["rq3"]["heldout"]
        cells = []
        for unit in ("0", "1", "2"):
            entry = held["shapley"][unit]
            star = "$^\\ast$" if held["matters"][unit] else ""
            cells.append(f"{pct(entry['value'])}{star} [{pct(entry['low'])}, {pct(entry['high'])}]")
        distilled = next(row for row in report["rq4"] if row["method"] == "exact" and abs(row["delta"] - 0.02) < 1e-9)
        product = "seed" if not distilled["product"] else "\\{" + ",".join(f"U{u}" for u in distilled["product"]) + "\\}"
        table.append(f"{names[bench]} & {pct(gain['gain'])} [{pct(gain['low'])}, {pct(gain['high'])}] & " + " & ".join(cells) +
                     f" & {product} & {'yes' if distilled['certified'] else 'no'} \\\\")
        tag = {"gsmhard": "Gsm", "lcb": "Lcb", "bird": "Bird"}[bench]
        lines += [macro(f"Con{tag}Gain", pct(gain["gain"])), macro(f"Con{tag}GainCI", f"[{pct(gain['low'])}, {pct(gain['high'])}]"),
                  macro(f"Con{tag}Real", "yes" if gain["real_gain"] else "no")]
        rq2 = json.loads((base / f"rq2_constructed_{bench}_stmt_heldout.json").read_text())
        top = rq2["budgets"][-1]
        by = {(r["estimator"], r["regime"]): r["mae"] for r in rq2["rows"] if r["budget"] == top}
        for est, short in (("permutation", "Perm"), ("kernelshap", "Kernel"), ("plugin_exact", "Plugin")):
            lines += [macro(f"Con{tag}{short}Coupled", f"{100 * by[(est, 'coupled')]:.2f}"), macro(f"Con{tag}{short}Ind", f"{100 * by[(est, 'independent')]:.2f}")]
    table += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    (PAPER / "tables" / "constructed.tex").write_text("\n".join(table) + "\n")


def name_vs_statement(lines: list[str]) -> None:
    """RQ1: name-level vs statement-level reads on the GPU lineages (results/rq1_name_vs_stmt.json)."""
    path = RESULTS / "rq1_name_vs_stmt.json"
    if not path.exists():
        return
    summary = json.loads(path.read_text())["summary"]
    def num(value):
        return f"{value:,}".replace(",", "{,}")
    lines += [macro("NvsLineages", summary["lineages"]), macro("NvsProductCells", num(summary["product_cells"])),
              macro("NvsDiffering", summary["differing"]), macro("NvsDifferingLcb", summary["differing_lcb"]),
              macro("NvsProductCellsLcb", num(summary["product_cells_lcb"])),
              macro("NvsExecStmt", num(summary["executions_statement"])), macro("NvsExecName", num(summary["executions_name"]))]


def transfer(lines: list[str]) -> None:
    """RQ6 on the planted lineage (results/transfer.json, results/planted_gain/)."""
    path = RESULTS / "transfer.json"
    if not path.exists():
        return
    planted = json.loads(path.read_text())["planted"]["settings"]
    files = {"Qwen3-4B MLX 4-bit T0.7 (Mac)": "planted_stmt_heldout", "Llama-3.1-8B MLX 4-bit T0.7 (Mac)": "planted_llama_stmt_heldout",
             "Qwen3-4B AWQ T0.7": "planted_gpu_stmt_heldout", "Qwen3-4B AWQ T1.0": "planted_gpu_t1_stmt_heldout",
             "Qwen3-4B bf16 T0.7": "planted_gpu_bf16_stmt_heldout",
             "Llama-3.1-8B AWQ vLLM T0.7 (GPU, exploratory)": "planted_llama_awq_stmt_heldout"}
    labels = {"Qwen3-4B MLX 4-bit T0.7 (Mac)": "Qwen3-4B, 4-bit (MLX), $T{=}0.7$", "Llama-3.1-8B MLX 4-bit T0.7 (Mac)": "Llama-3.1-8B, 4-bit (MLX), $T{=}0.7$",
              "Qwen3-4B AWQ T0.7": "Qwen3-4B, AWQ, $T{=}0.7$", "Qwen3-4B AWQ T1.0": "Qwen3-4B, AWQ, $T{=}1.0$", "Qwen3-4B bf16 T0.7": "Qwen3-4B, bf16, $T{=}0.7$",
              "Llama-3.1-8B AWQ vLLM T0.7 (GPU, exploratory)": "Llama-3.1-8B, AWQ (vLLM), $T{=}0.7^\\ast$"}
    table = ["\\begin{table}[t]", "\\centering",
             "\\caption{Transfer (RQ6) on the planted lineage. Columns: held-out Shapley values in points for each solver "
             "setting, and the gain of the full lineage over the seed with its 95\\% CI. $^\\dagger$: a real gain under the "
             "registered criterion, applied post hoc. $^\\ast$: exploratory, not registered.}",
             "\\label{tab:transfer}", "\\small", "\\begin{tabular}{lrrrrl}", "\\toprule",
             "Solver & U0 & U1 & U2 & U3 & gain \\\\", "\\midrule"]
    for setting, row in planted.items():
        gain = json.loads((RESULTS / "planted_gain" / f"gain_lattice_{files[setting]}.json").read_text())
        mark = "$^\\dagger$" if gain["real_gain"] else ""
        table.append(f"{labels[setting]} & " + " & ".join(pct(row["shapley"][u]) for u in ("0", "1", "2", "3")) +
                     f" & {pct(gain['gain'])} [{pct(gain['low'])}, {pct(gain['high'])}]{mark} \\\\")
    table += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (PAPER / "tables" / "transfer.tex").write_text("\n".join(table) + "\n")
    # Quoted in the RQ6 text: the planted lineage's gain under the Qwen3-4B settings and under Llama.
    gains = {setting: json.loads((RESULTS / "planted_gain" / f"gain_lattice_{files[setting]}.json").read_text())["gain"] for setting in planted}
    qwen = [g for setting, g in gains.items() if setting.startswith("Qwen3")]
    llama = [g for setting, g in gains.items() if setting.startswith("Llama")]
    assert min(llama) > max(qwen)
    lines += [macro("TransPlantedQwenLow", f"{100 * min(qwen):.1f}"), macro("TransPlantedQwenHigh", f"{100 * max(qwen):.1f}"),
              macro("TransPlantedLlamaLow", f"{100 * min(llama):.1f}"), macro("TransPlantedLlamaHigh", f"{100 * max(llama):.1f}")]


def transfer_lineages(lines: list[str]) -> None:
    """RQ6 on the constructed lineages and the evolved real-gain lineages (results/transfer.json, results/transfer_gain/)."""
    path = RESULTS / "transfer.json"
    if not path.exists():
        return
    data = json.loads(path.read_text())
    lineages = [("constructed_gsmhard", "Constructed GSM-Hard", "constructed_gsmhard"),
                ("constructed_lcb", "Constructed LiveCodeBench", "constructed_lcb"),
                ("constructed_bird", "Constructed BIRD", "constructed_bird"),
                ("phaseA3_lcb_metaharness_B1_s2", "Evolved LiveCodeBench", "phaseA3_lcb_metaharness_B1_s2"),
                ("phaseA3_bird_metaharness_B1_s1", "Evolved BIRD (B1)", "phaseA3_bird_metaharness_B1_s1"),
                ("phaseA3_bird_metaharness_B3_s1", "Evolved BIRD (B3)", "phaseA3_bird_metaharness_B3_s1")]
    suffix = {"Qwen3-4B AWQ T0.7": "", "Qwen3-4B AWQ T1.0": "_t1", "Qwen3-4B bf16 T0.7": "_bf16",
              "Llama-3.1-8B MLX 4-bit T0.7 (Mac, exploratory)": "_llama", "Llama-3.1-8B AWQ vLLM T0.7 (GPU, exploratory)": "_llama_awq"}
    labels = {"Qwen3-4B AWQ T0.7": "AWQ, $T{=}0.7$", "Qwen3-4B AWQ T1.0": "AWQ, $T{=}1.0$", "Qwen3-4B bf16 T0.7": "bf16, $T{=}0.7$",
              "Llama-3.1-8B MLX 4-bit T0.7 (Mac, exploratory)": "Llama, MLX$^\\ast$",
              "Llama-3.1-8B AWQ vLLM T0.7 (GPU, exploratory)": "Llama, AWQ$^\\ast$"}
    table = ["\\begin{table}[p]", "\\centering",
             "\\caption{Transfer (RQ6) of the constructed lineages and the evolved real-gain lineages, in full. Held-out "
             "Shapley values in points under each solver setting, and the gain of the full lineage over the seed with its "
             "95\\% task-bootstrap interval. Qwen3-4B settings: AWQ at $T{=}0.7$ (the setting of evolution and of every "
             "other analysis), AWQ at $T{=}1.0$, and bf16. Llama-3.1-8B in two builds: 4-bit MLX on a laptop and AWQ "
             "INT4 under vLLM on the GPU. $^\\dagger$: a real gain under the registered criterion. $^\\ast$: "
             "exploratory, not registered. The first evolved BIRD lineage has two units.}",
             "\\label{tab:transfer-lineages}", "\\small", "\\begin{tabular}{llrrrl}", "\\toprule",
             "Lineage & Solver & U0 & U1 & U2 & gain \\\\", "\\midrule"]
    same_top, constructed_same = 0, 0
    for key, label, file in lineages:
        rows = data[key]["settings"]
        for i, (setting, row) in enumerate(rows.items()):
            name = f"{file}{suffix[setting]}"
            if file.startswith("constructed") and setting == "Qwen3-4B bf16 T0.7" and "gsmhard" not in file:
                name = f"{file}_bf16full"
            gain = json.loads((RESULTS / "transfer_gain" / f"gain_lattice_{name}_stmt_heldout.json").read_text())
            mark = "$^\\dagger$" if gain["real_gain"] else ""
            values = [pct(row["shapley"][u]) if u in row["shapley"] else "n/a" for u in ("0", "1", "2")]
            table.append(f"{label if i == 0 else ''} & {labels[setting]} & " + " & ".join(values) +
                         f" & {pct(gain['gain'])} [{pct(gain['low'])}, {pct(gain['high'])}]{mark} \\\\")
        table.append("\\addlinespace" if key != lineages[-1][0] else "")
        registered = {k: v for k, v in rows.items() if "exploratory" not in k}
        top = {v["top_unit"] for v in registered.values()}
        same_top += len(top) == 1
        constructed_same += len(top) == 1 and key.startswith("constructed")
    table += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (PAPER / "tables" / "transfer_lineages.tex").write_text("\n".join(t for t in table if t) + "\n")
    # Facts stated in words in the text.
    real = lambda name: json.loads((RESULTS / "transfer_gain" / f"gain_lattice_{name}_stmt_heldout.json").read_text())["real_gain"]  # noqa: E731
    assert constructed_same == 3 and same_top == 5 and len({v["top_unit"] for k, v in data["phaseA3_bird_metaharness_B3_s1"]["settings"].items()
                                                            if "exploratory" not in k}) > 1
    assert all(real(f"phaseA3_lcb_metaharness_B1_s2{x}") for x in ("", "_t1", "_bf16"))
    for name in ("phaseA3_bird_metaharness_B1_s1", "phaseA3_bird_metaharness_B3_s1"):
        assert real(name) and not real(f"{name}_t1") and not real(f"{name}_bf16")
    b3 = data["phaseA3_bird_metaharness_B3_s1"]["settings"]
    b3_gain = json.loads((RESULTS / "transfer_gain" / "gain_lattice_phaseA3_bird_metaharness_B3_s1_bf16_stmt_heldout.json").read_text())
    lines += [macro("TransConstructedSameTop", constructed_same), macro("TransSameTop", same_top),
              macro("TransBirdBBfGain", pct(b3_gain["gain"])), macro("TransBirdBBfGainCI", f"[{pct(b3_gain['low'])}, {pct(b3_gain['high'])}]"),
              macro("TransBirdBBfShapTwo", pct(b3["Qwen3-4B bf16 T0.7"]["shapley"]["2"])),
              macro("TransBirdBBfSeed", f"{100 * b3['Qwen3-4B bf16 T0.7']['seed']:.1f}\\%"),
              macro("TransBirdBAwqSeed", f"{100 * b3['Qwen3-4B AWQ T0.7']['seed']:.1f}\\%")]
    llama = "Llama-3.1-8B MLX 4-bit T0.7 (Mac, exploratory)"
    if llama in b3:
        g = json.loads((RESULTS / "transfer_gain" / "gain_lattice_phaseA3_bird_metaharness_B3_s1_llama_stmt_heldout.json").read_text())
        assert b3[llama]["top_unit"] == 2 and not g["real_gain"]  # stated in words in the text
        lines += [macro("TransBirdBLlamaGain", pct(g["gain"])), macro("TransBirdBLlamaGainCI", f"[{pct(g['low'])}, {pct(g['high'])}]"),
                  macro("TransBirdBLlamaShapTwo", pct(b3[llama]["shapley"]["2"]))]
    b1 = data["phaseA3_bird_metaharness_B1_s1"]["settings"]
    if llama in b1:
        g = json.loads((RESULTS / "transfer_gain" / "gain_lattice_phaseA3_bird_metaharness_B1_s1_llama_stmt_heldout.json").read_text())
        # Stated in words: the lineage loses accuracy under Llama (interval below zero), and U0, first under every
        # Qwen3-4B setting, is negative there.
        assert g["high"] < 0 and b1[llama]["shapley"]["0"] < 0 and all(v["top_unit"] == 0 for k, v in b1.items() if k.startswith("Qwen3"))
        lines += [macro("TransBirdALlamaGain", pct(g["gain"])), macro("TransBirdALlamaGainCI", f"[{pct(g['low'])}, {pct(g['high'])}]"),
                  macro("TransBirdALlamaShapZero", pct(b1[llama]["shapley"]["0"]))]
        awq = "Llama-3.1-8B AWQ vLLM T0.7 (GPU, exploratory)"
        if awq in b1:
            assert b1[awq]["shapley"]["0"] < 0  # stated: the SQLite guidance is negative under both Llama builds
            lines += [macro("TransBirdAAwqShapZero", pct(b1[awq]["shapley"]["0"]))]
    lcb = data["phaseA3_lcb_metaharness_B1_s2"]["settings"]
    if llama in lcb:
        g = json.loads((RESULTS / "transfer_gain" / "gain_lattice_phaseA3_lcb_metaharness_B1_s2_llama_stmt_heldout.json").read_text())

        def flips(suffix: str, unit: int = 0) -> tuple[int, int]:
            """(cell, product) pairs where adding the unit changes the outcome, out of all such pairs."""
            cells = json.loads((ROOT / "results" / "lattices" / f"phaseA3_lcb_metaharness_B1_s2{suffix}_stmt_heldout.json").read_text())["cells"]
            changed = total = 0
            for cell in cells:
                outcomes = cell["outcomes"]
                for key, value in outcomes.items():
                    present = {int(x) for x in key.split(",") if x}
                    if unit in present:
                        continue
                    total += 1
                    changed += outcomes[",".join(str(x) for x in sorted(present | {unit}))] != value
            return changed, total
        llama_flips, pairs = flips("_llama")
        # Stated in words: U2 still first, no real gain, U0 inert under every Qwen3-4B setting but not under Llama,
        # where its changes cancel (Shapley value zero).
        assert lcb[llama]["top_unit"] == 2 and not g["real_gain"] and abs(lcb[llama]["shapley"]["0"]) < 1e-12
        assert all(flips(s)[0] == 0 for s in ("", "_t1", "_bf16")) and llama_flips > 0
        lines += [macro("TransLcbLlamaGain", pct(g["gain"])), macro("TransLcbLlamaGainCI", f"[{pct(g['low'])}, {pct(g['high'])}]"),
                  macro("TransLcbLlamaShapTwo", pct(lcb[llama]["shapley"]["2"])), macro("TransLcbLlamaUZeroFlips", llama_flips),
                  macro("TransLcbLlamaUZeroPairs", f"{pairs:,}".replace(",", "{,}"))]
        awq = "Llama-3.1-8B AWQ vLLM T0.7 (GPU, exploratory)"
        if awq in lcb:
            g2 = json.loads((RESULTS / "transfer_gain" / "gain_lattice_phaseA3_lcb_metaharness_B1_s2_llama_awq_stmt_heldout.json").read_text())
            assert g2["real_gain"] and lcb[awq]["top_unit"] == 2  # stated: real under the AWQ build, test and repair first
            lines += [macro("TransLcbAwqGain", pct(g2["gain"])), macro("TransLcbAwqGainCI", f"[{pct(g2['low'])}, {pct(g2['high'])}]")]
    cbird = data["constructed_bird"]["settings"]
    if llama in cbird:
        g = json.loads((RESULTS / "transfer_gain" / "gain_lattice_constructed_bird_llama_stmt_heldout.json").read_text())
        # Stated in words: the technique unit still ranks first and the lineage keeps a real gain under Llama.
        assert cbird[llama]["top_unit"] == 2 and g["real_gain"]
        lines += [macro("TransConBirdLlamaGain", pct(g["gain"])), macro("TransConBirdLlamaGainCI", f"[{pct(g['low'])}, {pct(g['high'])}]"),
                  macro("TransConBirdLlamaShapTwo", pct(cbird[llama]["shapley"]["2"])),
                  macro("TransConBirdLlamaShapOne", pct(cbird[llama]["shapley"]["1"]))]


def transfer_summary(lines: list[str]) -> None:
    """RQ6 in one table: for every lineage, the unit that carries the gain under the evolution setting, in how many
    settings it still ranks first, in how many settings the lineage has a real gain, and whether the two Llama builds
    agree. Every count here is stated in the text; the asserts pin the statements."""
    path = RESULTS / "transfer.json"
    if not path.exists():
        return
    data = json.loads(path.read_text())
    awq_llama, mlx_llama = "Llama-3.1-8B AWQ vLLM T0.7 (GPU, exploratory)", "Llama-3.1-8B MLX 4-bit T0.7 (Mac, exploratory)"
    rows = [("planted", "Planted GSM-Hard", "Qwen3-4B AWQ T0.7"),
            ("constructed_gsmhard", "Constructed GSM-Hard", "Qwen3-4B AWQ T0.7"),
            ("constructed_lcb", "Constructed LiveCodeBench", "Qwen3-4B AWQ T0.7"),
            ("constructed_bird", "Constructed BIRD", "Qwen3-4B AWQ T0.7"),
            ("phaseA3_lcb_metaharness_B1_s2", "Evolved LiveCodeBench", "Qwen3-4B AWQ T0.7"),
            ("phaseA3_bird_metaharness_B1_s1", "Evolved BIRD (B1)", "Qwen3-4B AWQ T0.7"),
            ("phaseA3_bird_metaharness_B3_s1", "Evolved BIRD (B3)", "Qwen3-4B AWQ T0.7")]
    planted_files = {"Qwen3-4B MLX 4-bit T0.7 (Mac)": "planted", "Llama-3.1-8B MLX 4-bit T0.7 (Mac)": "planted_llama",
                     "Qwen3-4B AWQ T0.7": "planted_gpu", "Qwen3-4B AWQ T1.0": "planted_gpu_t1", "Qwen3-4B bf16 T0.7": "planted_gpu_bf16",
                     awq_llama: "planted_llama_awq"}
    suffix = {"Qwen3-4B AWQ T0.7": "", "Qwen3-4B AWQ T1.0": "_t1", "Qwen3-4B bf16 T0.7": "_bf16", mlx_llama: "_llama", awq_llama: "_llama_awq"}

    def gain(key: str, setting: str) -> dict:
        if key == "planted":
            return json.loads((RESULTS / "planted_gain" / f"gain_lattice_{planted_files[setting]}_stmt_heldout.json").read_text())
        name = key + suffix[setting]
        if key.startswith("constructed") and setting == "Qwen3-4B bf16 T0.7" and "gsmhard" not in key:
            name = key + "_bf16full"
        return json.loads((RESULTS / "transfer_gain" / f"gain_lattice_{name}_stmt_heldout.json").read_text())

    table = ["\\begin{table}[t]", "\\centering",
             "\\caption{Transfer (RQ6) in summary. Carrier: the unit with the largest held-out Shapley value under the "
             "evolution setting (Qwen3-4B, AWQ, $T{=}0.7$). First: settings in which it still ranks first. Real gain: settings "
             "in which the full lineage has a real gain over the seed. The two Llama-3.1-8B builds (MLX and AWQ): do they rank "
             "the same unit first, and do they agree on whether the gain is real? Appendix~\\ref{app:full} gives every value.}",
             "\\label{tab:transfer-summary}", "\\small", "\\begin{tabular}{llcccc}", "\\toprule",
             "Lineage & carrier & first & real gain & same first & same verdict \\\\", "\\midrule"]
    agree_top = agree_real = 0
    large_units = large_same_sign = 0
    counts = {}
    for key, label, reference in rows:
        settings = data[key]["settings"]
        carrier = settings[reference]["top_unit"]
        first = sum(1 for v in settings.values() if v["top_unit"] == carrier)
        reals = sum(1 for s in settings if gain(key, s)["real_gain"])
        llamas = [s for s in settings if s.startswith("Llama")]
        assert len(llamas) == 2, (key, llamas)
        same_top = settings[llamas[0]]["top_unit"] == settings[llamas[1]]["top_unit"]
        same_real = gain(key, llamas[0])["real_gain"] == gain(key, llamas[1])["real_gain"]
        agree_top += same_top
        agree_real += same_real
        for unit in settings[llamas[0]]["shapley"]:
            a, b = settings[llamas[0]]["shapley"][unit], settings[llamas[1]]["shapley"][unit]
            if max(abs(a), abs(b)) >= 0.02:
                large_units += 1
                large_same_sign += (a > 0) == (b > 0)
        counts[key] = (first, reals, len(settings))
        table.append(f"{label} & U{carrier} & {first}/{len(settings)} & {reals}/{len(settings)} & "
                     f"{'yes' if same_top else 'no'} & {'yes' if same_real else 'no'} \\\\")
    table += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (PAPER / "tables" / "transfer_summary.tex").write_text("\n".join(table) + "\n")
    technique = [k for k, _, _ in rows if not k.startswith("phaseA3")]
    # Stated in words: the technique unit of the planted and constructed lineages ranks first in every setting; the two
    # evolved BIRD gains are real only under the evolution setting; every large Shapley value has the same sign under
    # both Llama builds; the builds agree on the first unit everywhere.
    assert all(counts[k][0] == counts[k][2] for k in technique)
    assert all(counts[k][1] == 1 for k in ("phaseA3_bird_metaharness_B1_s1", "phaseA3_bird_metaharness_B3_s1"))
    assert large_same_sign == large_units and agree_top == len(rows)
    disagree = {k for k, _, _ in rows if gain(k, mlx_llama if k != "planted" else "Llama-3.1-8B MLX 4-bit T0.7 (Mac)")["real_gain"]
                != gain(k, awq_llama)["real_gain"]}
    assert disagree == {"constructed_lcb", "phaseA3_lcb_metaharness_B1_s2"}, disagree
    assert all(gain("constructed_gsmhard", s)["gain"] > 0.25 for s in (mlx_llama, awq_llama))
    # "The edit that carries each gain mostly stays the same" (abstract, conclusion): carrier first in most settings.
    carrier_first, carrier_settings = sum(c[0] for c in counts.values()), sum(c[2] for c in counts.values())
    assert carrier_first >= 0.9 * carrier_settings
    lines += [macro("TransCarrierFirst", carrier_first), macro("TransCarrierSettings", carrier_settings)]
    lines += [macro("TransTechniqueLineages", len(technique)), macro("TransTechniqueSettings", sum(counts[k][2] for k in technique)),
              macro("TransLlamaLineages", len(rows)), macro("TransLlamaAgreeTop", agree_top), macro("TransLlamaAgreeReal", agree_real),
              macro("TransLlamaLargeUnits", large_units),
              macro("TransEvoLcbReal", counts["phaseA3_lcb_metaharness_B1_s2"][1]), macro("TransEvoSettings", counts["phaseA3_lcb_metaharness_B1_s2"][2]),
              macro("TransConGsmReal", counts["constructed_gsmhard"][1]), macro("TransConBirdReal", counts["constructed_bird"][1]),
              macro("TransConLcbReal", counts["constructed_lcb"][1])]


def cost_table(lines: list[str]) -> None:
    """Cost of every real lattice by group (results/cost_by_lattice.json, from scripts/cost_by_lattice.py)."""
    path = RESULTS / "cost_by_lattice.json"
    if not path.exists():
        return
    groups = json.loads(path.read_text())
    labels = {"prior": "Earlier BIRD lineages", "planted": "Planted (every solver)", "constructed": "Constructed",
              "evolved": "Evolved (Phases A to A4)", "pooled": "Pooled recombination", "branched": "Branched (RQ1c)"}
    share = lambda a, b: a / b  # noqa: E731
    pc = lambda x: f"{100 * x:.0f}\\%" if x >= 0.095 else f"{100 * x:.1f}\\%"  # noqa: E731
    table = ["\\begin{table}[t]", "\\centering",
             "\\caption{Cost of every real lattice of the study, by group (audit lattices left out). Executions: runs the "
             "exploration executed, as a share of brute force (every product run on every cell). Tokens: tokens the model "
             "generated, as a share of the tokens the same executions would have generated without replay. Exploration "
             "skips runs only where runs differ in what they read; replay saves generation wherever runs send the same "
             "request.}",
             "\\label{tab:cost}", "\\small", "\\begin{tabular}{lrcrr}", "\\toprule",
             "Lineages & lattices & units & executions & generated tokens \\\\", "\\midrule"]
    for key in ("prior", "planted", "constructed", "evolved", "pooled", "branched"):
        g = groups[key]
        units = str(g["units_min"]) if g["units_min"] == g["units_max"] else f"{g['units_min']} to {g['units_max']}"
        table.append(f"{labels[key]} & {g['lattices']} & {units} & {pc(share(g['executions'], g['brute_force']))} & "
                     f"{pc(share(g['generated_tokens'], g['nominal_tokens']))} \\\\")
    table += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (PAPER / "tables" / "cost.tex").write_text("\n".join(table) + "\n")
    ex = {k: share(g["executions"], g["brute_force"]) for k, g in groups.items()}
    tok = {k: share(g["generated_tokens"], g["nominal_tokens"]) for k, g in groups.items()}
    # Stated in words: replay saves more than exploration in every group; exploration saves little on evolved,
    # constructed and pooled lattices, much on the planted one; replay generates at most half of the tokens.
    assert all(tok[k] < ex[k] for k in groups if k != "branched") and min(ex["evolved"], ex["constructed"], ex["pooled"]) > 0.9
    assert ex["planted"] < 0.5 and max(t for k, t in tok.items() if k != "branched") < 0.5
    # RQ1c text: on the branched lineages exploration, not replay, does most of the saving.
    assert ex["branched"] < 0.01 < tok["branched"]
    lines += [macro("CostEvolvedExec", pc(ex["evolved"])), macro("CostEvolvedTok", pc(tok["evolved"])),
              macro("CostConstructedTok", pc(tok["constructed"])), macro("CostPooledTok", pc(tok["pooled"])),
              macro("CostPriorTok", pc(tok["prior"])), macro("CostPlantedExec", pc(ex["planted"])),
              macro("CostPriorExec", pc(ex["prior"])), macro("CostBranchedExec", pc(ex["branched"])),
              macro("CostBranchedTok", pc(tok["branched"])),
              macro("CostMaxUnitsBefore", max(g["units_max"] for k, g in groups.items() if k != "branched")), macro("CostLattices", sum(g["lattices"] for g in groups.values()))]


def branched(lines: list[str]) -> None:
    """RQ1c: the branched lineages, whose units sit mostly on failure paths (results/branched.json, gain and report files)."""
    path = RESULTS / "branched.json"
    if not path.exists():
        return
    data = json.loads(path.read_text())

    def num(value):
        return f"{value:,}".replace(",", "{,}")

    names = {"bird": ("BIRD", "Bird"), "lcb": ("LiveCodeBench", "Lcb")}
    table = ["\\begin{table}[t]", "\\centering",
             "\\caption{RQ1c: branched lineages, 12 units each, most of them on failure paths (4{,}096 products). Audit: "
             "(product, cell) pairs of the exhaustive brute-force audit and mismatches. Executions: held-out executions "
             "against brute force, mean and largest number per cell, and the largest number of units one run read. Tokens: "
             "generated tokens as a share of the tokens the same executions would have generated without replay. Fresh: "
             "replayed minus fresh accuracy on the audited pairs, in points, with its 95\\% bootstrap interval.}",
             "\\label{tab:branched}", "\\small", "\\resizebox{\\textwidth}{!}{\\begin{tabular}{lrrrrrrl}", "\\toprule",
             "Benchmark & audit & held-out executions & per cell & largest & read width & tokens & fresh \\\\", "\\midrule"]
    for bench, (label, tag) in names.items():
        if bench not in data:
            continue
        entry, held = data[bench], data[bench]["heldout"]
        audit, fresh = entry["audit"], entry["fresh_audit"]
        share = held["executions"] / held["brute_force"]
        table.append(f"{label} & {num(audit['checked'])}; {audit['mismatches']} & {num(held['executions'])} of {num(held['brute_force'])} "
                     f"({100 * share:.2f}\\%) & {held['mean_leaves_per_cell']:.1f} & {held['max_leaves_per_cell']} & "
                     f"{held['max_read_width']} & {100 * held['generated_tokens'] / held['nominal_tokens']:.0f}\\% & "
                     f"{pct(fresh['difference'])} [{pct(fresh['low'])}, {pct(fresh['high'])}] \\\\")
        gain = json.loads((RESULTS / f"gain_lattice_branched_{bench}_stmt_heldout.json").read_text())
        lines += [macro(f"Br{tag}AuditChecked", num(audit["checked"])), macro(f"Br{tag}AuditMismatches", audit["mismatches"]),
                  macro(f"Br{tag}HeldExec", num(held["executions"])), macro(f"Br{tag}HeldBrute", num(held["brute_force"])),
                  macro(f"Br{tag}HeldShare", f"{100 * share:.2f}\\%"), macro(f"Br{tag}MeanCell", f"{held['mean_leaves_per_cell']:.1f}"),
                  macro(f"Br{tag}MaxCell", held["max_leaves_per_cell"]), macro(f"Br{tag}Width", held["max_read_width"]),
                  macro(f"Br{tag}HeldMinutes", f"{held['seconds'] / 60:.0f}"), macro(f"Br{tag}Cells", num(held["cells"])),
                  macro(f"Br{tag}Tok", f"{100 * held['generated_tokens'] / held['nominal_tokens']:.0f}\\%"),
                  macro(f"Br{tag}FreshPairs", num(fresh["pairs"])), macro(f"Br{tag}FreshDiff", pct(fresh["difference"])),
                  macro(f"Br{tag}FreshCI", f"[{pct(fresh['low'])}, {pct(fresh['high'])}]"),
                  macro(f"Br{tag}Gain", pct(gain["gain"])), macro(f"Br{tag}GainCI", f"[{pct(gain['low'])}, {pct(gain['high'])}]"),
                  macro(f"Br{tag}Unread", len(entry["unread_units"])), macro(f"Br{tag}Inert", len(entry["inert_units"]))]
    # Stated in the text: every registered prediction held, exploration ran under 1% of brute force, and most exactly
    # inert units were read by some run.
    inert = [u for entry in data.values() for u in entry["inert_units"]]
    read_inert = [u for entry in data.values() for u in entry["inert_units"] if u not in entry["unread_units"]]
    assert len(read_inert) > len(inert) / 2
    assert all(e["heldout"]["executions"] < 0.01 * e["heldout"]["brute_force"] and e["audit"]["mismatches"] == 0 for e in data.values())
    table += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    (PAPER / "tables" / "branched.tex").write_text("\n".join(table) + "\n")


def pool_audits(lines: list[str]) -> None:
    """Brute-force audits of the pooled lattices (results/pool_audits.json, scripts/pool_audit_summary.py)."""
    path = RESULTS / "pool_audits.json"
    if not path.exists():
        return
    audits = json.loads(path.read_text())
    # Registered: every conflict-free product of every pool that was run, on 10 search tasks x 1 replicate.
    assert all(a["checked"] == a["products"] * a["cells"] and a["cells"] == 10 for a in audits.values())
    checked = sum(a["checked"] for a in audits.values())
    lines += [macro("PoolAuditPools", len(audits)), macro("PoolAuditChecked", f"{checked:,}".replace(",", "{,}")),
              macro("PoolAuditMismatches", sum(a["mismatches"] for a in audits.values()))]


def decisions(lines: list[str]) -> None:
    """Exploratory: decisions from sampled estimates at equal cost (results/decisions.json)."""
    path = RESULTS / "decisions.json"
    if not path.exists():
        return
    data = json.loads(path.read_text())
    def rates(estimator, regime):
        return [r["inert_false_effect_rate"] for v in data.values() for r in v["rows"]
                if r["estimator"] == estimator and r["regime"] == regime and r["inert_false_effect_rate"] is not None]
    pct_text = lambda x: f"{100 * x:.0f}\\%"
    lines += [macro("DecPermCoupledMax", pct_text(max(rates("permutation", "coupled")))),
              macro("DecPermIndMin", pct_text(min(rates("permutation", "independent")))),
              macro("DecPermIndMax", pct_text(max(rates("permutation", "independent")))),
              macro("DecKernelCoupledMax", pct_text(max(rates("kernelshap", "coupled"))))]


def compute(lines: list[str]) -> None:
    """Compute used for lattices and evolution (results/compute_*.json, from scripts/compute_summary.py)."""
    hosts = [json.loads(path.read_text()) for path in sorted(RESULTS.glob("compute_*.json"))]
    if not hosts:
        return
    rows = [row for host in hosts for row in host["lattices"].values()]
    total = lambda key: sum(row[key] for row in rows)  # noqa: E731
    big = lambda value: f"{value:,}".replace(",", "{,}")  # noqa: E731
    lines += [macro("CompLatticeHours", f"{total('seconds') / 3600:.0f}"), macro("CompExecutions", big(total("executions"))),
              macro("CompGenTokens", f"{total('generated_tokens') / 1e6:.0f}"),
              macro("CompNominalTokens", f"{total('nominal_tokens') / 1e6:.0f}"),
              macro("CompTokenRatio", f"{total('nominal_tokens') / total('generated_tokens'):.0f}"),
              macro("CompEvolutionHours", f"{sum(sum(host['evolution_hours'].values()) for host in hosts):.0f}")]


def evolved(lines: list[str]) -> None:
    """The evolved real-gain lineage (Phase A3, LiveCodeBench, loop B1_s2): gain check, lattice report, audit."""
    gpu = RESULTS / "gpu"
    report_path = gpu / "report_phaseA3_lcb_metaharness_B1_s2_stmt.json"
    gain_path = gpu / "gain_gain_pilot3_lcb.json"
    audit_path = gpu / "a2_phaseA3_lcb_metaharness_B1_s2_stmt_heldout.json"
    if not (report_path.exists() and gain_path.exists() and audit_path.exists()):
        return
    report = json.loads(report_path.read_text())
    loop = next(row for row in json.loads(gain_path.read_text())["loops"] if row["loop"] == "metaharness_B1_s2")
    audit = json.loads(audit_path.read_text())["summary"]
    acc = lambda x: f"{100 * x:.1f}\\%"  # noqa: E731
    interval = lambda e: f"[{pct(e['low'])}, {pct(e['high'])}]"  # noqa: E731
    held, search = report["rq3"]["heldout"], report["rq3"]["search"]
    # The text says the lattice gain lies inside the gain check's interval; fail loudly if that stops being true.
    assert loop["low"] <= held["full"] - held["seed"] <= loop["high"]
    assert all(r["pairs_used"] == len(json.loads((RESULTS / "lattices" / "phaseA3_lcb_metaharness_B1_s2_stmt_heldout.json").read_text())["cells"])
               for r in report["rq4"])
    lines += [macro("EvoAccepted", loop["accepted"]), macro("EvoUnits", len(report["units"])),
              macro("EvoGain", pct(loop["gain"])), macro("EvoGainCI", interval(loop)),
              macro("EvoSeedAcc", acc(loop["seed_accuracy"])), macro("EvoFinalAcc", acc(loop["final_accuracy"])),
              macro("EvoLatSeed", acc(held["seed"])), macro("EvoLatFull", acc(held["full"])),
              macro("EvoLatGain", pct(held["full"] - held["seed"])),
              macro("EvoSearchSeed", acc(search["seed"])), macro("EvoSearchFull", acc(search["full"])),
              macro("EvoAuditPairs", audit["pairs"]), macro("EvoAuditSignificant", audit["products_with_p_below_0.05"]),
              macro("EvoAuditReplay", acc(audit["replayed_accuracy"])), macro("EvoAuditFresh", acc(audit["fresh_accuracy"]))]
    for unit, tag in (("0", "Zero"), ("1", "One"), ("2", "Two")):
        lines += [macro(f"EvoShap{tag}", pct(held["shapley"][unit]["value"])), macro(f"EvoShap{tag}CI", interval(held["shapley"][unit])),
                  macro(f"EvoSearchShap{tag}", pct(search["shapley"][unit]["value"])),
                  macro(f"EvoSearchShap{tag}CI", interval(search["shapley"][unit]))]
    lines += [macro("EvoIntOneTwo", pct(held["interaction"]["1,2"]["value"])), macro("EvoIntOneTwoCI", interval(held["interaction"]["1,2"])),
              macro("EvoSearchIntOneTwo", pct(search["interaction"]["1,2"]["value"]))]
    for delta, tag in ((0.02, "Small"), (0.05, "Large")):
        rows = [r for r in report["rq4"] if abs(r["delta"] - delta) < 1e-9]
        products = {tuple(r["product"]) for r in rows}
        assert len(products) == 1, products
        row = rows[0]
        lines += [macro(f"EvoDist{tag}Kept", row["units_kept"]), macro(f"EvoDist{tag}Held", acc(row["heldout_value"])),
                  macro(f"EvoDist{tag}Certified", sum(r["certified"] for r in rows))]
    # Lattice products quoted in the text (held-out and search accuracy of {U2} and {U1,U2}).
    for split in ("search", "heldout"):
        lattice = json.loads((RESULTS / "lattices" / f"phaseA3_lcb_metaharness_B1_s2_stmt_{split}.json").read_text())
        cells = lattice["cells"]
        mean = lambda p: sum(float(c["outcomes"][p]) for c in cells) / len(cells)  # noqa: E731
        tag = "Search" if split == "search" else "Held"
        lines += [macro(f"Evo{tag}OnlyTwo", acc(mean("2"))), macro(f"Evo{tag}OneTwo", acc(mean("1,2"))),
                  macro(f"Evo{tag}Executions", f"{lattice.get('executions', len(cells) * len(cells[0]['outcomes'])):,}".replace(",", "{,}"))]


def evolved_bird(lines: list[str]) -> None:
    """The two evolved real-gain lineages on BIRD (Phase A3 loops B1_s1 and B3_s1)."""
    gpu = RESULTS / "gpu"
    gain_path = gpu / "gain_gain_pilot3_bird.json"
    if not gain_path.exists():
        return
    loops = {row["loop"]: row for row in json.loads(gain_path.read_text())["loops"]}
    acc = lambda x: f"{100 * x:.1f}\\%"  # noqa: E731
    interval = lambda e: f"[{pct(e['low'])}, {pct(e['high'])}]"  # noqa: E731
    names = ["Zero", "One", "Two", "Three"]
    for loop, tag in (("metaharness_B1_s1", "BirdA"), ("metaharness_B3_s1", "BirdB")):
        report_path = gpu / f"report_phaseA3_bird_{loop}_stmt.json"
        audit_path = gpu / f"a2_phaseA3_bird_{loop}_stmt_heldout.json"
        if not (report_path.exists() and audit_path.exists()):
            continue
        report = json.loads(report_path.read_text())
        audit = json.loads(audit_path.read_text())["summary"]
        held, search = report["rq3"]["heldout"], report["rq3"]["search"]
        gain = loops[loop]
        # Facts stated in words in the text; fail loudly if the data no longer support them.
        expected = {"BirdA": set(), "BirdB": {"2"}}[tag]
        assert {u for u, v in held["matters"].items() if v} == expected
        assert gain["low"] <= held["full"] - held["seed"] <= gain["high"]
        lines += [macro(f"Evo{tag}Accepted", gain["accepted"]), macro(f"Evo{tag}Units", len(report["units"])),
                  macro(f"Evo{tag}Gain", pct(gain["gain"])), macro(f"Evo{tag}GainCI", interval(gain)),
                  macro(f"Evo{tag}LatSeed", acc(held["seed"])), macro(f"Evo{tag}LatFull", acc(held["full"])),
                  macro(f"Evo{tag}LatGain", pct(held["full"] - held["seed"])),
                  macro(f"Evo{tag}AuditPairs", audit["pairs"]), macro(f"Evo{tag}AuditSignificant", audit["products_with_p_below_0.05"]),
                  macro(f"Evo{tag}Matters", sum(bool(v) for v in held["matters"].values()))]
        for unit, entry in held["shapley"].items():
            name = names[int(unit)]
            lines += [macro(f"Evo{tag}Shap{name}", pct(entry["value"])), macro(f"Evo{tag}Shap{name}CI", interval(entry)),
                      macro(f"Evo{tag}SearchShap{name}", pct(search["shapley"][unit]["value"])),
                      macro(f"Evo{tag}SearchShap{name}CI", interval(search["shapley"][unit]))]
        for delta, size in ((0.02, "Small"), (0.05, "Large")):
            rows = [r for r in report["rq4"] if abs(r["delta"] - delta) < 1e-9]
            assert len({tuple(r["product"]) for r in rows}) == 1
            row = rows[0]
            lines += [macro(f"Evo{tag}Dist{size}Kept", row["units_kept"]), macro(f"Evo{tag}Dist{size}Held", acc(row["heldout_value"])),
                      macro(f"Evo{tag}Dist{size}Certified", sum(r["certified"] for r in rows))]


def limitation_example(lines: list[str]) -> None:
    """The split refactor quoted in Limitations (Phase A LiveCodeBench loop B3_s2, held-out lattice)."""
    path = RESULTS / "lattices" / "phaseA_lcb_metaharness_B3_s2_stmt_heldout.json"
    summary = RESULTS / "null_summary.json"
    if not (path.exists() and summary.exists()):
        return
    lattice = json.loads(path.read_text())
    full = ",".join(str(unit["index"]) for unit in lattice["units"])
    # The text says the refactor as a whole changes no outcome and that it became three units.
    assert len(lattice["units"]) == 3 and all(cell["outcomes"][full] == cell["outcomes"][""] for cell in lattice["cells"])
    units = [u for u in json.loads(summary.read_text())["units"] if u["lineage"] == "phaseA_lcb_metaharness_B3_s2"]
    values = [pct(u["heldout"]) for u in sorted(units, key=lambda u: int(u["unit"]))]
    lines += [macro("LimRefactorA", values[0]), macro("LimRefactorB", values[1]), macro("LimRefactorC", values[2])]


def sensitivity(lines: list[str]) -> None:
    """DEC-16: LiveCodeBench gain checks rerun with the evolution-time solver budget (2,048 tokens)."""
    sens = RESULTS / "sens_lcb2048"
    if not sens.exists():
        return
    tags = {"gain_pilot": "One", "gain_pilot2": "Two", "gain_pilot3": "Three"}
    total_reg = total_sens = 0
    for prefix, tag in tags.items():
        reg = {l["loop"]: l for l in json.loads((RESULTS / "gpu" / f"gain_{prefix}_lcb.json").read_text())["loops"]}
        new = {l["loop"]: l for l in json.loads((sens / f"gain_{prefix}_lcb.json").read_text())["loops"]}
        assert reg.keys() == new.keys()
        total_reg += sum(l["real_gain"] for l in reg.values())
        total_sens += sum(l["real_gain"] for l in new.values())
        lines += [macro(f"Sens{tag}Real", sum(l["real_gain"] for l in new.values())), macro(f"Sens{tag}RealReg", sum(l["real_gain"] for l in reg.values()))]
    a3 = {l["loop"]: l for l in json.loads((sens / "gain_gain_pilot3_lcb.json").read_text())["loops"]}["metaharness_B1_s2"]
    a1 = {l["loop"]: l for l in json.loads((sens / "gain_gain_pilot_lcb.json").read_text())["loops"]}["metaharness_B1_s2"]
    # Facts stated in words in the text.
    assert not a3["real_gain"] and a1["real_gain"] and total_reg == total_sens == 1
    lines += [macro("SensTotal", total_sens), macro("SensEvoGain", pct(a3["gain"])), macro("SensEvoGainCI", f"[{pct(a3['low'])}, {pct(a3['high'])}]"),
              macro("SensAGain", pct(a1["gain"])), macro("SensAGainCI", f"[{pct(a1['low'])}, {pct(a1['high'])}]")]


def main() -> None:
    """Write macros and tables."""
    lines = ["% generated by scripts/paper_numbers.py from results/*.json -- do not edit"]
    for section in (rq1_prior, planted, rq2, scaling, pilots, null_group, constructed, name_vs_statement, transfer, decisions, compute, evolved, evolved_bird, transfer_lineages, transfer_summary, cost_table, branched, pool_audits, limitation_example, sensitivity):
        section(lines)
    PAPER.mkdir(exist_ok=True)
    (PAPER / "macros.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {len(lines) - 1} macros")


if __name__ == "__main__":
    sys.exit(main())
