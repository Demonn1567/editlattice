# EditLattice: code, data and results

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24512751.svg)](https://doi.org/10.5281/zenodo.24512751)

This repository accompanies the paper "Which self-made edits matter? Exact attribution for self-improving
language-model agents through variational replay" by Krish Sharma and Jatin Bedi (Thapar Institute of Engineering and
Technology). It contains the method, the
experiment drivers, the registered analysis plan, the decision log and every result file from which the
paper's numbers, tables and figures are generated.

## Contents

* `src/editlattice/`: the method and the study.
  * `units.py`: net change units of a lineage, products, recombination conflicts and statement-level markers.
  * `variational.py`: the read model, the instrumentation and variational replay.
  * `tree.py`: exploration over ordered reads, closed-form attribution from the leaves and exact
    optimisation over the leaves.
  * `attribution.py`, `distill.py`, `estimators.py`: Shapley values, interactions, Möbius dividends, the
    task bootstrap, distillation with the anytime-valid certificate and the sampled Shapley estimators compared in the paper.
  * `evolution/`, `proposers/`, `gates/`: the evolution loop, the proposer and the acceptance gates.
  * `benchmarks/`, `seed_harnesses/`: GSM-Hard, LiveCodeBench and BIRD, and the seed harnesses.
* `examples/`: the planted lineage, the constructed lineages and the branched lineages.
* `tests/`: 113 tests, including fuzz tests of exactness, negative controls and a regression test for every
  defect listed in the deviations section of the supplementary material.
* `notes/preregistration.md`: the analysis plan and its addenda, each committed before the runs it governs.
* `notes/decisions.md`: every deviation from the plan (DEC-1 to DEC-20) with its evidence.
* `notes/theory.md`: proofs.
* `configs/`: solver and evolution settings and the data splits.
* `results/`: every result file the paper reads.
* `scripts/`: experiment drivers and analysis scripts. `scripts/gpu/` holds the stage scripts, which were run
  in the order of the decision log.

## Setup

Python 3.12 is required for the experiments, because the evolution ran with it and some evolved harnesses
use 3.12 syntax. Scripts that only read result files also run on 3.11.

```
pip install -e .
bash scripts/fetch_datasets.sh    # downloads the benchmarks from their official sources and checks SHA-256
pytest tests
```

Running experiments needs an OpenAI-compatible server for the solver. The study used vLLM 0.8.5 serving
Qwen3-4B-AWQ (see `configs/`).

## Reproducing the paper from the stored results

| Paper element | Command |
| --- | --- |
| Every number and main-text table | `python scripts/paper_numbers.py` |
| Every figure | `python scripts/make_figures.py` |
| Supplementary tables and listings, the worked example and the registered predictions | `python scripts/export_appendix.py` |
| Sizes of the pooled lattices | `python scripts/pool_sizes.py` |
| Summaries of the branched lineages and of the pooled-lattice audits | `python scripts/branched_summary.py` and `python scripts/pool_audit_summary.py`, which read the lattice files written by the stage scripts |

These scripts also assert many of the statements the paper makes about their numbers; for those
statements, a result that contradicted the text would stop them.

## Rerunning the experiments

The stage scripts in `scripts/gpu/` run the study on one 16 GB GPU. The 14B proposer of Phases A2 and A3 and
the MLX build of the exploratory Llama runs used a separate machine; `scripts/mac_llama_transfer2.sh` and the helpers in
`runs/` were used there. `scripts/run_evolution.py` runs one evolution loop, and `scripts/run_lattice.py`
explores one lattice (`--brute` adds the exhaustive audit).

Phase A5 (evolution from multi-step seed harnesses) was registered in `notes/preregistration.md` on 2026-11-24 and
runs from `scripts/gpu/stage_a5.sh` with the `configs/gain_pilot5_*.yaml` settings; its results are added to
`results/` when the runs finish.

## Citation and licence

The archived versions are on Zenodo under the DOI 10.5281/zenodo.24512751, which always resolves to the latest
version. `CITATION.cff` gives the citation. The code and the result files are released under the MIT licence (`LICENSE`).
