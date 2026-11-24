# Pre-registered analysis plan (fixed 2026-10-27, before any GPU result)

This file is committed before the gain pilot runs. Any later deviation is logged in
`notes/decisions.md` with its reason and reported in the paper. Every result below is reported
whatever it shows.

## Models and serving
- Solver: Qwen3-4B-AWQ, non-thinking mode, temperature 0.7, top-p 0.95, served by vLLM 0.8.5 on one
  RTX A4000 (16 GB). Same solver for evolution, lattices and held-out evaluation.
- Proposer: Qwen3-14B-AWQ, temperature 0.7 (fallback Qwen3-8B-AWQ only if the 14B model cannot be
  served next to the solver; logged as a deviation).
- Transfer (RQ6): Qwen3-4B (bf16) and temperature 1.0 with the AWQ solver.

## Benchmarks and splits
Frozen splits in `configs/splits/` (search / validation / held-out, stratified):
GSM-Hard 528/264/527, LiveCodeBench 86/43/85, BIRD Mini-Dev 198/97/199. Evolution sees only the
first 60 search and 40 validation tasks. Held-out tasks are used only for the gain check, held-out
attribution, certification and transfer, never to choose anything.
GSM-Hard: results also reported on the subset with plausible targets.

## Phase A: gain pilot (lineages)
- Seeds: math_direct (GSM-Hard), lcb_oneshot (LCB), bird_oneshot (BIRD).
- Loop: Meta-Harness-style proposer; gates B1 (greedy) and B3 (Self-Harness, 2 repeats); 4 seeds
  each; at most 30 iterations; 2,500 task-runs per loop; CRN streams.
- Real-gain criterion (design D8): held-out accuracy of the final harness minus the seed's, paired
  per task on shared streams, 200 held-out tasks (all 85 for LCB) x 4 replicates, 95% task-cluster
  bootstrap CI above 0 and gain >= 3 points.
- A benchmark enters the main study if at least half of its loops are real-gain. All loops, real-gain
  or not, are reported; null-gain lineages form a separate contrast group.

## Phase B: lattices
- Units: net units seed -> final (`units.build_units`). If a lineage has more than 12 units, units are
  grouped by top-level item before analysis (coarser features), and the grouping is reported.
- Each lineage's full lattice is evaluated exactly with variational replay (statement-level reads)
  on (a) the 60 search tasks x 4 replicates and (b) the held-out tasks x 4 replicates.
- Exactness audit on every lineage: brute-force comparison (`--brute`) on 10 tasks x 1 replicate.
- A2 audit: for 10% of (product, cell) pairs, a fresh run on an independent stream; replay and fresh
  accuracies compared per product (Fisher exact test); reported as in RQ2 of prior work.

## Research questions and analyses
- RQ1 exactness and cost: mismatches (expected 0), executions vs brute force, generated vs nominal
  tokens, name-level vs statement-level reads, per benchmark.
- RQ2 estimation at equal budget: ground truth = exact lattice values on the held-out cells.
  Compared with Monte Carlo permutation Shapley, KernelSHAP, leave-one-out and leave-one-in, each run
  on independent streams with the same number of task-runs as the coupled exact computation, and with
  their coupled (replay) versions. Error = mean absolute error of Shapley values and of pairwise
  interactions; 20 repetitions per estimator and budget.
- RQ3 what matters: per lineage, Shapley values and pairwise interactions with 95% bootstrap CIs (2,000
  resamples) on search and held-out cells; a unit "matters" if its held-out Shapley CI excludes 0
  (Holm correction across units within a lineage); interaction counted if its CI excludes 0 (Holm).
  Overfitting unit: search Shapley CI above 0 and held-out Shapley CI containing or below 0.
  Churn statistics for every lineage.
- RQ4 distillation: delta in {0.02, 0.05}; cost = number of units and mean generated + prompt tokens
  per task; methods: exact minimum, greedy backward, ddmin, LOO threshold, Shapley threshold;
  products chosen on search cells; certification on held-out cells with the anytime-valid
  non-inferiority test at alpha = 0.05 against the full harness.
- RQ5 recombination: per benchmark and gate, pool the units of the 4 seeds; choose the best product
  on search cells; on held-out cells test superiority over the best single final harness (anytime-valid,
  alpha = 0.05) and report the difference with a bootstrap CI.
- RQ6 transfer: Spearman correlation of held-out Shapley values across solvers/temperatures; does the
  distilled product remain within delta of the full harness under the other solver?

## What is not done
- No hyper-parameter is tuned on held-out cells.
- No lineage is dropped after seeing its attribution.
- No unit grouping is changed after seeing results.

## Addendum 2026-10-27 (before running): planted-structure validation (Mac)
Lineage math_direct -> examples/gsm_planted_final.py, 4 planted units:
U0 prompt asks for a Python program; U1 extract_answer truncates the answer to its integer part
(harmful on non-integer targets); U2 solve executes the program and answers with its output;
U3 a new helper that nothing calls. Solver: Qwen3-4B (MLX 4-bit) on the Mac, temperature 0.7.
Cells: GSM-Hard search 40 tasks x 2 replicates and held-out 40 x 2; statement-level reads;
brute-force exactness check on the search cells.
Predictions, fixed now:
- P1 exactness: 0 mismatches.
- P2 U3 is never read and its Shapley value, all its interactions and its Möbius terms are exactly 0.
- P3 synergy: the pairwise interaction of U0 and U2 is positive (its CI may include 0 at this size).
- P4 U2 alone (without U0) changes little: |LOI(U2)| is smaller than LOO(U2).
- P5 U1 is harmful or neutral: Shapley(U1) <= 0, and U1 is read only on runs that reach the
  fallback answer path (fewer runs read U1 than read U2).
- P6 masking: interaction(U1, U2) >= 0 (code execution bypasses the truncation).
All outcomes are reported whether or not they match.

## Deviation log
- DEC-2 (2026-10-27): proposer Qwen3-8B-AWQ (the pre-registered fallback); evidence in notes/decisions.md.
- DEC-4 (2026-10-27): first-line indentation tolerance in edit application; LCB pilot restarted before any gate decision; GSM-Hard unaffected (checked by re-applying every logged proposal).

## Addendum 2026-10-27 (before running): D11 tree exploration and the RQ1b scaling study
Method change (no result seen): exploration is depth-first over ordered reads (tree.explore_tree);
attribution is computed in closed form from the leaves (tree.TreeLattice); distillation and
recombination are exact MILPs over the leaves; static analysis is compositional over independent
groups (units.independent_groups). On every lattice size where both run, all quantities must equal
the exhaustive ones (tests: 102+ pass, closed forms to 1e-12). From now on lattices are explored
with the tree; for lineages with at most 12 units the exhaustive explorer is also run on the 10-task
brute-force audit, as pre-registered.
RQ1b scaling (synthetic, fake LLM; scripts/scaling.py; seed 20261027): m in {4, 8, 12, 16, 24, 32,
48, 64} x route length L in {2, 4, 8} (L <= m), plus L = m for m in {10, 12, 14, 16}; 24 tasks x 2
replicates; exactness checked against plain runs on seed, full and 16 random products per cell.
Predictions: (P1) 0 mismatches; (P2) executions per cell <= 2^(read width) and read width <= L + 0
(each run reads at most the units of its route); (P3) executions grow with L, not with m, at fixed L;
(P4) compositional static analysis time grows roughly linearly in m while the exhaustive one grows
like 2^m; (P5) closed-form Shapley/interaction values equal exhaustive formulas (m <= 10) to 1e-12;
(P6) MILP optima found in seconds for every m. All rows are reported.
Real-scale use (Phase B, pre-registered here): RQ5 pools every loop of a benchmark (both gates, 4
seeds) without grouping units, explored with the tree; the pooled optimum is chosen on search cells
by MILP and tested on held-out cells exactly as RQ5. Unit grouping (>12 units) is no longer needed
for single lineages; grouped results are still reported as pre-registered, alongside the ungrouped ones.

## Addendum 2026-10-28 (before running): Phase A2 gain pilot
Why: Phase A gave 0 of 8 real-gain loops on GSM-Hard and 0 of 8 on LCB (BIRD running). Search-side
logs only (no held-out data) show proposer failures that are partly our re-implementation's: stale
seed-version SEARCH text copied from history diffs (53 of 65 LCB not-found), unified-diff replies
(21 LCB), a Markdown fence broken by backticks inside LCB's prompt, no use of env.run_python, and
the weaker fallback proposer (DEC-2).
Changes (all others identical to Phase A: seeds 1-4, gates B1 and B3, 30 iterations, 2,500 runs,
splits, solver Qwen3-4B-AWQ on the A4000, CRN streams, real-gain criterion, entry rule):
1. Proposer: Qwen3-14B, the originally pre-registered proposer, as mlx-community/Qwen3-14B-4bit on the
   Mac (MLX 4-bit instead of AWQ: a deviation in quantisation and serving stack, not in model),
   temperature 0.7, top-p 0.95, non-thinking, max_tokens 3072; reached from the GPU through an SSH
   reverse tunnel; one request at a time (max_in_flight 1).
2. Prompt v2 (proposers.ProposerBase, prompt_version 2): fence longer than any backtick run in the
   harness; history shows candidates' new lines only; an environment description generated by one
   procedure for every benchmark (describe_environment); "copy SEARCH lines from the CURRENT harness".
3. accept_diff: unified-diff bodies between SEARCH/REPLACE markers are read as edits (same
   exactly-once rule).
4. DEC-6: proposer outages (server unreachable) wait and retry with the same examples and stream,
   instead of being logged as invalid proposals.
Probe before this addendum (scripts/probes/proposer_probe.py, search-side runs of the seed only): 14B
gave valid edits on 5 of 5 LCB proposals so far (8B in Phase A: 55 of 142+), about 100 s each.
Order: GSM-Hard, then LCB (experiments gain_pilot2_*); BIRD only if BIRD's Phase A fails the entry
rule. Phase A loops stay the null-gain contrast group; A2 loops are reported separately with their
own real-gain counts. Phase B runs on every benchmark that passes the entry rule in A or A2. If A2
also fails everywhere, the paper reports the method on the null-gain lineages, the planted lineage
and the synthetic study, and says so.

## Addendum 2026-10-29 (before running): Phase C (constructed lineages) and Phase A3 (reasoning proposer)
Why: 0 of 48 evolved loops (Phase A and A2) reached a real gain, so RQ3-RQ6 have no real-gain lineage.
Phase C: one constructed lineage per benchmark, written from standard published techniques before any
lattice of it was run (examples/constructed/*_constructed_final.py; checked only for crashes on 5
search tasks; no held-out data seen). Units (build_units): GSM-Hard: U0 prompt asks for exact
arithmetic, U1 answer normalisation (commas, $), U2 program-aided solving (write a program, run it,
retry once, prefer its printed number). LiveCodeBench: U0 system prompt (read all input with
sys.stdin, exact output format), U1 user prompt ("think about the time limit"), U2 self-repair on the
public examples (two rounds). BIRD: U0 prompt (schema-only columns, follow the hint), U1 three example
rows per table in the schema text, U2 execution-guided repair (error or empty result, two rounds).
Protocol: the D8 real-gain test of the full lineage vs the seed on held-out cells; then the registered
Phase B lattices (statement-level tree exploration, 60 search x 4 and all held-out x 4, 10-task
brute-force audit, 10% fresh-run audit, RQ3/RQ4 reports), RQ2 on the held-out lattice, and RQ6 on
held-out under AWQ T=1.0 and bf16 T=0.7. Predictions, fixed now: (C1) GSM-Hard and LCB full lineages
are real-gain; BIRD uncertain. (C2) The technique unit (U2) has the largest held-out Shapley value in
each lineage. (C3) GSM-Hard U1 has Shapley <= 0.01 in absolute value (the solver already answers
cleanly). (C4) In LCB, I(U0,U2) and I(U1,U2) are not significantly positive. (C5) The distilled
product at delta = 0.02 keeps U2 in every real-gain lineage. (C6) Shapley ranks of U2 transfer (first
under every RQ6 setting). All outcomes reported.
Phase A3: Phase A2 with the proposer's thinking mode on (enable_thinking true, max_tokens 6144; the
reasoning text before </think> is removed before edit blocks are parsed), everything else identical
(experiments gain_pilot3_*; GSM-Hard, LCB, BIRD in this order). Same entry rule; Phase B on any
benchmark that passes.
Phase C detail: the fresh-run audit runs at fraction 1.0 (every product and cell on a private stream),
a superset of the registered 10% audit that also provides RQ2's independent-stream estimates.

## Addendum 2026-11-01 (before running): RQ6 on the evolved real-gain lineages; DEC-11 reruns
Phase A3 produced three real-gain loops under the registered D8 criterion: LCB metaharness_B1_s2, BIRD
metaharness_B1_s1 and BIRD metaharness_B3_s1. RQ6 applies to them exactly as registered (top of this
file): held-out lattices (all held-out tasks x 4, statement-level tree) under AWQ T=1.0 and under bf16
Qwen3-4B T=0.7, each in its own stream namespace; Spearman correlation of held-out Shapley values with
the AWQ T=0.7 lattice, and whether the distilled product stays within delta of the full harness. No
other lineage is added and none is dropped. Order: the three AWQ T=1.0 lattices after Phase B (stage_b4)
ends; then both vLLM servers are stopped and the bf16 server is restarted alone with an 8,192-token
context (the solver's context), and the bf16 lattices run: first the DEC-11 reruns of constructed LCB and
BIRD (new namespaces *_bf16full; the quarantined invalid_* files are not used), then the three evolved
lineages. A request the server rejects stops the run (DEC-11), so no rejection can be scored. Script:
scripts/gpu/stage_rq6c.sh. Results are reported whatever they are.

## Addendum 2026-11-02 (before running; EXPLORATORY, not part of the registered RQ6): cross-family transfer
of an evolved real-gain lineage. The registered RQ6 settings (AWQ T=1.0, bf16 T=0.7) keep the solver
family. As for the planted lineage, we also run the held-out lattice of BIRD metaharness_B3_s1 (the
evolved lineage with a single carrier unit) under Llama-3.1-8B-Instruct (MLX 4-bit, T=0.7, max_tokens
768) on the Mac: all 199 held-out tasks x 4, statement-level tree, own namespace. Reported as exploratory:
Spearman correlation of held-out Shapley values with the AWQ T=0.7 lattice, rank of U2, and the lineage's
gain under Llama. No other lineage is run under Llama and nothing is chosen from the result.
Script: runs/mac_llama_transfer.sh.

## Addendum 2026-11-02 (before running): Phase A4, the reasoning proposer at a second size
Why: Phase A3 (14B proposer, thinking on) produced 3 of 24 real-gain loops, Phases A and A2 (thinking
off) none of 48. A4 asks whether thinking mode also helps a smaller proposer, and adds evolved lineages.
Design: Phase A3 unchanged (prompt v2, thinking on, max_tokens 6144, reasoning text stripped before edits
are parsed, gates B1/B3, seeds 1-4, 30 iterations, 2,500 task-runs, same splits, same solver) except the
proposer: Qwen3-8B-AWQ (the Phase A proposer) served by vLLM on the GPU next to the solver, with an
11,264-token context (raised to 18,432 before any result, DEC-15) (the largest A3 proposer prompt was 3,561 tokens, plus 6,144 output). Experiments
gain_pilot4_{gsmhard,lcb,bird}; GSM-Hard, LCB, BIRD in this order; each followed by the registered D8
gain check (200/85/199 held-out tasks x 4). Phase B (stage_b4 protocol) on every lineage with accepted
edits, pools only if <= 500 conflict-free products. Real-gain A4 lineages enter RQ3/RQ4 and, later,
RQ6 (AWQ T=1.0 and bf16 T=0.7, as registered). Prediction (not a hypothesis test): A4 has at least one
real-gain loop. Outcomes are reported whatever they are; nothing is tuned on held-out data.
Script: scripts/gpu/stage5.sh (Phase B: scripts/gpu/stage_b5.sh).

## Addendum 2026-11-03 (before running; EXPLORATORY, not part of the registered RQ6): cross-family transfer of the remaining lineages
Why: the GPU is paused for three to four days (it is needed for other work), and the Mac is free. The
2026-11-02 addendum ran one lineage (evolved BIRD B3_s1) under Llama and said no other lineage would be
run. This addendum is written after that result was seen, so its outcome is not independent of it; to
avoid choosing lineages from results, it covers every remaining lineage of Table transfer_lineages, in a
fixed order: evolved BIRD B1_s1, evolved LCB B1_s2 (both Phase A3), constructed BIRD, constructed LCB,
constructed GSM-Hard. Same protocol as 2026-11-02: Llama-3.1-8B-Instruct (MLX 4-bit, T=0.7, top-p 0.95,
max_tokens 768), all held-out tasks (199/85/200) x 4, statement-level tree, own namespace, preceded by a
brute-force exactness audit on 10 search tasks x 1. Reported as exploratory for every lineage that
completes before the GPU resumes: Shapley values, top unit, and the lineage's gain under Llama. A
lineage that does not complete is reported as not run. Nothing is chosen or tuned from the results.
Script: scripts/mac_llama_transfer2.sh.

## Addendum 2026-11-06 (before running; EXPLORATORY, not part of the registered RQ6): a second cross-family setting on the GPU
Why: the GPU is free after Phase A4, and the exploratory Llama runs of 2026-11-02/03 use one build of the model
(MLX 4-bit on the laptop). A second build on different hardware and serving stack lets a reader check whether the
cross-family findings depend on the build. Setting: Llama-3.1-8B-Instruct, AWQ INT4
(hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4, revision db1f81ad4b8c7e39777509fac66c652eb0a52f91), served by
vLLM 0.8.5 alone on the A4000 (seed 0, 8,192-token context), temperature 0.7, top-p 0.95, max_tokens 768: the same
sampling as every other setting. Lineages, in this order, all of them: planted (GSM-Hard), constructed GSM-Hard,
LCB, BIRD, evolved A3 LCB B1_s2, BIRD B1_s1, BIRD B3_s1. Each: brute-force audit on 10 search tasks x 1, then the
held-out lattice (200/85/199 tasks x 4), statement-level tree, own namespace; lattice_gain.py for the gain.
Reported for every lineage, whatever the outcome, as exploratory: Shapley values, top unit, gain with CI, and the
agreement with the MLX Llama build (same top unit; sign of each Shapley value). Nothing is chosen or tuned from the
results. Script: scripts/gpu/stage_llama_awq.sh; config: configs/backend_gpu_llama_awq.yaml.

## Addendum 2026-11-09 (before running): RQ1c, branched lineages (cost when runs read few of many units, on real model output)
Why: written after every earlier result was known. On every real lineage so far (evolved, constructed, pooled), almost
every run reads every unit, so exploration ran 88-99% of the brute-force executions and the savings came from replay.
The only evidence that the cost follows the read width rather than the number of units is the synthetic study (RQ1b,
fake LLM). This addendum tests that claim on real model output. The lineages are constructed for this purpose and say
nothing about evolution.
Lineages (examples/branched/, fixed before any run of them; checked only for crashes on 5 search tasks):
- BIRD: seed = the constructed BIRD harness with its repair feedback split by failure class (7 error classes, any other
  error, empty result, all-NULL result), each class reporting the failure plainly. Final = class-specific advice in
  each of the 10 branches, a quoting rule in the system prompt, primary keys in the schema text. 12 units: 2 on the
  main path (read by every run), 10 read only when their failure class occurs.
- LiveCodeBench: seed = the constructed LiveCodeBench harness with its repair feedback split by failure class
  (time limit, compile error, 5 exception types, any other crash, empty output, wrong output, no program in the
  reply). Final = class-specific advice in each of the 11 branches and a recursion rule in the system prompt.
  12 units: 1 on the main path, 11 on failure paths.
Protocol (Phase C protocol, solver Qwen3-4B-AWQ T=0.7 under vLLM, statement-level tree exploration, own namespaces
branched_bird and branched_lcb; BIRD first): exhaustive brute-force audit on 10 search tasks x 1 replicate (4,096
products each, as registered for lineages of at most 12 units); search lattice 60 x 4; held-out lattice (199 / 85
tasks x 4); D8 gain of the final over the seed; fresh-run audit on a random 0.1% of held-out (product, cell) pairs
(the registered 10% would be 326,000 fresh BIRD runs), compared in aggregate: replayed vs fresh accuracy with a 95%
bootstrap interval of the difference; RQ3/RQ4 report (2,000 bootstrap resamples). Script: scripts/gpu/stage_branched.sh.
Predictions, fixed now:
- R1: 0 mismatches in both brute-force audits.
- R2: every run reads at most 4 units on BIRD (2 main path + one class per repair attempt) and at most 3 on
  LiveCodeBench, so no cell needs more than 16 (BIRD) or 8 (LiveCodeBench) executions.
- R3: on held-out cells, exploration runs fewer than 1% of the brute-force executions (4,096 per cell) on both.
- R4: the fresh-run audit's interval for (replayed - fresh accuracy) contains 0 on both.
- R5: every unit that no held-out run reads has a held-out Shapley value, interactions and Möbius terms exactly 0.
Reported whatever the outcome, with the generated and nominal tokens, the wall-clock time and, for comparison, the
executions brute force would need. The gain, the attributions and the distilled product are reported without
predictions. Nothing is tuned; no other branched lineage is run.

## Addendum 2026-11-09 (before running): brute-force audit of every pooled lattice
Why: the registered exactness audit covers every lineage, but the pooled lattices of RQ5 were never audited on their own;
their product lines contain alternatives (conflicting units), which single lineages do not. Protocol: for each of the 10
pooled lattices that were run (A, A2, A3 GSM-Hard and BIRD; A2, A3, A4 LiveCodeBench; A4 GSM-Hard), the registered
audit: 10 search tasks x 1 replicate, every conflict-free product run alone (--brute) and compared with the tree,
same units (same loop logs, DEC-19 rule), same read model as the pooled lattice, the pool's own namespace.
Prediction: 0 mismatches in every pool. Reported whatever the outcome. Runs after stage_branched on the GPU.
Script: scripts/gpu/stage_pool_audit.sh.

## Addendum 2026-11-24 (before running): Phase A5, evolution from multi-step seed harnesses
Why: written after every earlier result. In Phases A to A4 every loop started from a
one-call seed, so every accepted edit sat in code that every run executes, and exploration ran 88-100% of the
brute-force executions on evolved lineages; the large savings were shown only on constructed (branched) lineages.
Phase A5 asks the same questions of lineages that evolution itself produces from seeds with branching control flow.
Design (all else as Phase A3: prompt v2, proposer thinking on, max_tokens 6144, reasoning text stripped before edits are
parsed, gates B1 and B3 (repeats 2), seeds 1-4, at most 30 iterations, 2,500 task-runs, the registered splits, CRN
streams, the D8 real-gain criterion):
- Seed harnesses (existing files, unchanged): src/editlattice/seed_harnesses/bird_react.py (ReAct agent with schema tools,
  at most 8 steps), bird_modular.py (capabilities as primitives activated per question by a router), lcb_repair.py
  (generate, run the public tests, repair up to 3 times). GSM-Hard has no multi-step seed in the code base and is not run.
- Proposer: Qwen/Qwen3-14B-AWQ under vLLM 0.8.5 on the RTX A4000, sharing the card with the solver (Qwen3-4B-AWQ) during
  evolution. To fit both, the proposer keeps its KV cache in fp8 (e5m2; e4m3 is not supported on this GPU) and both
  servers use chunked prefill (proposer 1,024 and solver 2,048 tokens per step); neither changes what a model computes
  for a request beyond KV-cache rounding in the proposer. The proposer is the 14B model of Phase A3 in another
  quantisation (AWQ instead of MLX 4-bit), one request at a time (max_in_flight 1, as in A3). The solver gets 768 output
  tokens on both benchmarks during evolution, as in every evaluation, so the LiveCodeBench mismatch of DEC-16 does not recur.
- Every held-out gain check (199 BIRD and 85 LiveCodeBench tasks x 4) and every Phase B lattice runs afterwards with the
  solver alone on the A4000 (scripts/gpu/serve.sh lattice), the build used by all other phases.
- Phase B on every A5 lineage with accepted edits: statement-level tree exploration, search 60 x 4 and all held-out x 4,
  exhaustive brute-force audit on 10 search tasks x 1 (every product, with --brute-all above 12 units), fresh-run
  audit on 10% of held-out (product, cell) pairs, RQ3/RQ4 report.
Predictions, fixed now:
- F1: 0 mismatches in every brute-force audit of A5 lineages.
- F2: summed over the held-out lattices of A5 lineages, exploration runs fewer executions relative to brute force than
  over the evolved lineages of Phases A to A4 (93%).
- F3: at least one A5 loop is a real-gain loop (A3, the same proposer model with thinking, had 3 of 24).
- F4: in the fresh-run audits, no product's replayed and fresh accuracy differ (Fisher p < 0.05) beyond the rate expected
  by chance at the number of products tested.
Order: bird_react, bird_modular, lcb_repair; the 8 loops of a seed run concurrently, as in earlier phases. Reported whatever the
outcome; nothing is tuned on held-out data; no loop is rerun or dropped because of its result.
Configs: configs/gain_pilot5_{birdreact,birdmodular,lcbrepair}.yaml. Script: scripts/gpu/stage_a5.sh.
