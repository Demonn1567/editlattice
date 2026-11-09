
## DEC-1 (2026-10-27) Per-lineage stream namespace for cost accounting
The Mac RQ1 runs shared streams across lineages of one seed, so generated-token savings mixed
within-lineage replay with cross-lineage and repeated-run reuse. From now on run_lattice uses the
lineage name as stream namespace by default; recombination lattices use their own namespace. Exactness
is unaffected; cost numbers in the paper come only from namespaced runs.

## DEC-2 (2026-10-27) Proposer: Qwen3-8B-AWQ (pre-registered fallback), not Qwen3-14B-AWQ
The pre-registration names Qwen3-14B-AWQ with Qwen3-8B-AWQ as the fallback "only if the 14B model
cannot be served next to the solver". On the 16 GB RTX A4000 it cannot: 14B-AWQ weights take
9.36 GiB; at gpu-memory-utilization 0.68 vLLM had no memory for KV blocks, and at 0.72 with
--enforce-eager and 2,048-token prefill chunks only 0.58 GiB of KV cache remained, while one
8,192-token request needs 1.25 GiB (logs: runs/logs/vllm_proposer.log, 12:43 and 12:56). Raising the
proposer's share far enough would leave the solver ~0.3 GiB of KV cache, below one LiveCodeBench
request. Alternating the two models per iteration (~90 s load) or serving the 14B from the Mac
(~10 min per proposal) is infeasible at pilot scale. The 8B-AWQ proposer is used for every loop.

## DEC-3 (2026-10-27) Edit application tolerates a uniform indentation shift (before the pilot)
In the GPU smoke test 3 of 6 proposals by the 8B proposer failed with "SEARCH text not found": the
proposer copied the right code but indented every line by the same extra amount. If an exact match
fails, the common indentation is now removed from SEARCH and REPLACE alike, and the edit is applied
only if the shifted SEARCH occurs exactly once (edit_format.shift_indentation; tests cover the
non-uniform and ambiguous cases, which remain errors). All 3 failed smoke proposals then apply.
Decided before any pilot loop ran; the same rule applies to every loop.

## DEC-4 (2026-10-27) First-line indentation tolerance; LCB pilot restarted (before any LCB gate decision)
After 69 minutes of the LCB pilot, all 4 proposals so far (3 loops) had failed with "SEARCH text not
found", for the same reason. The proposer copied the module-level multi-line `SYSTEM_PROMPT`
assignment exactly but indented only its first line by 4 spaces. The continuation lines of a
string literal have no indentation to copy, so DEC-3's uniform shift cannot apply. New rule
(edit_format.shift_first_line): if both the exact match and the uniform shift fail, the extra
indentation is removed from the first line of SEARCH and REPLACE. The edit applies only if REPLACE's
first line has the same indentation, the result occurs exactly once, and it starts a line.
Otherwise it is still an error.
Check: all logged proposals were re-applied under the new rules.
- LCB: 4 of 4 then apply.
- GSM-Hard: no proposal changes from invalid to valid (75 invalid stay invalid), so the GSM-Hard
  pilot is identical under the new rule and is not rerun.
The LCB pilot was stopped and restarted from scratch, with all 8 loops under the same rule. No
LCB proposal had reached a gate and no held-out data was used, so the restart depends on no outcome.
The aborted run is kept in runs/gain_pilot_lcb_aborted_dec4.

## DEC-5 (2026-10-27) MILP node formulation; scaling study rerun
Run 1 of the RQ1b scaling study (runs/scaling_run1_leafmilp.log, 25 of 28 rows) confirmed P1-P3 and P5,
but not P4 (compositional static analysis grows roughly like m^2, because each group check re-parses
the whole module; still polynomial against 2^m) and not P6. The per-(leaf, unit) MILP took 163 s
(m = 10) and 2,228 s (m = 12) when every run reads every unit, and did not finish m = 14 in over an
hour. The MILP now uses node constraints (each internal node of a cell's tree bounds the leaves below
its two branches by x_u and 1 - x_u, as in tree-ensemble optimisation). It is exact for integral x and
tighter: on the m = 10 worst case it gives the same optimum in 42 s against 456 s (both measured
while the study ran in parallel). Run 1 was stopped at the m = 14 dense row and the full grid is rerun
with the node formulation. Both runs are reported. Exploration, exactness and attribution do not
depend on the MILP.
Addendum: `TreeLattice.optimise` also enumerates exactly (filling each leaf's subcube: 2^m work per
cell, since the leaves partition the cube). `auto` picks enumeration when 2^m <= 4 x leaves and
m <= 24, else the MILP. On the m = 12 worst case the node MILP took 222 s (leaf MILP 2,228 s).
Run 2 times the MILP (all rows except the dense rows with m > 12) and enumeration (m <= 24), and
checks that both give the same optimum value and the same distilled size.

## DEC-6 (2026-10-28) Proposer outages are retried, not logged as invalid proposals (before Phase A2)
With the proposer on the Mac behind an SSH tunnel, a dropped connection would otherwise be logged as
an invalid proposal (after the client's 8 retries) and silently waste an iteration. Now the loop
waits 60 s and asks again, restoring the random generator so the retried prompt uses the same
examples and stream, for up to 12 hours, then fails the job loudly. Test: a loop interrupted by
three outages logs exactly the same entries as an uninterrupted one. Phase A had one proposer HTTP
400 on LCB (a rejected request, not an outage), which stays an invalid proposal under both rules.
Addendum (2026-10-28 12:39 IST): the 14B server was killed by the laptop memory guard three times
(free memory <= 6%) within 45 minutes. It now runs through runs/mlx_server_capped.py, which only caps
MLX's buffer cache at 512 MB (freed buffers released sooner; generation unchanged), with its own guard
(runs/mem_guard_a2.sh, same 7% threshold). Proposals interrupted by kills were retried under DEC-6.
Addendum (2026-10-28 13:10 IST): the cache cap did not stop the kills (6 more in 17 minutes, all at
free = 4-6%). macOS normally runs a loaded 16 GB machine at 5-10% "free" (it compresses and swaps
first), so the guard now kills only when free memory is below 3% or free disk below 3 GB.

## DEC-7 (2026-10-28) Pooled lattices of the null-gain group: which pools are run (before running any)
Per-lineage lattices showed that 90-100% of executions read every unit (units sit in prompts, answer
parsing and code extraction, which every run executes), so a pooled lattice costs about (conflict-free
products) x (cells). Pools (all loops with accepted edits of one pilot): GSM-Hard A: 10 units, 14
conflicts, 124 products; GSM-Hard A2: 8 units, 34 products; BIRD A: 11 units, 300 products; LCB A: 17
units, 16 conflicts, 5,184 products. Run at the pre-registered size (60 search x 4, all held-out x 4):
GSM-Hard A and A2 (their units only parse answers, so all products share requests and replay) and BIRD
A (short runs; run last). Not run: the LCB pool (~3 million executions, mostly new generations because
its units change prompts); reported as infeasible with these numbers, the 2^w limit of Section 6.
Chosen from cost alone, before any pooled result.
Addendum (2026-10-29 00:30 IST): the Mac sleeps daily 10:00-16:00 IST. A2 loops then wait (DEC-6;
same examples and streams on resumption, nothing lost). The tunnel's remote shell records its pid and
the supervisor ends only that stale session before reconnecting (a sleeping Mac leaves the GPU's sshd
holding port 8101); tested by ending the session from the GPU side: reconnection in 22 s. BIRD A2's
proposer request timeout is 900 s instead of 3600 s (recovery speed only). All A2 logs so far contain
0 "proposer error" entries.

## DEC-8 (2026-10-29) Phase B for the A2 lineages (before running)
Same pre-registered protocol as the Phase A null-gain group: per-lineage lattices of the 13 A2 LCB and
BIRD lineages with accepted edits (units: LCB 2,1,2,1,1,3,2; BIRD 4,1,1,2,2,1), then the pooled A2
BIRD lattice (11 units, 6 conflicts, 432 products) and the pooled A2 LCB lattice (11 units, 33
conflicts, 62 products), both at the pre-registered size (unlike the Phase A LCB pool of DEC-7, these
are affordable).
Exploratory, labelled as such (not pre-registered): the pooled Phase A BIRD optimum (units 0, 6, 9,
from two loops) vs the seed on held-out cells: +3.4 points [-0.6, +7.5] (task bootstrap, 199 tasks);
not a real gain under D8. RQ5 (vs best single final): +2.0 [-0.1, +4.3], not certified.

## DEC-9 (2026-10-29) RQ6 transfer on the planted lineage (before running)
RQ6 was registered for real-gain lineages; there are none (0 of 48 loops). It is run on the planted
lineage (4 units with known structure), the only lineage whose attribution is not exactly zero, under
the registered transfer settings: solver Qwen3-4B-AWQ at temperature 0.7 (main), the same at
temperature 1.0, and Qwen3-4B bf16 at 0.7 (served on port 8100 after the idle 8B proposer server was
stopped; its weights re-downloaded and checked against Hugging Face SHA-256 after an earlier partial
download was found). Each: held-out 200 x 4 with statement-level tree exploration; AWQ 0.7 also on 60
search tasks x 4 (RQ3/RQ4), with the 10-task brute-force audit and the 10% fresh-run audit. Spearman
correlation of held-out Shapley values across the three settings and with the Mac runs (Qwen3-4B 4-bit
MLX, Llama-3.1-8B 4-bit), reported per unit since m = 4.

## DEC-10 (2026-10-29) Soundness bug in statement-level markers: found, fixed, affected lattices rerun
Found by the pre-registered brute-force audit: phaseA_lcb_metaharness_B3_s2_audit reported 2
mismatches (product {0,1,2} received the run of {0,1} on two cells; reproduced deterministically).
Cause: unit 2's hunk replaces a line inside an inner `if` block and adds statements one level up; the
marker sat inside the inner block, so execution could reach the added statements without passing it.
The four syntactic marking checks do not see this. Fix (units.add_markers): a unit is marked only if,
in every product, every line of each of its regions belongs to statements that follow the region's
marker directly in the marker's own block (so every execution reaching the region executes the marker
first). Regression test (tests/test_variational.py) fails without the fix and passes with it; 107
tests pass. Re-check of every stored statement-level lattice (scripts/check_markers.py, and the 20
prior lineages rebuilt from their logs): affected only phaseA_gsmhard_metaharness_B3_s3 (unit 1),
phaseA_lcb_metaharness_B3_s2 (unit 2) and prior rq5s_bird_metaharness_B1_s5 (unit 0); planted
lattices (Mac, Llama, tree) and all 27 scaling configurations unaffected. Affected lattices are rerun
with the fix; the old files are kept with the prefix buggy_ and both are reported. The audit that
found the bug was a 10-task brute-force check; the other affected lattices passed theirs, so the
paper states that the audit is necessary, not that it is sufficient.
Result of the DEC-10 reruns: brute-force audits 0 mismatches. Every product-cell outcome of the
fixed lattices equals the buggy ones (GSM B3_s3: 0 of 960 search / 3,200 held-out differ; LCB B3_s2:
0 of 1,920 / 2,720), so attribution is unchanged; the two original audit mismatches were in the
answer field (both programs scored 0). The buggy runs undercounted executions (GSM B3_s3 954 -> 960,
3,154 -> 3,200; LCB B3_s2 1,103 -> 1,920, 1,576 -> 2,720): cost numbers use the fixed lattices.
Prior rq5s B1_s5: same 40 executions, 0 mismatches.
Addendum to DEC-9 (2026-10-29): the bf16 server failed three times with "no memory for the cache
blocks": vLLM 0.8.5 V1 counts memory held by other processes (the AWQ solver, 6.6 GB) against this
instance's gpu_memory_utilization, so the fraction must cover both (0.97; 1,536-token context and batch, 8
sequences). The AWQ solver and all other stages were unaffected (separate process).

## DEC-11 (2026-10-30) Server-rejected requests were scored as failed runs; two transfer runs invalid
The bf16 server for RQ6 ran with a 1,536-token context (to fit next to the AWQ solver, DEC-9
addendum). Requests longer than that were rejected (HTTP 400) and, because the harness caught the
exception, scored as failed runs: 1,239 of 2,720 executions in constructed_lcb_bf16 and 2,881 of
6,368 in constructed_bird_bf16 (found because the BIRD lineage showed -18.8 points under bf16).
These two lattices are invalid (kept as invalid_*) and excluded; they are rerun with a full context
after Phase A3 frees the GPU. A scan of every lattice, evolution record and seed evaluation on both
machines found no other rejected request (planted_gpu_bf16 and constructed_gsmhard_bf16: 0).
Fix (llm/client.py): a rejected request now marks the run as an infrastructure failure, so it raises
(bounded retries) and is never scored; test in tests/test_infra_failure.py (109 tests pass).

## DEC-12 (2026-10-30) KernelSHAP could exceed its budget by one sample
In RQ2, KernelSHAP checked the budget before drawing a coalition but not the cost of evaluating it,
so the last draw could overshoot by one sample's cells (independent arm of Phase C: 6,650 and 6,625
runs at a 6,400 / 6,368 budget; the permutation estimator already discarded overruns). It now prices
each draw first (free if the coupled sampler has the value) and stops if it would exceed the budget.
Test (cells per value that do not divide the budget) fails without the fix, passes with it. RQ2
rerun everywhere: planted (Mac) unchanged in every row (its budgets divide evenly); Phase C rerun.

## DEC-13 (2026-10-30) Concurrent duplicate runs could overwrite cached responses; scores shared across processes
Found by the registered RQ1 name-level vs statement-level comparison (same namespaces, so outcomes must
agree): 19 product-cells differed in 5 of 23 lineages.
(1) planted_gpu (4 product-cells, one cell): the RQ6 stage had been launched several times while the bf16
server failed to start; earlier launches were still waiting when it came up and ran the same lattices
concurrently. The solver log shows three requests with the same prompt and seed within four seconds; the
cache kept the last response (768 tokens, truncated, outcome 0) while the lattice used the first (414
tokens, outcome 1). Fix: ResponseCache.put_if_absent (first writer wins, atomic hard link) and the client
uses the stored response, so every consumer of a key sees one response even under concurrent runs; stage
scripts take an exclusive lock. Test fails with the old behaviour, passes with the new. All planted RQ6
lattices are rerun under the lock (cached responses now unique per key).
(2) LCB lineages (15 product-cells over 4 lineages): LiveCodeBench scores programs with time limits; scores
were cached per process, so a program near its limit could score differently in separate runs. Each
lattice is internally consistent (one score per answer within a run). Fix: scores are cached on disk
across processes, first writer wins (runner.cached_score); test added. 112 tests pass.
Phase A3 (running since before the fix) keeps its loaded code; the fixes apply to every later process.

## DEC-14 (2026-10-30) Phase B for the A3 lineages; pool size rule fixed in advance
A3 lineages get the same Phase B protocol as A and A2 (stage_b4.sh, locked), each benchmark after its A3
gain check. Pools run only with at most 500 conflict-free products (every pool run so far had at most
432; the Phase A LCB pool with 5,184 was skipped, DEC-7); larger pools are reported with their size.
Fixed before any A3 pool was built.
Addendum (2026-10-31 20:53 IST): A3 throughput fell to about one proposal per hour because the Mac was
thrashing (4 GB swap; the 14B weights were paged out during generation, CPU ~44%). The server now wires
9.5 GB with mx.set_wired_limit (below MLX's recommended working set of 10.7 GB on this 16 GB machine),
so macOS swaps other processes instead of the weights. Memory management only; generation is unchanged.

## DEC-15 (2026-11-02) Phase A4 restarted with a larger proposer context, before any result
The addendum sized the 8B proposer's context at 11,264 tokens from the A3 logs (largest prompt 3,561
tokens). Those counts understate the prompt: the first GSM-Hard prompt was 8,117 tokens under the 8B
tokenizer, so prompt plus the 6,144-token reasoning budget (14,261) exceeded the context. vLLM rejected
4 requests (HTTP 400) within the first minutes, and the loop logged one of them as an invalid proposal,
which would have cost a loop an iteration for an infrastructure reason. A4 was stopped (no gain check had
run, no lattice existed), the partial GSM-Hard directory was set aside as discarded_dec15_*, and A4 was
restarted from scratch with the proposer at 0.62 of the GPU and an 18,432-token context, the solver
started second with a whole-card budget of 0.93 (vLLM counts the proposer's 9.3 GB against the second
server's utilization; at 0.34 to 0.38 it had no memory left for its cache). The solver's settings are
otherwise those of Phases A to A3 (32 sequences, 8,192-token context).
The response store kept the proposer responses that the discarded start had received; requests that recur
identically on the same stream are answered from it (the same model and sampling parameters; the context
length does not affect generation). The rejected requests had no stored response and are sent anew. Everything else is as registered. Any further HTTP 400 from the proposer stops A4 for investigation.

## DEC-16 (2026-11-03) LiveCodeBench solver output budget: 2,048 tokens in evolution, 768 in evaluation
Found while generating the settings appendix from the configuration files. Since the configs were written
(b432d6a, 2026-10-27), the LiveCodeBench evolution configs of every phase give the solver max_tokens 2,048,
while every held-out gain check and every LiveCodeBench lattice uses configs/backend_gpu.yaml with 768
(GSM-Hard and BIRD use 768 throughout). The registration does not fix the solver's output budget. Within
each evaluation the seed and every product share the setting, so all reported contrasts are internally
consistent; but loops were selected under a larger budget than they were judged under, and 768 tokens can
cut long programs (see DEC-13's 768-token truncation). Response: reported as a deviation, and a sensitivity
analysis reruns the LiveCodeBench gain checks of Phases A, A2 and A3 with 2,048 tokens (results/sens_lcb2048/,
scripts/gpu/stage_sens.sh). The registered analysis stays the 768-token one; both are reported whatever
they show. GSM-Hard and BIRD are unaffected.

## DEC-17 (2026-11-03) GPU pause of three to four days during Phase A4
The GPU was needed for other work. At 13:57 UTC both stage scripts (stage5, stage_b5), their Python jobs
and both vLLM servers were stopped by PID, and runs/ was copied to ~/editlattice_snapshots/2026-11-03_preempt
on the GPU machine (A4 data also copied to backups/gpu_2026-11-03 on the Mac). State at the stop: A4
GSM-Hard complete (8 loops, gain check 0 of 8 real, Phase B on its 4 lineages with accepted edits complete,
reports written); the pooled A4 GSM-Hard lattice had run 10 minutes and has no output; the 8 A4 LCB loops
were mid-run (7 to 14 proposals each); A4 BIRD not started. Nothing is lost: response-store writes are
atomic and first-writer-wins, a finished loop is never rerun, and a restarted loop replays its earlier
iterations from the store and rewrites its log, as after DEC-15. stage_b5 now skips lineages whose report
exists and pool lattices that exist. Resume: notes/resume_after_pause.md. During the pause the Mac runs
the exploratory Llama transfer of the 2026-11-03 addendum.

## DEC-18 (2026-11-04) BIRD scorer reads the predicted result row by row
The exploratory Llama lattice of BIRD B1_s1 ran for six hours on the Mac, then was killed by macOS at every
restart within a minute (exit 137): one Llama answer's result is so large that fetching it whole for the
set comparison needed about 15 GB (peak footprint measured with /usr/bin/time). The scorer now reads the
predicted rows in batches and stops at the first row outside the gold set. The verdict is unchanged: such a
row makes the sets unequal whether the query would later finish or exceed its budget, and both give 0.
Check: scripts/check_bird_scorer.py rescored all 8,318 distinct (task, answer) pairs in the stored records
of Phases A, A2 and A3 BIRD and agreed with every stored outcome (0 disagreements); 112 tests pass. Stored
scores and lattices stay valid. The GPU copy of the code takes this change when it resumes (DEC-17).

## DEC-19 (2026-11-06) Recombination: an insertion at the first line of another unit's replaced range is a conflict
Found when the pooled A4 LCB lattice stopped with InvalidProduct (product [3, 7] is not a valid harness). Unit 3
(one run) inserts lines at seed line 13 and unit 7 (another run) replaces seed line 13. hunks_conflict treated an
insertion as conflicting only strictly inside a replaced range, so the two were combinable; but applying them is
ambiguous: in one order the replacement overwrites the inserted lines and keeps the seed line (here, breaking a
triple-quoted prompt). Such ties cannot occur within one lineage (net hunks of one diff never touch), only across
runs. Fix: an insertion at the first line of a replaced range now conflicts (the units become alternatives, as for
overlapping rewrites; Section "Recombination" of the paper already says units rewriting the same seed lines are
alternatives). Regression check of every pool: A, A2, A3 on all three benchmarks unchanged (no new conflict);
A4 GSM-Hard gains one alternative pair {2, 5} (its products were valid harnesses but not the union of the two
edits), A4 LCB gains {0, 3} and {3, 7} (16 invalid products -> 0). The A4 GSM-Hard pool lattices are rerun under the
fixed rule (previous files kept as *_pre_dec19.json); A4 LCB is run for the first time; A4 BIRD is checked on the
GPU machine (one of its harnesses uses Python 3.12 f-string syntax that the Mac's 3.11 cannot parse). Test:
tests/test_recombination.py::test_insertion_at_start_of_replaced_range_conflicts.

## DEC-20 (2026-11-09): pooled-lattice audits check every product
The registered pool audit (addendum 2026-11-09) checks every conflict-free product. run_lattice.py --brute checks every
product only up to 12 units and 64 random subsets per cell above that; on the 13-unit A3 BIRD pool most of these subsets
conflict, so its first audit checked 39 pairs (0 mismatches). New flag --brute-all checks every feasible product; the
A3 BIRD pool audit is rerun with it (the sampled audit is kept as pooledA3_bird_audit_sampled). The other nine pools have
at most 12 units and were already checked in full. No result was seen that this could change: the sampled audit had 0
mismatches.
