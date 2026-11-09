# Theory (draft for the paper), 2026-10-27

## Setting
- Seed harness H (a Python module). Net units U = {1..m} (`units.build_units`): disjoint changed
  line ranges of seed -> final, merged until every subset gives a valid harness. Product family
  F ⊆ 2^U: all subsets (one lineage) or the conflict-free valid subsets (recombination,
  `units.combine_lineages`). H_S is the seed with the units in S applied.
- Task x, stream s. A run of a harness on (x, s) sends LLM requests; the k-th occurrence of
  request q receives the stream's response r_s(q, k). Outcome C(S, x, s) ∈ {0, 1}.
- Assumptions. A1: environment calls and scoring are deterministic given their inputs. A2: a
  response depends only on the request (and stream, occurrence). A3: runs share no mutable state
  (fresh module, fresh environment). A5 (new): module-level code of the harness has no effect
  other than binding names (no I/O, no LLM or environment calls at import); a differing top-level
  statement that does not bind a name makes its unit global.

## Read model
For each top-level name n, dep(n) ⊆ U is the set of units u such that toggling u changes the
normalised definition of n for some pair of products in F, closed under names that n's
definition evaluates at import time (decorators, defaults, class bodies, assignment right-hand
sides), to a fixpoint (`variational.ReadModel`). G ⊆ U are units that change non-binding
top-level code, and all units if any product uses dynamic globals or introspection.

Instrumented product Ĥ_S: every load, inside a function scope, of a global name n with
dep(n) ≠ ∅ is replaced by `__el_get__("n")`, which logs dep(n) and returns the binding (or
raises the same NameError). Statement-level mode additionally inserts, for each marked unit u, a
statement `__el_mark__(u)` at the start of each of u's regions in both versions; a unit is marked
only if (i) every product parses, (ii) every marker is inside a function body, (iii) deleting
the marker lines gives the unmarked product text exactly, (iv) toggling u never changes any
function's scope signature (locals, declared globals, free variables, generator/coroutine flag), and
(vi, DEC-10) in every product every line of each region belongs to statements directly following the
region's marker in the marker's own block.
The reads R(S, x, s) of a run are G ∪ dep(solve) ∪ the logged sets.

## Lemma 1 (instrumentation preserves behaviour)
If no product uses introspection (traceback, inspect, sys._getframe) or dynamic globals, the run
of Ĥ_S on (x, s) sends the same requests, receives the same responses and returns the same answer
and outcome as the run of H_S. Otherwise Ĥ_S := H_S (plain sources), so the claim is trivial.
*Proof.* The accessor returns exactly the object a global lookup returns (module dict, then
builtins) or raises a NameError with Python's message; it changes no state other than the read
log. Markers call a function that only logs. Line numbers and file layout differ, but only code
that inspects frames or tracebacks can observe them, and such harnesses are run unmodified.
Instrumented code is compiled under the same content-addressed file path a plain run uses.
(Residual: one transient extra call frame per accessor call, observable only through recursion
depth limits; not used by harnesses.) Checked empirically by the plain-vs-instrumented tests.

## Lemma 2 (variational replay)
Fix (x, s), S ∈ F, and let R = R(S, x, s). For every S' ∈ F with S' ∩ R = S ∩ R, the runs of Ĥ_S
and Ĥ_S' are identical step by step (same requests, responses, answer, outcome).
*Proof sketch.* Induction over executed steps of the two runs, which start from fresh modules
(A3). Import: by A5, module execution only binds names; the bindings may differ for names whose
dep is not in R, but such names are never loaded at run time (a load of n inside a function goes
through the accessor and would add dep(n) to R), and import-time uses are part of the closure.
Non-binding top-level differences are in G ⊆ R. The runner calls solve, and dep(solve) ⊆ R, so
both runs enter the same function definition. Inductive step: the current statement belongs to
a function definition reached through a name with dep ⊆ R (or a name that is identical in all
products), hence has the same text in both runs, except inside regions of marked units; a marked
region is entered only after its marker executed, which put u in R, so S and S' agree on u and
the region has the same text. Scope-signature stability (iv) means statements outside regions
resolve names identically. An LLM request is then identical, so both receive the same response
(A2, shared stream); environment calls are identical and return identical results (A1, A3).
Objects reached through attributes were created by code already shown identical. ∎
For recombination families, S and S' are connected by removing S \ S' and adding S' \ S one unit
at a time; every intermediate set is a subset of S or of S', hence conflict-free (validity is
checked by enumeration), so the single-toggle construction of dep(n) suffices.

## Proposition 3 (exploration is exact; cost)
`explore_cell` returns, for every S ∈ F, a record identical to running S alone, and performs one
execution per element of the cover it builds; this is at most |F| and equals the number of
distinct read classes it encounters. Generated tokens per execution are only those after the
first request not already in the stream's cache (replay), i.e. after the run diverges from every
earlier execution on the stream.
*Proof.* Each iteration runs some pending S and removes every pending S' with S' ∩ R = S ∩ R; by
Lemma 2 their records equal S's record; S itself is removed, so the loop terminates. ∎
Empirical checks: toy, fuzz (25 random lineages, both read modes), recombination, introspection
and 560 product x cell comparisons on real BIRD lineages: no mismatch. Negative controls (unread
unit, silent markers) are detected.

## Proposition 4 (exact attribution; unbiasedness; variance)
v(S) = mean over the sampled cells of C(S, x, s). Shapley values, interaction indices and Möbius
dividends are linear functionals of v and are computed exactly from the lattice
(`attribution.py`). Each run's marginal law is that of a fresh run of H_S (its draws are fresh
stream draws), so v(S) is unbiased for the population accuracy of H_S on the split's task
distribution and every linear functional is unbiased for its population counterpart. For a pair
of products, the paired difference on a cell is non-zero only if the runs diverge, so
Var(C(S ∪ {u}) − C(S) | x) ≤ P(runs diverge | x); coupling makes
all lattice contrasts low-variance simultaneously. Uncertainty: task-cluster bootstrap that
resamples tasks once per draw and recomputes the whole lattice (`attribution.bootstrap`).

## Proposition 5 (certified distillation)
Let D be chosen on the search set (any procedure). On held-out cells (tasks uniform from the
held-out split, fresh streams, independent of the search set), pairs Δ_i = C(D) − C(full) on
shared streams are i.i.d. with mean μ. The betting e-process for H0: μ ≤ −δ (`distill.
certify_non_inferior`, aGRAPA bets, clipped) is a non-negative supermartingale under H0, so by
Ville's inequality P(certify | μ ≤ −δ) ≤ α at any stopping time. Selection on the search set does
not bias the held-out test because the held-out cells are independent of it. Simulated type-I
check in tests: ≤ α + 0.03 at the margin.
Caveat: with a finite set of held-out tasks and streams drawn without
replacement the guarantee is approximate; use many streams per task (cheap under replay).

## Lemma 2' (ordered reads)  [D11, 2026-10-27]
Let the run of Ĥ_S on (x, s) first-read the units r_1, r_2, ..., r_k in this order (units logged by
one event are ordered by index). If S' ∈ F agrees with S on r_1..r_j, then the run of Ĥ_S' is
identical to the run of Ĥ_S up to the event at which the latter first reads r_{j+1} (or to the
end if j = k), and at that event it also reads r_{j+1}; hence its first j+1 reads are r_1..r_{j+1}.
*Proof.* The induction of Lemma 2 needs agreement only on the units read so far: before r_{j+1}
is first read, every executed statement depends only on units in {r_1..r_j}. At the event itself
both runs execute the same statement in the same state, so both log the same set. ∎

## Proposition 6 (tree exploration: exactness, partition, cost)
`tree.explore_tree` maintains branches (prefixes of (unit, present) decisions). For a branch
P = ((u_1, b_1), .., (u_j, b_j)) it runs the product S_P = {u_i : b_i = 1}, checks that the run's
first j reads are u_1..u_j (otherwise A1 is violated: `DeterminismViolation`), records a leaf
(A = read units present, B = read units absent) and pushes, for each later read r_t (t > j), the
branch P + (r_{j+1},0) .. (r_{t-1},0) + (r_t,1). Branches whose present units contain a
conflicting pair are skipped.
(a) Exactness: every S' in a leaf (A ⊆ S', B ∩ S' = ∅) has exactly the leaf's run (Lemma 2', j = k).
(b) Partition: every S' ∈ F lies in exactly one leaf. Existence: follow S' from the root; at a
    leaf whose run read r_1..r_k, either S' agrees on all of them (S' in the leaf) or at the first
    disagreement r_t, S' agrees with the branch pushed for r_t, by Lemma 2' its run has the same
    prefix, and we recurse into a strictly deeper branch. Uniqueness: two leaves differ in the
    value of the first unit at which their branch paths split.
(c) Cost: one execution per leaf. Every root-to-leaf path has length = the leaf's read count, so
    #leaves ≤ 2^w with w the largest number of units one run of the cell reads (the read width),
    and ≤ |F|. No step enumerates F. The number of leaves equals the number of read classes of
    Proposition 3 (both are the partition of F by S ∩ R(S)); checked empirically (tests).

## Proposition 7 (closed-form exact attribution on trees)
v(S) = Σ_cells w_c Σ_{leaves ℓ of c} y_ℓ 1[A_ℓ ⊆ S, B_ℓ ∩ S = ∅] for all S ⊆ U (one lineage: F = 2^U).
The game g_ℓ(S) = 1[A ⊆ S, B ∩ S = ∅] = Π_{a∈A} x_a Π_{b∈B} (1 − x_b) has Möbius dividends
m(A ∪ C) = (−1)^{|C|} for C ⊆ B, and 0 elsewhere. Using Σ_k C(q,k)(−1)^k/(p+k) = B(p, q+1):
- Shapley: φ_i = B(a, b+1) for i ∈ A, −B(a+1, b) for i ∈ B, 0 otherwise (a = |A|, b = |B|).
- Shapley interaction (Grabisch-Roubens), I_ij = Σ_{T ⊇ ij} m(T)/(|T|−1): B(a−1, b+1) for i,j ∈ A;
  −B(a, b) for i ∈ A, j ∈ B; B(a+1, b−1) for i,j ∈ B.
- Banzhaf: ±2^{−(a+b−1)}; any dividend m(T) directly.
All are linear in v, so the lattice's values are the weighted sums over leaves: exact, in
O(#leaves · w^2) time and without evaluating v on 2^m sets. (These are the interventional
TreeSHAP forms for background = seed and explicand = full harness; the trees here are execution
trees produced by the program, not fitted models.) The task-cluster bootstrap re-weights per-task
totals (linearity), so each draw costs O(#tasks · m^2). Checked against brute force (tests).

## Proposition 8 (exact distillation and recombination over the trees)
With x ∈ {0,1}^U and z_ℓ ∈ [0,1]: z_ℓ ≤ x_a (a ∈ A_ℓ), z_ℓ ≤ 1 − x_b (b ∈ B_ℓ), Σ_{ℓ ∈ c} z_ℓ = 1 per cell,
pairwise conflict constraints x_u + x_v ≤ 1. For integral x exactly one leaf per cell has all
upper bounds 1 (partition), so z is forced to its indicator and Σ w_c y_ℓ z_ℓ = v(x). Hence
max v(x) (recombination) and min |x| s.t. v(x) ≥ v(U) − δ (distillation) are exact MILPs
(`TreeLattice.optimise`, HiGHS with zero gap), with as many binaries as units.

## Proposition 9 (compositional static analysis)
`units.independent_groups` cuts the seed into top-level statements and gaps, joins segments
touched by one hunk into regions and units touching a common region into groups. If every
variant of every region parses on its own (checked; else fall back to all subsets), the module
body of H_S is the concatenation of region bodies, each depending only on S ∩ (its units). Every
static check we run is per top-level statement (binding definitions and their import-time
loads, non-binding top-level code, dynamic constructs, marker placement and AST equality,
per-function scope signatures), and asks either "for all S" or "exists S, u": both quantify
over each region's variants independently, which the subsets of each group with all other units
absent enumerate. So dep, G, the dynamic flag and the marked set equal the exhaustive ones (also
checked on 31 fuzz lineages), at cost Σ_groups 2^{|group|} parses instead of 2^m. Remaining
global check: every executed product is compiled (and a lazily built product that fails to
compile raises), which guards against the corner cases a per-region parse cannot see
(`from __future__` placement).

## Open items
- Expected cost: E[#leaves] in terms of per-unit reach probabilities (worst case 2^w is Prop. 6c).
- Whether a minimum cover order exists that is computable online (current order: smallest first).
