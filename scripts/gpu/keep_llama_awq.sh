#!/usr/bin/env bash
# Unattended runner for stage_llama_awq.sh (addendum 2026-11-06): reruns it until all seven held-out lattices and
# their gains exist, at most 20 attempts (a lineage can fail on a stalled download or a server crash; the stage
# logs DONE even then). Finished lattices are skipped on rerun and stored responses replay. Exclusive lock.
exec 8>runs/locks_keep_llama_awq.lock
flock -n 8 || { echo "already running"; exit 1; }
cd "$(dirname "$0")/../.."
NAMES="planted_llama_awq constructed_gsmhard_llama_awq constructed_lcb_llama_awq constructed_bird_llama_awq phaseA3_lcb_metaharness_B1_s2_llama_awq phaseA3_bird_metaharness_B1_s1_llama_awq phaseA3_bird_metaharness_B3_s1_llama_awq"
complete() {
  for n in $NAMES; do
    [ -f runs/lattice/${n}_stmt_heldout.json ] && [ -f results/gain_lattice_${n}_stmt_heldout.json ] || return 1
  done
}
for attempt in $(seq 1 20); do
  while ! flock -n runs/locks_stage_llama_awq.lock true; do sleep 60; done
  complete && { echo "$(date '+%F %T') all seven lattices and gains present"; exit 0; }
  echo "$(date '+%F %T') attempt $attempt"
  timeout 36h bash scripts/gpu/stage_llama_awq.sh >> runs/logs/stage_llama_awq.log 2>&1 < /dev/null 8>&-
  sleep 120
done
complete && echo "$(date '+%F %T') complete" || echo "$(date '+%F %T') incomplete after 20 attempts"
