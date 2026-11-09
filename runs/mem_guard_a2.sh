#!/bin/zsh
# Laptop guard for the A2 proposer server: kill it when the Mac is really in danger, i.e. free memory
# below 3% or free disk below 3 GB (swap files live on the disk). A 16 GB Mac holding an 8 GB model
# runs at 5-10% "free" normally, so the old 7% rule killed a healthy server every few minutes.
# 2026-11-06: loading a model dips free disk below 3 GB for a few seconds while macOS swaps; the guard
# now acts only after three low readings in a row (about 30 s), so a brief dip no longer kills the server.
LOG=runs/mem_guard_a2.log
low=0
while pgrep -f mlx_server_capped.py >/dev/null; do
  free=$(memory_pressure | awk -F': ' '/free percentage/{gsub("%","",$2); print $2}')
  diskfree=$(df -m ~ | awk 'NR==2{print $4}')
  if { [ -n "$free" ] && [ "$free" -lt 3 ]; } || [ "$diskfree" -lt 3072 ]; then
    low=$((low + 1))
  else
    low=0
  fi
  if [ "$low" -ge 3 ]; then
    echo "$(date '+%F %T') free=$free% diskfree=${diskfree}M (3 readings) -> killing mlx server" >> $LOG
    pkill -f mlx_server_capped.py
    low=0
  fi
  sleep 10
done
