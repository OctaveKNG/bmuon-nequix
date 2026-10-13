#!/usr/bin/env bash
# One training run spread over several machines (data parallel, one GPU per machine).
#
#   bash bmuon/experiments/launch_dist.sh check  "anchois ablette"
#   bash bmuon/experiments/launch_dist.sh start  NAME CONFIG "anchois ablette ..." [--set k=v ...]
#   bash bmuon/experiments/launch_dist.sh status NAME "anchois ablette ..."
#   bash bmuon/experiments/launch_dist.sh stop   "anchois ablette ..."
#
# The first host is process 0 (coordinator, writes logs and checkpoints to bmuon/runs/NAME,
# which must be on the shared home). Every host needs the environment and the data in
# /tmp/$USER (bmuon/experiments/setup_machine.sh; `check` says which ones are missing).
# The global batch is batch_size x grad_accum_steps x number of hosts: keep it at 256
# (e.g. 4 hosts: batch_size=64 grad_accum_steps=1; 16 hosts: batch_size=16 grad_accum_steps=1).
# If a host dies the others stop; `start` again with the same arguments resumes from the
# last epoch.
set -euo pipefail
USER="${USER:-$(id -un)}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="/tmp/$USER/nequix-venv"
DATA="/tmp/$USER/bmuon-data"
SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10)

cmd="${1:-}"; shift || true

case "$cmd" in
check)
  for h in $1; do
    out=$("${SSH[@]}" "$h" "test -f $VENV/.bmuon_ready && echo env || echo NO-ENV; \
      test -f $DATA/mptrj-aselmdb/.complete && echo data || echo NO-DATA; \
      test -d $REPO && echo repo || echo NO-REPO; \
      nvidia-smi --query-gpu=name,memory.used --format=csv,noheader" 2>&1 | tr '\n' ' ') \
      || out="unreachable"
    echo "$h: $out"
  done
  ;;
start)
  name="$1"; config="$2"; hosts=($3); shift 3
  extra=("$@"); [ "${extra[0]:-}" = "--set" ] && extra=("${extra[@]:1}")
  n=${#hosts[@]}
  port=$((12000 + $(echo -n "$name" | cksum | cut -d' ' -f1) % 1000))
  coord="${hosts[0]}:$port"
  echo "run $name on $n hosts, coordinator $coord"
  for i in "${!hosts[@]}"; do
    h="${hosts[$i]}"
    log="/tmp/$USER/dist-$name-rank$i.log"
    "${SSH[@]}" "$h" "cd $REPO && source $VENV/bin/activate && \
      export BMUON_DATA=$DATA XLA_PYTHON_CLIENT_PREALLOCATE=false && \
      nohup python -m bmuon.train $config --coordinator $coord --num-processes $n \
        --process-id $i --set out_dir=bmuon/runs/$name ${extra[*]} > $log 2>&1 < /dev/null &"
    echo "  rank $i on $h  (log $h:$log)"
  done
  ;;
status)
  name="$1"; hosts=($2)
  for i in "${!hosts[@]}"; do
    h="${hosts[$i]}"
    line=$("${SSH[@]}" "$h" "pgrep -u $USER -f '[b]muon[.]train' >/dev/null && echo -n 'running | ' || echo -n 'stopped | '; \
      grep -v '^[WIE][0-9]' /tmp/$USER/dist-$name-rank$i.log 2>/dev/null | tail -n 1 | cut -c1-160; \
      nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader" 2>&1 | tr '\n' ' ') \
      || line="unreachable"
    echo "rank $i $h: $line"
  done
  if [ -f "$REPO/bmuon/runs/$name/DONE" ]; then echo "DONE"; fi
  ;;
stop)
  for h in $1; do
    "${SSH[@]}" "$h" "pkill -u $USER -f '[b]muon[.]train' && echo '$h: stopped' || echo '$h: nothing running'" || echo "$h: unreachable"
  done
  ;;
*)
  sed -n '2,15p' "$0"; exit 1
  ;;
esac
