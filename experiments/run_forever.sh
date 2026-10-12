#!/usr/bin/env bash
# Run one training job until it finishes, resuming after crashes or reboots.
#   experiments/run_forever.sh <out_dir> <config.yml> [overrides...]
# Env:
#   BMUON_DATA_SRC  shared copy of the preprocessed data (e.g. on NFS)
#   BMUON_DATA      local copy used by the configs (default /tmp/$USER/bmuon-data)
#   BMUON_PYTHON    python interpreter (default: python)
set -u
OUT="$1"; CONFIG="$2"; shift 2
export BMUON_DATA="${BMUON_DATA:-/tmp/$USER/bmuon-data}"
PY="${BMUON_PYTHON:-python}"
mkdir -p "$OUT"
echo "$(hostname) $(date -Is) start" >> "$OUT/hosts.log"

if [ -n "${BMUON_DATA_SRC:-}" ]; then
  mkdir -p "$BMUON_DATA"
  # copy once per machine; a lock avoids two jobs copying at the same time
  ( flock 9; rsync -a "$BMUON_DATA_SRC"/ "$BMUON_DATA"/ ) 9>"/tmp/$USER-bmuon-data.lock"
fi

tries=0
while [ ! -f "$OUT/DONE" ]; do
  tries=$((tries + 1))
  "$PY" -m bmuon.train "$CONFIG" --set "out_dir=$OUT" "$@" >> "$OUT/stdout.log" 2>&1
  code=$?
  [ -f "$OUT/DONE" ] && break
  echo "$(hostname) $(date -Is) exit $code (try $tries), resuming in 60 s" >> "$OUT/hosts.log"
  if [ "$tries" -ge 20 ]; then echo "giving up" >> "$OUT/hosts.log"; exit 1; fi
  sleep 60
done
echo "$(hostname) $(date -Is) done" >> "$OUT/hosts.log"
