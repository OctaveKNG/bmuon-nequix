#!/usr/bin/env bash
# Run one training job until it finishes, resuming after crashes or reboots.
#   bmuon/experiments/run_forever.sh <out_dir> <config.yml> [overrides...]
# Env:
#   BMUON_DATA_SRC  where to copy the preprocessed data from: a path, or host:/path
#                   (e.g. ablette:/tmp/octave.koenig/bmuon-data)
#   BMUON_DATA      local copy used by the configs (default /tmp/$USER/bmuon-data)
#   BMUON_PYTHON    python interpreter (default: the venv made by setup_machine.sh)
# The python environment lives on the local disk (/tmp/$USER/nequix-venv); it is installed
# with setup_machine.sh on first use, and again if /tmp was wiped.
set -u
USER="${USER:-$(id -un)}"
OUT="$1"; CONFIG="$2"; shift 2
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/$USER/uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/tmp/$USER/uv-python}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/tmp/$USER/nequix-venv}"
export BMUON_DATA="${BMUON_DATA:-/tmp/$USER/bmuon-data}"
PY="${BMUON_PYTHON:-$UV_PROJECT_ENVIRONMENT/bin/python}"
mkdir -p "$OUT" "/tmp/$USER"
echo "$(hostname) $(date -Is) start" >> "$OUT/hosts.log"

if [ ! -f "$UV_PROJECT_ENVIRONMENT/.bmuon_ready" ]; then
  ( flock 8; [ -f "$UV_PROJECT_ENVIRONMENT/.bmuon_ready" ] || \
      bash "$(dirname "$0")/setup_machine.sh" >> "$OUT/setup.log" 2>&1 ) 8>"/tmp/$USER-bmuon-setup.lock"
  if [ ! -f "$UV_PROJECT_ENVIRONMENT/.bmuon_ready" ]; then
    echo "$(hostname) $(date -Is) setup failed, see $OUT/setup.log" >> "$OUT/hosts.log"; exit 1
  fi
fi

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
