#!/usr/bin/env bash
# Commit and push the results of this machine to GitHub (as Octave Koenig, noreply address).
#   bash bmuon/experiments/publish_results.sh ["message"]
# Pushes bmuon/results/ and the small files of bmuon/runs/ (see bmuon/.gitignore).
# Safe to run from several machines at once: it rebases and retries.
set -u
cd "$(dirname "$0")/../.."
MSG="${1:-results from $(hostname) $(date -Is)}"
GIT=(git -c user.name="Octave Koenig" -c user.email="OctaveKNG@users.noreply.github.com")
"${GIT[@]}" add -A bmuon/results bmuon/runs 2>/dev/null
if "${GIT[@]}" diff --cached --quiet; then echo "nothing new to publish"; exit 0; fi
"${GIT[@]}" commit -q -m "$MSG" || exit 1
for i in 1 2 3 4 5 6; do
  if "${GIT[@]}" pull -q --rebase origin main && "${GIT[@]}" push -q origin main; then
    echo "published: $MSG"; exit 0
  fi
  sleep $((RANDOM % 30 + 10))
done
echo "push failed after 6 tries; the commit is kept locally, run this script again" >&2
exit 1
