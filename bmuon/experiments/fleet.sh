#!/usr/bin/env bash
# Les 5 runs du week-end d'oct. 2026 (16 machines chacun = 80 machines).
# À lancer depuis ablette (clé SSH inter-machines id_cluster).
#
#   bash bmuon/experiments/fleet.sh hosts            liste des machines de chaque run
#   bash bmuon/experiments/fleet.sh provision        copie env+données (réseau local) sur les 32 nouvelles machines
#   bash bmuon/experiments/fleet.sh verify           env/données/GPU de chaque machine, et aucun de nos process
#   bash bmuon/experiments/fleet.sh start [RUN ...]  démarre les runs (tous par défaut) ; REFUSE hors fenêtre DSI
#   bash bmuon/experiments/fleet.sh status           état de chaque run
#   bash bmuon/experiments/fleet.sh count            nombre de nos process par run (doit être 0 après 7h30)
#   bash bmuon/experiments/fleet.sh stop             tue tout, puis VÉRIFIE qu'il ne reste rien (code 1 sinon)
#   bash bmuon/experiments/fleet.sh repair           (en fenêtre seulement) relance les runs anormaux depuis leur state.pkl ;
#                                                    ne supprime/écrase JAMAIS un state ; DRY=1 pour simuler
#
# Arrêt automatique : launch_dist.sh enveloppe chaque process dans `timeout` jusqu'à 7h10
# (voir window.sh). `stop` et le cron ne sont que des filets de sécurité.
set -u
USER="${USER:-$(id -un)}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
LD="$REPO/bmuon/experiments/launch_dist.sh"
source "$REPO/bmuon/experiments/window.sh"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new)
EXTRA="save_every_steps=1000 snapshot_every_epochs=5 batch_size=16 grad_accum_steps=1 max_n_nodes=1024 max_n_edges=52000"

# --- machines : le premier hôte de chaque groupe est le process 0 (coordinateur) ---------------
H_BMUON_WD1E2="ain allier ardennes carmor charente cher creuse dordogne doubs essonne finistere gironde indre jura landes loire"
H_BMUON_WD1E3="manche marne mayenne morbihan moselle saone somme vendee vosges phalange radius rotule sacrum sternum tarse temporal"
H_SELWD="bentley bugatti cadillac chrysler corvette ferrari fiat ford jaguar lada maserati mazda nissan niva peugeot pontiac"
H_UNI_WD1E2="albatros autruche bengali coucou epervier faisan gelinotte harpie hibou jabiru kamiche linotte loriol mouette nandou ombrette"
H_UNI_WD1E3="acromion apophyse astragale atlas axis coccyx cote cubitus cuboide femur frontal humerus malleole metacarpe parietal perone"
# anciennes machines poissons (3090) : plus utilisées, mais incluses dans `stop` par sécurité
H_FISH="ablette anchois anguille barbeau brochet carrelet gardon gymnote labre lieu mulet murene piranha raie requin rouget"
# machines de rechange (Ada libres, non installées)
H_SPARE="tibia xiphoide porsche renault rover royce simca skoda venturi volvo"

RUNS=(mp1-bmuon-wd1e2 mp1-bmuon-wd1e3 mp1-bmuon-selwd mp1-uniform-shrink-wd1e2 mp1-uniform-shrink-wd1e3)
run_cfg()   { case "$1" in
  mp1-bmuon-wd1e2)          echo "bmuon/configs/nequix-mp-1-bmuon.yml";;
  mp1-bmuon-wd1e3)          echo "bmuon/configs/nequix-mp-1-bmuon.yml";;
  mp1-bmuon-selwd)          echo "bmuon/configs/nequix-mp-1-bmuon-selwd.yml";;
  mp1-uniform-shrink-wd1e2) echo "bmuon/configs/nequix-mp-1-uniform.yml";;
  mp1-uniform-shrink-wd1e3) echo "bmuon/configs/nequix-mp-1-uniform-wd1e3.yml";; esac; }
run_wd()    { case "$1" in
  mp1-bmuon-wd1e2|mp1-uniform-shrink-wd1e2) echo "0.01";;
  *) echo "0.001";; esac; }
run_hosts() { case "$1" in
  mp1-bmuon-wd1e2)          echo "$H_BMUON_WD1E2";;
  mp1-bmuon-wd1e3)          echo "$H_BMUON_WD1E3";;
  mp1-bmuon-selwd)          echo "$H_SELWD";;
  mp1-uniform-shrink-wd1e2) echo "$H_UNI_WD1E2";;
  mp1-uniform-shrink-wd1e3) echo "$H_UNI_WD1E3";; esac; }
ALL_HOSTS="$H_BMUON_WD1E2 $H_BMUON_WD1E3 $H_SELWD $H_UNI_WD1E2 $H_UNI_WD1E3"
STOP_HOSTS="$ALL_HOSTS $H_FISH $H_SPARE"
NEW_HOSTS="$H_BMUON_WD1E2 $H_BMUON_WD1E3"

# nombre de nos process d'entraînement sur $1 (le motif [b]muon[.]train ne se reconnaît pas lui-même)
nproc_on() {  # "?" si la machine est injoignable ; pgrep -c renvoie 1 quand il compte 0, d'où le `; true`
  local o; o=$("${SSH[@]}" "$1" "pgrep -u \$USER -fc 'python.*[b]muon[.]train'; true" 2>/dev/null) || o="?"
  echo "${o:-?}"
}

active_count() {  # nombre de machines de la liste $1 qui ont au moins un de nos process
  local n=0 c h; for h in $1; do c=$(nproc_on "$h"); [[ "$c" =~ ^[0-9]+$ ]] && [ "$c" -gt 0 ] && n=$((n+1)); done; echo "$n"
}
log_age() {  # secondes depuis la dernière écriture du log du process 0 du run $1 (hôtes $2) ; 999999 si absent
  local coord=${2%% *}
  "${SSH[@]}" "$coord" "f=/tmp/\$USER/dist-$1-rank0.log; if [ -f \$f ]; then echo \$(( \$(date +%s) - \$(stat -c %Y \$f) )); else echo 999999; fi" 2>/dev/null || echo 999999
}

cmd="${1:-}"; shift || true
case "$cmd" in
hosts)
  for r in "${RUNS[@]}"; do echo "$r (wd=$(run_wd "$r")): $(run_hosts "$r")"; done ;;

provision)
  # Copie depuis 4 machines déjà prêtes (réseau local) plutôt que 32 téléchargements internet.
  SRC=(bentley bugatti cadillac chrysler)
  tgt=($NEW_HOSTS); [ $# -gt 0 ] && tgt=("$@")
  prov_one() {  # $1 = source, $2 = cible ; 3 tentatives
    for try in 1 2 3; do prov_try "$1" "$2" && return 0; sleep 15; done
    echo "  $2: ECHEC depuis $1 (3 tentatives)"
  }
  prov_try() {
    "${SSH[@]}" "$1" "set -e; ssh -o BatchMode=yes $2 'mkdir -p /tmp/\$USER'; cd /tmp/\$USER;
      rsync -a --exclude=.bmuon_ready nequix-venv uv-python cuda12 $2:/tmp/\$USER/;
      rsync -a --exclude=.complete bmuon-data $2:/tmp/\$USER/;
      rsync -a nequix-venv/.bmuon_ready $2:/tmp/\$USER/nequix-venv/.bmuon_ready;
      rsync -a bmuon-data/mptrj-aselmdb/.complete $2:/tmp/\$USER/bmuon-data/mptrj-aselmdb/.complete" \
      && echo "  $2: copié depuis $1"
  }
  for s in 0 1 2 3; do
    ( for ((i=s; i<${#tgt[@]}; i+=4)); do prov_one "${SRC[$s]}" "${tgt[$i]}"; done ) &
  done
  wait; echo "provision terminée (lancer ensuite : verify)" ;;

verify)
  bad=0
  # une seule ligne par machine, écrite dans un fichier à part (pas d'entrelacement entre machines)
  chk='e=$( [ -f /tmp/$USER/nequix-venv/.bmuon_ready ] && echo ok || echo NO ); d=$( [ -e /tmp/$USER/bmuon-data/mptrj-aselmdb/.complete ] && echo ok || echo NO ); o=$(pgrep -u $USER -fc "python.*[b]muon[.]train"); g=$(nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -1); echo "$(hostname -s)|env=$e|data=$d|ours=$o|$g"; true'
  tmp=$(mktemp -d)
  for h in $ALL_HOSTS; do ( r=$("${SSH[@]}" "$h" "$chk" 2>/dev/null) || r="$h|UNREACHABLE"; echo "$r" > "$tmp/$h" ) & done; wait
  out=$(cat "$tmp"/* | sort); rm -rf "$tmp"
  echo "$out"
  echo "$out" | grep -E "UNREACHABLE|env=NO|data=NO" >/dev/null && { echo ">>> PROBLÈMES ci-dessus"; bad=1; }
  echo "$out" | grep -E "ours=[1-9]" >/dev/null && { echo ">>> des process tournent déjà"; bad=1; }
  exit $bad ;;

start)
  sel=("$@"); [ ${#sel[@]} -eq 0 ] && sel=("${RUNS[@]}")
  for r in "${sel[@]}"; do
    hosts="$(run_hosts "$r")"; [ -n "$hosts" ] || { echo "run inconnu : $r"; continue; }
    busy=0; for h in $hosts; do c=$(nproc_on "$h"); [ "$c" != "0" ] && { echo "  $r : $h a déjà $c process -> run NON lancé"; busy=1; }; done
    [ $busy -eq 1 ] && continue
    echo "=== $r (wd=$(run_wd "$r"))"
    bash "$LD" start "$r" "$(run_cfg "$r")" "$hosts" --set weight_decay="$(run_wd "$r")" $EXTRA || echo "  >>> $r : lancement refusé/échoué"
  done ;;

status)
  for r in "${RUNS[@]}"; do
    hosts=$(run_hosts "$r"); n=0
    for h in $hosts; do c=$(nproc_on "$h"); [ "$c" != "0" ] && [ "$c" != "?" ] && n=$((n+1)); done
    coord=${hosts%% *}
    last=$("${SSH[@]}" "$coord" "grep -a \"'epoch'\" /tmp/\$USER/dist-$r-rank0.log 2>/dev/null | tail -n 1 | cut -c1-110" 2>/dev/null)
    echo "$r : $n/16 machines actives | $last"
  done ;;

count)
  tot=0
  for r in "${RUNS[@]}"; do
    n=0; for h in $(run_hosts "$r"); do c=$(nproc_on "$h"); [[ "$c" =~ ^[0-9]+$ ]] && n=$((n+c)); done
    echo "$r : $n process"; tot=$((tot+n))
  done; echo "TOTAL : $tot" ;;

stop)
  for h in $STOP_HOSTS; do ( "${SSH[@]}" "$h" "pkill -9 -u \$USER -f 'python.*[b]muon[.]train'; true" 2>/dev/null ) & done; wait
  sleep 4
  left=0
  for h in $STOP_HOSTS; do ( c=$(nproc_on "$h"); [ "$c" != "0" ] && echo "  $h : $c process restants (ou injoignable: $c)" ) & done | tee /tmp/$USER/stop_left.txt; wait
  if [ -s /tmp/$USER/stop_left.txt ]; then echo "ATTENTION : des machines ne sont pas propres (voir ci-dessus)"; exit 1; fi
  echo "OK : plus aucun process d'entraînement sur $(echo $STOP_HOSTS | wc -w) machines" ;;

repair)
  # Filets : jamais hors fenêtre ; jamais de suppression/écrasement de state ; pas de boucle de relances.
  if ! window_secs_until_stop >/dev/null; then echo "hors fenêtre (ou <30 min avant l'arrêt de 7h10) : aucune action"; exit 0; fi
  for r in "${RUNS[@]}"; do
    hosts="$(run_hosts "$r")"; dir="$REPO/bmuon/runs/$r"
    [ -f "$dir/DONE" ] && { echo "$r : terminé (DONE)"; continue; }
    n=$(active_count "$hosts"); age=$(log_age "$r" "$hosts")
    if [ "$n" -eq 16 ] && [ "$age" -lt 1200 ]; then echo "$r : OK ($n/16 machines, log il y a ${age}s)"; continue; fi
    cf="$HOME/.fleet_repairs_$r"; now=$(date +%s)
    recent=$(awk -v t=$((now-10800)) '$1>t' "$cf" 2>/dev/null | wc -l)
    if [ "$recent" -ge 3 ]; then echo "$r : PROBLEME PERSISTANT ($n/16, log il y a ${age}s) - 3 réparations en 3 h, je n'insiste pas : à regarder"; continue; fi
    echo "$r : anormal ($n/16 machines, log il y a ${age}s) -> réparation"
    if [ "${DRY:-0}" = "1" ]; then echo "  [dry] arrêt des process du run puis start (state.pkl* intacts)"; continue; fi
    echo "$now" >> "$cf"
    for h in $hosts; do ( "${SSH[@]}" "$h" "pkill -9 -u \$USER -f 'python.*[b]muon[.]train'; true" 2>/dev/null ) & done; wait; sleep 6
    echo "  states conservés :"; ls -l --time-style=+%H:%M "$dir"/state.pkl* 2>/dev/null | awk '{print "   ",$5,$6,$7}'
    bash "$0" start "$r"
  done ;;

*) sed -n '2,15p' "$0"; exit 1 ;;
esac
