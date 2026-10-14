# Registre des runs — BMuon × Nequix

> **Convention de nommage :** `mp1` = Nequix-MP-1 (le modèle de référence de la section 5 du papier).
> Ajouter une entrée dès qu'un run démarre ; mettre à jour `status` et `résultat` quand il se termine.

---

## Runs actifs (oct 2026)

### `mp1-bmuon-wd1e2` — BMuon référence, WD uniforme 0.01
- **Config :** `bmuon/configs/nequix-mp-1-bmuon.yml`
- **Modèle :** Nequix-MP-1 plein — `128x0e + 64x1o + 32x2e + 32x3o`
- **Optimizer :** BMuon · budget=0.3 · anneal_fraction=0.25 · cost_model="paths"
- **weight_decay :** 0.01 uniforme sur tous les canaux budgétés
- **Machines :** 16 × RTX 3090 24 Go — poissons (ablette, anchois, …, rouget) · salle 34
- **Coordinateur (rank 0) :** ablette · logs NFS : `bmuon/runs/mp1-bmuon-wd1e2/`
- **Lancé :** nuit 13 oct 2026
- **But :** run principal de la thèse — mesurer l'allocation apprise par BMuon et comparer
  la performance au modèle shrink uniforme de même coût total (B=0.3 paths).
- **Ce qui le distingue :** seul run avec BMuon *et* WD=0.01. Hypothèse à tester : ce WD élevé
  crée-t-il un "plafond de verre" qui empêche les canaux actifs d'atteindre leur plein potentiel ?
- **Statut :** en cours — 10 époques terminées au 14/10/2026

---

### `mp1-uniform-shrink-wd1e2` — Baseline uniforme, WD 0.01
- **Config :** `bmuon/configs/nequix-mp-1-uniform.yml`
- **Modèle :** shrink uniforme à 30 % du coût paths — `40x0e + 16x1o + 8x2e + 8x3o`
  (même coût total que BMuon budget=0.3 → comparaison équitable)
- **Optimizer :** Muon standard (pas de budget, pas d'allocation)
- **weight_decay :** 0.01 — identique à mp1-bmuon-wd1e2
- **Machines :** 16 × oiseaux (albatros, autruche, …, ombrette)
- **Coordinateur :** albatros
- **Lancé :** nuit 13 oct 2026
- **But :** baseline principale — à iso-coût de calcul, est-ce que BMuon fait mieux qu'une
  réduction uniforme naïve ? Même WD que le run BMuon pour isoler l'effet du budget.
- **Statut :** en cours

---

### `mp1-uniform-shrink-wd1e3` — Baseline uniforme, WD 0.001 (défaut Nequix)
- **Config :** `bmuon/configs/nequix-mp-1-uniform-wd1e3.yml`
- **Modèle :** `40x0e + 16x1o + 8x2e + 8x3o` — identique à uniform-wd1e2
- **Optimizer :** Muon standard
- **weight_decay :** 0.001 — valeur par défaut de Nequix (×10 moins fort que wd1e2)
- **Machines :** 16 × anatomy (acromion, apophyse, …, perone) · RTX 4000 Ada 20 Go
- **Coordinateur :** acromion
- **Lancé :** nuit 14 oct 2026 (relancé après crash de 3 machines mal provisionnées)
- **But :** vérifier si WD=0.01 pénalise les runs wd1e2 (uniform et BMuon) par rapport au
  défaut Nequix. Si uniform-wd1e3 > uniform-wd1e2, alors WD=0.01 est bien un handicap.
- **Ce qui le distingue :** seul WD à differer entre wd1e2 et wd1e3 — tout le reste est identique.
- **Statut :** en cours — époque 1 en cours au 14/10/2026

---

### `mp1-bmuon-selwd` — BMuon avec weight decay sélectif
- **Config :** `bmuon/configs/nequix-mp-1-bmuon-selwd.yml`
- **Modèle :** Nequix-MP-1 plein — `128x0e + 64x1o + 32x2e + 32x3o`
- **Optimizer :** BMuon · budget=0.3 · anneal_fraction=0.25
- **weight_decay :** 0.001 (wd_base) + WD sélectif : λ(μ_c) = 0.001 + 0.049 · max(0, 1−μ_c/μ̄)²
  → canaux actifs (μ_c ≈ 1) : WD ≈ 0.001 ; canaux mourants (μ_c → 0) : WD → 0.050
- **Machines :** 16 × voitures (bentley, bugatti, …, pontiac) · RTX 4000 Ada 20 Go
- **Coordinateur :** bentley
- **Lancé :** nuit 14 oct 2026
- **But :** supprimer le "plafond de verre" hypothétique de wd1e2 — les canaux actifs ne sont
  plus pénalisés au-delà du défaut Nequix, mais les mourants sont poussés plus fort.
  Implémenté dans `bmuon/optimizer.py` (`scale_by_bmuon`, paramètre `selective_wd`).
- **Ce qui le distingue :** WD n'est pas uniforme — dépend de l'allocation μ_c courante.
  Nouveau mécanisme, pas encore dans la littérature. À comparer à mp1-bmuon-wd1e2.
- **Statut :** en cours — époque 1 en cours au 14/10/2026

---

## Runs terminés / archivés

### `essai`, `essai1m`, `essai2m`, `essai16m`, `essai64` — Essais techniques
- Tests de fonctionnement du launcher distribué et de la scalabilité (1, 2, 16 machines).
- `essai2m` : validation ~36 s/époque sur 2 × RTX 3090 (conforme aux repères CLAUDE.md).
- Pas de résultats scientifiques — infrastructure uniquement.

---

## Matrice de comparaison

| Run | Modèle | Optimizer | WD | WD type | GPU |
|-----|--------|-----------|-----|---------|-----|
| mp1-bmuon-wd1e2 | plein (B=1.0) | BMuon B=0.3 | 0.01 | uniforme | 3090 |
| mp1-uniform-shrink-wd1e2 | shrunk (B≈0.3) | Muon | 0.01 | uniforme | bird |
| mp1-uniform-shrink-wd1e3 | shrunk (B≈0.3) | Muon | 0.001 | uniforme | 4000 Ada |
| mp1-bmuon-selwd | plein (B=1.0) | BMuon B=0.3 | 0.001+0.049 | **sélectif** | 4000 Ada |

> **Comparaisons clés :**
> - BMuon vs baseline iso-coût : `mp1-bmuon-wd1e2` vs `mp1-uniform-shrink-wd1e2`
> - Effet du WD sur baseline : `mp1-uniform-shrink-wd1e2` vs `mp1-uniform-shrink-wd1e3`
> - WD sélectif vs uniforme : `mp1-bmuon-selwd` vs `mp1-bmuon-wd1e2`
>   (NB : GPU différents — normaliser par temps/époque ou par coût FLOPs)
