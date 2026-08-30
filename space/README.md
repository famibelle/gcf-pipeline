---
title: Annotation Kreyòl GCF
emoji: 🎧
colorFrom: teal
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Correction humaine des transcriptions Whisper du corpus GCF
---

# Annotation Kreyòl GCF

Interface de correction des transcriptions Whisper du corpus
`POTOMITAN/potomitan-gcf-transcription`, avec prédiction kréyòl à la saisie.

## Pourquoi un Space plutôt qu'une page statique

Le corpus est sous accès contrôlé : un visiteur non connecté reçoit un `401` et
n'entend rien. Ici, le serveur détient le jeton et relaie l'audio pour le
compte de l'annotateur — **rien n'est copié, rien n'est dégaté**, et les
40 000 segments restent piochables un par un. Le jeton ne descend jamais dans
le navigateur.

## Secrets à régler

| Variable | Rôle |
|---|---|
| `HF_TOKEN` | lecture du corpus, écriture des corrections. **Obligatoire.** |
| `CORPUS_DATASET` | corpus source (défaut `POTOMITAN/potomitan-gcf-transcription`) |
| `ANNOT_DATASET` | où sont rangées les corrections (défaut `POTOMITAN/gcf-annotations`) |

Un jeton en lecture seule suffit à écouter ; l'écriture des corrections
demande un jeton *write* sur le dépôt d'annotations. Sans lui, le travail
reste en mémoire et l'interface le signale — pense alors à exporter le CSV.

## Ce que fait le serveur

- `GET /api/segments` — piocher : recherche plein texte, motif de rejet, avancement.
- `GET /api/audio/{chemin}` — relais authentifié, `Range` compris (donc barre de progression utilisable).
- `POST /api/corrections` — enregistre, un fichier JSONL par annotateur.
- `GET /api/export.csv` — une ligne par (segment, annotateur).

## Savoir si une correction vaut quelque chose

Les corrections sont rangées par **(segment, annotateur)**, jamais par segment
seul : deux personnes peuvent traiter le même extrait sans s'effacer, ce qui est
la condition pour mesurer un jour leur accord. Chacune ne voit que son propre
avancement, et le nombre de versions des autres — jamais leur texte, qui
biaiserait la comparaison.

Chaque correction emporte `ecoute_ms` (audio réellement entendu, pas le temps
passé devant l'écran) et `lectures`, à rapprocher de `duree_ms` : une correction
écrite sans avoir écouté se repère seule.

Avec deux ou trois annotateurs, personne ne repasse derrière personne et
l'accord inter-annotateurs est hors de portée. Deux mécanismes le remplacent,
tous deux invisibles pour l'annotateur :

- **Les témoins.** Des extraits dont la bonne transcription est connue, glissés
  dans le flux en navigation libre (jamais sous un filtre par motif, où ils
  dépareilleraient). `scripts/build_temoins.py` les constitue depuis un CSV
  validé ; le fichier `data/temoins.jsonl` n'est pas versionné.
- **Le re-test.** Un extrait déjà corrigé, resservi **vierge** après
  `DELAI_RETEST` (14 jours par défaut) à un taux de `TAUX_RETEST` pour mille.
  L'écart entre les deux versions mesure l'accord de l'annotateur avec
  lui-même. Le tirage est déterministe et la reprise n'a lieu qu'une fois.

Les corrections successives d'une même séance se remplacent ; un retour plus
tard que `DELAI_VERSION` ajoute une version. C'est cet historique que lit
`scripts/qualite_annotations.py`, qui sort l'écart aux témoins, l'accord avec
soi-même et les segments à revoir.

Trois issues, et non deux : **corrigé**, **ignoré** (je passe, à reprendre) et
**inutilisable** (l'extrait ne vaut rien — inaudible, vide, hors sujet).
Confondre les deux dernières pousserait à inventer une transcription plutôt que
de laisser un blanc, et un faux corpus est pire qu'un corpus plus petit.

Le disque d'un Space est éphémère : les corrections vivent en mémoire et sont
poussées vers le dataset d'annotations toutes les 20 secondes.

## Source

Développé dans [`famibelle/gcf-pipeline`](https://github.com/famibelle/gcf-pipeline),
répertoire `space/`. Déploiement : `scripts/deploy_space.sh`.
