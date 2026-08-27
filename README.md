# Pipeline Whisper-ht → créole guadeloupéen (gcf)

Transformation des transcriptions Whisper-haïtien de `POTOMITAN/potomitan-gcf-transcription`
en corpus audio-texte aligné gcf, en prenant `POTOMITAN/PawolKreyol-gfc` comme cible
linguistique (norme GEREC-2).

## Installation

```bash
pip install -r requirements.txt
export HF_TOKEN=...        # si les dépôts sont privés
```

KenLM, simalign et faster-whisper sont optionnels : le code bascule sur un repli
fonctionnel quand ils sont absents (voir `src/lm.py`).

## Exécution

```bash
# Phase 1 — extraction des segments à corriger
python -m src.phase1_diagnostic sample --n 300
#   >>> correction humaine de data/diagnostic_to_correct.csv <<<
#   >>> enregistrer sous data/diagnostic_corrected.csv       <<<
python -m src.phase1_diagnostic mine --min-support 5 --min-confidence 0.6
python -m src.phase1_diagnostic audit

# Phase 2 — paires synthétiques
python -m src.phase2_synthetic --n 80000

# Modèle de langue gcf (requis par la Phase 4)
python -m src.build_lm --order 5

# Phase 3 — correcteur
python -m src.phase3_train --epochs 4 --batch-size 8

# Phase 4 — filtrage et livrable
python -m src.phase4_filter_decode --calibrate --audit-sample 200

# Rapport
python -m src.report

# Phase 5 (optionnelle)
python -m src.phase5_pseudolabel transcribe --audio-dir ./audio --nbest 10
python -m src.phase5_pseudolabel merge --agreement 0.85
python -m src.phase5_pseudolabel retrain --epochs 2
```

Ou d'un bloc : `python run_pipeline.py --from 2 --to 4`
(le script s'arrête volontairement en Phase 1 si l'annotation manuelle manque).

Les mêmes phases sont disponibles en notebooks dans `notebooks/`.

## Arborescence

```
src/config.py               paramètres, chemins, seuils
src/normalize.py            nettoyage + garde-fou alphabet GEREC-2
src/rules.py                alignement, décomposition, fouille et application des règles
src/metrics.py              WER / CER sans dépendance
src/lm.py                   KenLM + repli n-grammes caractère
src/build_lm.py             construction du LM gcf
src/phase1_diagnostic.py    sample / mine / audit
src/phase2_synthetic.py     génération des paires pseudo-ht → gcf
src/phase3_train.py         fine-tuning ByT5, évaluation, chargement
src/phase4_filter_decode.py scoring, filtrage, correction, livrables
src/phase5_pseudolabel.py   n-best, filtre d'accord, ré-entraînement
src/report.py               rapport.md
data/seed_rules_ht2gcf.json règles amorces (hypothèses, à valider)
```

## Tests

```bash
pip install -e ".[dev]"
pytest -q          # 35 tests, ~0.3 s
ruff check src tests
```

La suite ne couvre que la logique pure : alignement, fouille de règles, métriques,
LM de repli, découpage des splits. Ni torch ni transformers ne sont installés en
CI. C'est délibéré : une régression dans l'alignement ne lève aucune exception,
elle produit silencieusement des règles fausses — c'est exactement là qu'un test
rapporte quelque chose, pas sur une boucle d'entraînement qui exige un GPU.

## Livrables produits

| Fichier | Phase |
|---|---|
| `artifacts/substitution_rules_ht2gcf.json` | 1 |
| `artifacts/residual_cases.json` | 1 |
| `data/{train,val,test}_pairs.csv` | 2 |
| `artifacts/gcf_corrector_bytesm.pt` + `artifacts/gcf_corrector/` | 3 |
| `artifacts/dataset_corrected.jsonl` | 4 |
| `artifacts/dataset_high_confidence.jsonl` | 4 |
| `artifacts/rapport.md` | — |

## Données et licence

Le code est sous MIT. **Les corpus ne le sont pas** : `potomitan-gcf-transcription`
et `PawolKreyol-gfc` ont leurs propres conditions, et le corpus produit en Phase 4
en est un dérivé. Vérifiez leur licence avant de publier `dataset_high_confidence.jsonl`
ou un modèle entraîné dessus. Le `.gitignore` exclut par défaut `data/` et
`artifacts/` : rien de tout cela ne part sur GitHub sans un geste explicite.

## Cinq points où j'ai dévié du plan initial, et pourquoi

**1. La perte n'est pas une MAE.** Le plan proposait « MAE ou label smoothing ».
La MAE n'est pas définie pour une sortie discrète de génération séquentielle.
L'entraînement utilise l'entropie croisée avec label smoothing, qui répond à
l'intention derrière la demande : ne pas laisser le modèle devenir sur-confiant
sur des cibles synthétiques imparfaites.

**2. Le WER du livrable est estimé, jamais mesuré.** Il n'existe aucune vérité
terrain sur le corpus Whisper — c'est précisément ce qu'on cherche à construire.
Le seuil « <5% WER » ne peut donc pas être vérifié directement. Les champs
s'appellent `wer_estime`, la calibration vient de `test_pairs`, et
`--audit-sample` produit une feuille de ré-annotation humaine qui est le seul
moyen de confirmer le chiffre. Sans cet audit, le WER annoncé reste une
extrapolation depuis des corruptions synthétiques.

**3. Le seuil de perplexité (100) est calibré, pas codé en dur.** La perplexité
n'a pas la même échelle selon l'ordre du n-gramme et le tokenizer ; `100` ne
signifie rien en soi. `--calibrate` fixe le seuil au quantile 0.95 de la
perplexité mesurée sur du gcf propre.

**4. La corruption inverse est stochastique.** Appliquer les règles de façon
déterministe apprendrait au correcteur une bijection parfaite, et il s'effondrerait
sur le premier segment réel qui échappe aux règles. D'où `--corruption-prob 0.85`,
un bruit typographique additionnel, et une part de paires identité pour que le
modèle apprenne aussi à ne rien changer.

**5. Le split se fait par empreinte du texte cible.** PawolKreyol contient des
phrases proches ; un split aléatoire ferait fuiter du train vers le test et
gonflerait les scores de plusieurs points.

## Limites qu'il faut connaître avant de faire tourner ça

- **Les règles amorces de `data/seed_rules_ht2gcf.json` sont des hypothèses.**
  Certaines sont solides (`ap→ka`, `te→té`), d'autres douteuses (`kounye a→atchouman`).
  Elles servent à pré-remplir la feuille de correction, pas à être crues. La
  Phase 1 les confirme, les corrige ou les marque `seed_unconfirmed`.

- **`mwen→an` et `li→i` dépendent de la position syntaxique.** En gcf, `an` est
  sujet mais l'objet et le possessif restent `mwen`. Une règle sans contexte
  surgénère : le pipeline le montre lui-même sur `fanmi an mwen`, qui redevient
  `fanmi an an` après un aller-retour. Les règles contextuelles
  (`left_context` / `right_context` dans `Rule`) existent mais doivent être
  renseignées à partir des annotations ; c'est le principal travail restant.

- **300 segments annotés, c'est peu.** Les substitutions rares n'atteindront pas
  le seuil de support et resteront des erreurs systématiques du correcteur.
  Doubler l'annotation est le levier le plus rentable du projet, avant tout
  réglage d'hyperparamètre.

- **La qualité plafonne à celle des transcriptions Whisper-ht.** Un correcteur
  orthographique ne récupère pas un mot que le modèle acoustique n'a pas entendu.
  Sur les segments où Whisper hallucine, le correcteur produira du gcf bien formé
  et faux — d'où le filtrage en amont plutôt qu'en aval.

- **La Phase 5 peut dégrader.** Chaque tour de pseudo-labeling rend le correcteur
  plus sûr de ce qu'il produit déjà. Le filtre d'accord et l'ancrage sur les paires
  synthétiques limitent la dérive ; au-delà de deux tours, il faut ré-annoter.
