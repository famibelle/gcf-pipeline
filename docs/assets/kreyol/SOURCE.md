# Moteur de suggestion kréyòl — copie figée

Ces fichiers viennent de **[KreyolKeyb](https://github.com/famibelle/KreyolKeyb)**,
`docs/assets/`. Ils sont copiés ici plutôt que chargés depuis
`famibelle.github.io/KreyolKeyb/` pour que l'outil d'annotation ne dépende pas
d'un autre dépôt en cours d'évolution : une session de correction ne doit pas
casser parce qu'un fichier a été renommé ailleurs.

| fichier | rôle |
|---|---|
| `simulateur-engine.js` | moteur, port JS de `SuggestionEngine.kt` |
| `creole_dict.json` | 5 284 mots kréyòl avec fréquence |
| `creole_ngrams.json` | bigrammes : 5 continuations par mot |
| `french_simple_dict.json` | appoint français du mode bilingue |

Pour resynchroniser après une amélioration du clavier :

    scripts/sync_kreyol_engine.sh /chemin/vers/KreyolKeyb
