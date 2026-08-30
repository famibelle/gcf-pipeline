#!/usr/bin/env python3
"""Fabrique les deux interfaces d'annotation depuis un seul gabarit.

Même page, deux hébergements : GitHub Pages sert un lot construit à l'avance,
le Space Hugging Face interroge une API qui détient le jeton du corpus gaté.
Structure et CSS viennent de web/annotation.html, la logique de web/app.js ;
seule la ligne de mode et le bouton d'export les séparent.

    scripts/build_pages.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
GABARIT = RACINE / "web" / "annotation.html"
LOGIQUE = RACINE / "web" / "app.js"

CSS = """
/* --- recherche, motifs, pagination --- */
.qbox{display:flex;gap:6px;margin-bottom:8px}
.qbox input{flex:1;min-width:0;padding:7px 9px;border:1px solid var(--line);border-radius:8px;
  background:var(--surface);color:var(--ink);font:inherit;font-size:13px}
.qbox input:focus{outline:2px solid var(--accent);outline-offset:-1px}
.meta{padding:6px 2px 8px;color:var(--ink-3);font-size:12px;display:flex;justify-content:space-between;gap:8px}
.motif{display:inline-block;margin-left:6px;padding:1px 6px;border-radius:999px;
  font-size:10px;font-weight:600;letter-spacing:.02em;vertical-align:1px}
.motif[data-m="français"]{background:var(--warn-soft);color:var(--warn)}
.motif[data-m="hallucination"]{background:var(--mark);color:var(--mark-ink)}
.motif[data-m="vide"]{background:var(--surface-2);color:var(--ink-3)}
.plus{width:100%;margin:8px 0 4px;padding:8px;border:1px dashed var(--line-2);border-radius:8px;
  background:none;color:var(--ink-2);font:inherit;font-size:12px;cursor:pointer}
.plus:hover{background:var(--surface-2)}
.qui{padding:5px 8px;border:1px solid var(--line);border-radius:8px;background:var(--surface);
  color:var(--ink);font:inherit;font-size:12px;width:120px}
.vide{padding:24px 8px;color:var(--ink-3);font-size:13px;text-align:center}
/* ★ vaut jugement sur l'extrait, « ignoré » sur le moment : deux couleurs. */
.tag.rebut{background:var(--warn-soft);color:var(--warn)}
.row[data-state="rebut"] .dot{background:var(--warn);border-radius:2px}
/* --- note de confiance --- */
.etoiles{display:flex;align-items:center;gap:2px;margin:10px 0 2px}
.etoile{background:none;border:0;padding:1px 1px;font-size:20px;line-height:1;cursor:pointer;
  color:var(--ink-3);transition:color .1s}
.etoile[data-on="1"]{color:var(--warn)}
.etoiles:hover .etoile{color:var(--ink-2)}
.etoiles:hover .etoile[data-on="1"]{color:var(--warn)}
.etoile:focus-visible{outline:2px solid var(--accent);outline-offset:2px;border-radius:4px}
.etoile-txt{margin-left:8px;color:var(--ink-3);font-size:12px}
.etoile-txt[data-note="5"]{color:var(--good);font-weight:500}
.etoile-txt[data-note="1"]{color:var(--warn)}
.note-mini{color:var(--warn);font-size:10px;letter-spacing:-1px;margin-left:5px}
.bon{margin-top:8px;color:var(--good);border-color:var(--good)}
.bon:hover{background:var(--good-soft)}
</style>"""

FILTRES = """<div class="qbox">
      <input type="search" id="q" placeholder="Chercher dans les transcriptions…" autocomplete="off">
      <button class="btn" id="hasard" title="Sauter au hasard dans le corpus">🎲</button>
    </div>
    <div class="filters">
      <button class="chip" data-filter="tous" aria-pressed="true">Tous</button>
      <button class="chip" data-filter="français" aria-pressed="false">Français</button>
      <button class="chip" data-filter="hallucination" aria-pressed="false">Hallucinations</button>
      <button class="chip" data-filter="vide" aria-pressed="false">Vides</button>
      <button class="chip" data-etat="todo" aria-pressed="false">À faire</button>
      <button class="chip" data-etat="done" aria-pressed="false">Corrigés</button>
      <button class="chip" data-etat="rebut" aria-pressed="false">Inutilisables</button>
    </div>
    <div class="meta"><span id="trouves">…</span><span id="stockage"></span></div>
    """


# L'ancrage verbal fait tout le travail : une échelle 1-5 sans libellés
# s'effondre vers « 5 ou 1 », et deux annotateurs n'y mettent pas la même chose.
ANCRAGES = [
    (1, "inexploitable — audio inaudible, rien à en tirer"),
    (2, "j'ai deviné, plusieurs passages me résistent"),
    (3, "le sens est bon, l'orthographe gcf reste incertaine"),
    (4, "correct, un doute ponctuel — un mot, un accent"),
    (5, "je réponds de ce texte : gcf validé, il fait foi"),
]
ETOILES = ('      <div class="etoiles" id="etoiles" role="radiogroup" '
           'aria-label="Confiance dans la transcription">\n'
           + "".join(f'        <button type="button" class="etoile" data-note="{n}" '
                     f'aria-label="{n} sur 5 — {t}" title="{n} ★ — {t}">☆</button>\n'
                     for n, t in ANCRAGES)
           + '        <span class="etoile-txt" id="etoile-txt">non notée</span>\n'
             '      </div>\n')


def gabarit_commun() -> str:
    src = GABARIT.read_text(encoding="utf-8")
    src = src.replace("</style>", CSS, 1)
    src = src.replace(
        "<h1>Diagnostic Kreyòl GCF</h1>\n    <div class=\"sub\">"
        "Phase 1 — correction manuelle des segments Whisper-ht</div>",
        "<h1>Annotation Kreyòl GCF</h1>\n    <div class=\"sub\" id=\"sub\">chargement…</div>")
    src = src.replace(
        '<button class="btn" id="theme" title="Basculer le thème">Thème</button>',
        '<input class="qui" id="annotateur" placeholder="ton nom" '
        'title="Sépare ton travail de celui des autres">\n'
        '  <button class="btn" id="theme" title="Basculer le thème">Thème</button>')
    # La colonne de gauche gagne une recherche : on ne choisit plus dans 80
    # segments figés mais dans un lot, voire dans le corpus entier.
    debut = src.index('<div class="filters">')
    fin = src.index('<div id="rows"></div>')
    src = src[:debut] + FILTRES + src[fin:]
    # Plus de « pré-correction » : le corpus n'en a pas, seule la sortie Whisper existe.
    src = src.replace('<button class="btn" id="fill">Partir de la pré-correction</button>\n        ', '')
    src = src.replace('id="copysrc">Copier la source', 'id="copysrc">Reprendre la transcription')
    # Un vote, rien de plus : la transcription affichée est bonne telle quelle.
    # Trois gestes — reprendre, noter, valider — deviennent un.
    src = src.replace(
        '<div class="ref kreyol" id="source"></div>',
        '<div class="ref kreyol" id="source"></div>\n'
        '      <button class="btn bon" id="bon" title="Cette transcription est correcte en gcf">'
        '+ Bonne telle quelle</button>')
    # La note de confiance s'intercale entre la saisie et les actions : on la
    # pose après avoir écrit, avant de valider.
    src = src.replace(
        '      <div class="actions">',
        ETOILES + '      <div class="actions">')
    src = src.replace(
        '<span><kbd>Tab</kbd> accepter la suggestion</span>',
        '<span><kbd>Tab</kbd> accepter la suggestion</span>\n'
        '      <span><kbd>Alt</kbd>+<kbd>1</kbd>…<kbd>5</kbd> noter la transcription</span>\n'
        '      <span><kbd>Alt</kbd>+<kbd>Entrée</kbd> transcription bonne telle quelle</span>')
    return src


def ecrire(cible: Path, mode: str, export: str) -> None:
    src = gabarit_commun().replace(
        '<button class="btn primary" id="export">Exporter le CSV</button>', export)
    cible.parent.mkdir(parents=True, exist_ok=True)
    cible.write_text(
        src + f'<script>window.GCF_MODE = "{mode}";</script>\n'
        '<script src="app.js"></script>\n</body>\n</html>\n', encoding="utf-8")
    shutil.copyfile(LOGIQUE, cible.parent / "app.js")
    print(f"  {cible.relative_to(RACINE)} ({cible.stat().st_size/1024:.0f} Ko) — mode {mode}")


def main() -> int:
    print("interfaces générées :")
    # GitHub Pages : pas de serveur, l'export se fabrique dans le navigateur.
    ecrire(RACINE / "docs" / "index.html", "statique",
           '<button class="btn primary" id="export">Exporter le CSV</button>')
    # Space : l'export vient du serveur, qui a toutes les versions.
    ecrire(RACINE / "space" / "static" / "index.html", "serveur",
           '<a class="btn primary" id="export" href="api/export.csv" download>Exporter le CSV</a>')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
