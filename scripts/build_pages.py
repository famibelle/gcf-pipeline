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
/* « Inutilisable » juge l'extrait, « ignoré » juge le moment : deux couleurs. */
.tag.rebut{background:var(--warn-soft);color:var(--warn)}
.row[data-state="rebut"] .dot{background:var(--warn);border-radius:2px}
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
    # Distinguer « je passe » de « cet extrait ne vaut rien » : sans cette
    # sortie, on écrit n'importe quoi plutôt que de laisser un blanc.
    src = src.replace(
        '<button class="btn" id="skip">Ignorer ce segment</button>',
        '<button class="btn" id="skip">Ignorer ce segment</button>\n'
        '        <button class="btn" id="rebut" title="Inaudible, vide ou hors sujet">Inutilisable</button>')
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
