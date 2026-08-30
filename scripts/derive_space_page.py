"""Fabrique space/static/index.html à partir de docs/index.html."""
from pathlib import Path

racine = Path("/home/medhi/SourceCode/gcf-pipeline")
src = (racine / "docs/index.html").read_text(encoding="utf-8")

marque = '<script src="assets/kreyol/simulateur-engine.js"></script>'
tete = src.split(marque)[0]

# --- CSS additionnel -------------------------------------------------------
css = """
/* --- ajouts du Space : recherche, motifs, pagination --- */
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
</style>"""
tete = tete.replace("</style>", css, 1)

# --- en-tête ---------------------------------------------------------------
tete = tete.replace(
    "<h1>Diagnostic Kreyòl GCF</h1>\n    <div class=\"sub\">Phase 1 — correction manuelle des segments Whisper-ht</div>",
    "<h1>Annotation Kreyòl GCF</h1>\n    <div class=\"sub\" id=\"sub\">chargement du corpus…</div>")
tete = tete.replace(
    '<button class="btn" id="theme" title="Basculer le thème">Thème</button>',
    '<input class="qui" id="annotateur" placeholder="ton nom" title="Sépare ton travail de celui des autres">\n'
    '  <button class="btn" id="theme" title="Basculer le thème">Thème</button>')
# Plus de « pré-correction » : le corpus n'en a pas, seule la sortie Whisper existe.
tete = tete.replace('<button class="btn" id="fill">Partir de la pré-correction</button>\n        ', '')
tete = tete.replace('id="copysrc">Copier la source', 'id="copysrc">Reprendre la transcription')
tete = tete.replace(
    '<button class="btn primary" id="export">Exporter le CSV</button>',
    '<a class="btn primary" id="export" href="api/export.csv" download>Exporter le CSV</a>')

# --- colonne de gauche : recherche + motifs + pagination -------------------
ancien_filtres = tete[tete.index('<div class="filters">'):tete.index('<div id="rows"></div>')]
tete = tete.replace(ancien_filtres, """<div class="qbox">
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
    </div>
    <div class="meta"><span id="trouves">…</span><span id="stockage"></span></div>
    """)

entete = ("<!-- Dérivée de docs/index.html : même CSS, même ergonomie.\n"
          "     La logique vit dans app.js, éditable à la main. -->\n")
(racine / "space/static/index.html").write_text(
    entete + tete + marque + '\n<script src="app.js"></script>\n</body>\n</html>\n', encoding="utf-8")
print("écrit :", (racine / "space/static/index.html").stat().st_size, "octets")
