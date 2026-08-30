"""Service d'annotation : sert l'interface, relaie l'audio gaté, garde les corrections.

Le jeton Hugging Face vit ici, dans un secret du Space. Il ne descend jamais
dans le navigateur : c'est tout l'intérêt d'avoir un serveur plutôt qu'une page
statique. Le corpus reste sous accès contrôlé, et les 40 333 segments sont
malgré tout écoutables un par un.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from io import StringIO
from pathlib import Path
from urllib.parse import quote

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import HfApi

RACINE = Path(__file__).parent
CORPUS = os.environ.get("CORPUS_DATASET", "POTOMITAN/potomitan-gcf-transcription")
REVISION = os.environ.get("CORPUS_REVISION", "main")
ANNOTATIONS = os.environ.get("ANNOT_DATASET", "POTOMITAN/gcf-annotations")
JETON = os.environ.get("HF_TOKEN", "")
DELAI_FLUSH = int(os.environ.get("DELAI_FLUSH", "20"))
# Deux passages sur un même extrait séparés de plus d'une heure font deux
# versions ; en deçà, c'est la même séance d'édition qui se poursuit.
DELAI_VERSION = int(os.environ.get("DELAI_VERSION", "3600")) * 1000
# Re-test : un extrait déjà corrigé, resservi vierge après un délai, pour
# mesurer l'accord d'un annotateur avec lui-même. C'est ce qui remplace
# l'accord entre deux personnes quand il n'y en a qu'une.
DELAI_RETEST = int(os.environ.get("DELAI_RETEST", "14")) * 86400 * 1000
TAUX_RETEST = int(os.environ.get("TAUX_RETEST", "30"))  # pour mille

app = FastAPI(title="Annotation GCF")
session = requests.Session()
api = HfApi(token=JETON or None)


# ---------------------------------------------------------------- index

def charger_index() -> tuple[list[dict], dict[str, int]]:
    """Index des segments, construit par scripts/build_space_index.py."""
    chemin = RACINE / "data" / "index.jsonl"
    segments: list[dict] = []
    if chemin.exists():
        with chemin.open(encoding="utf-8") as fh:
            for ligne in fh:
                ligne = ligne.strip()
                if ligne:
                    segments.append(json.loads(ligne))
    return segments, {s["c"]: i for i, s in enumerate(segments)}


def charger_temoins() -> dict[str, str]:
    """Extraits dont la bonne transcription est connue, glissés dans le flux.

    L'annotateur ne sait pas lesquels : c'est ce qui permet de mesurer son
    travail sans qu'un second annotateur ait à repasser derrière lui.
    """
    chemin = RACINE / "data" / "temoins.jsonl"
    connus: dict[str, str] = {}
    if chemin.exists():
        with chemin.open(encoding="utf-8") as fh:
            for ligne in fh:
                ligne = ligne.strip()
                if ligne:
                    row = json.loads(ligne)
                    connus[row["c"]] = row["ref"]
    return connus


SEGMENTS, PAR_CHEMIN = charger_index()
TEMOINS = charger_temoins()
PAR_DUREE = {s["c"]: s.get("d", 0) for s in SEGMENTS}


def nettoyer(annotateur: str | None) -> str:
    """Un nom d'annotateur sert de nom de fichier : il doit rester anodin."""
    propre = "".join(c for c in (annotateur or "").strip() if c.isalnum() or c in "-_")
    return propre[:40] or "anonyme"


# ---------------------------------------------------------------- corrections

# En mémoire, et poussées vers un dataset : le disque d'un Space est éphémère,
# tout ce qui n'est pas commité disparaît à la mise en veille.
#
# La clé est (segment, annotateur) et non le seul segment : deux personnes
# doivent pouvoir corriger le même extrait sans s'effacer, sans quoi il devient
# impossible de mesurer leur accord — et donc de savoir si le travail est bon.
CORRECTIONS: dict[str, dict[str, list[dict]]] = {}
VERROU = threading.Lock()
A_ECRIRE: set[str] = set()
ETAT_DISTANT = "démarrage"


def poser(row: dict) -> None:
    """Range une correction en conservant les reprises ultérieures.

    Les frappes successives d'une même séance remplacent la dernière version ;
    un retour sur l'extrait des jours plus tard en ajoute une. Sans cet
    historique, impossible de comparer un annotateur à lui-même.
    """
    versions = CORRECTIONS.setdefault(row["id"], {}).setdefault(row["annotateur"], [])
    if versions and row.get("at", 0) - versions[-1].get("at", 0) < DELAI_VERSION:
        versions[-1] = row
    else:
        versions.append(row)


def versions_de(chemin: str, annotateur: str) -> list[dict]:
    return CORRECTIONS.get(chemin, {}).get(annotateur, [])


def mienne(chemin: str, annotateur: str) -> dict:
    versions = versions_de(chemin, annotateur)
    return versions[-1] if versions else {}


def remplie(row: dict) -> bool:
    return bool((row.get("corrected") or "").strip())


def charger_corrections() -> None:
    global ETAT_DISTANT
    if not JETON:
        ETAT_DISTANT = "aucun jeton : les corrections restent en mémoire"
        return
    try:
        fichiers = api.list_repo_files(ANNOTATIONS, repo_type="dataset")
    except Exception as err:  # dépôt absent ou jeton sans accès
        ETAT_DISTANT = f"dépôt d'annotations illisible ({type(err).__name__})"
        return
    lus = 0
    moisson: list[dict] = []
    for nom in fichiers:
        if not nom.startswith("corrections/") or not nom.endswith(".jsonl"):
            continue
        try:
            local = api.hf_hub_download(ANNOTATIONS, nom, repo_type="dataset")
        except Exception:
            continue
        with open(local, encoding="utf-8") as fh:
            for ligne in fh:
                ligne = ligne.strip()
                if not ligne:
                    continue
                row = json.loads(ligne)
                # Le nom du fichier fait foi : une ligne sans annotateur vient
                # d'une version antérieure du service.
                row.setdefault("annotateur", nom[len("corrections/"):-len(".jsonl")])
                moisson.append(row)
                lus += 1
    # L'ordre chronologique fait foi : c'est lui qui découpe les versions.
    for row in sorted(moisson, key=lambda r: r.get("at", 0)):
        poser(row)
    ETAT_DISTANT = f"{lus} corrections reprises depuis {ANNOTATIONS}"


def ecrire_corrections(a_ecrire: set[str]) -> None:
    """Un fichier par annotateur, et seulement ceux qui ont bougé."""
    global ETAT_DISTANT
    with VERROU:
        par_annotateur: dict[str, list[dict]] = {a: [] for a in a_ecrire}
        for par_ann in CORRECTIONS.values():
            for annotateur, versions in par_ann.items():
                if annotateur in par_annotateur:
                    par_annotateur[annotateur].extend(versions)
    for annotateur, rows in par_annotateur.items():
        contenu = "".join(
            json.dumps(r, ensure_ascii=False) + "\n"
            for r in sorted(rows, key=lambda r: (r["id"], r.get("at", 0)))
        )
        api.upload_file(
            path_or_fileobj=contenu.encode("utf-8"),
            path_in_repo=f"corrections/{annotateur}.jsonl",
            repo_id=ANNOTATIONS,
            repo_type="dataset",
            commit_message=f"Corrections de {annotateur} ({len(rows)} segments)",
        )
    ETAT_DISTANT = f"sauvegardé dans {ANNOTATIONS} à {time.strftime('%H:%M')}"


def boucle_ecriture() -> None:
    global A_ECRIRE, ETAT_DISTANT
    while True:
        time.sleep(DELAI_FLUSH)
        if not A_ECRIRE:
            continue
        with VERROU:
            a_ecrire, A_ECRIRE = A_ECRIRE, set()
        try:
            ecrire_corrections(a_ecrire)
        except Exception as err:
            with VERROU:
                A_ECRIRE |= a_ecrire  # rien n'est abandonné, on retentera
            # Un jeton en lecture seule tombe ici. On garde tout en mémoire et
            # on le dit à l'interface, plutôt que de perdre le travail en silence.
            ETAT_DISTANT = f"écriture distante impossible ({type(err).__name__}) — export CSV conseillé"


@app.on_event("startup")
def demarrage() -> None:
    charger_corrections()
    threading.Thread(target=boucle_ecriture, daemon=True).start()


# ---------------------------------------------------------------- API

def est_retest(chemin: str, annotateur: str) -> bool:
    """Un extrait déjà corrigé, resservi vierge longtemps après.

    Le tirage est déterministe : le même extrait reste choisi d'une requête à
    l'autre, sinon il réapparaîtrait corrigé au rechargement de la page.
    """
    versions = versions_de(chemin, annotateur)
    # Une seule reprise suffit à la mesure ; au-delà on ferait retravailler
    # quelqu'un pour rien.
    if len(versions) != 1 or not remplie(versions[0]):
        return False
    if time.time() * 1000 - versions[0].get("at", 0) < DELAI_RETEST:
        return False
    graine = hashlib.blake2b(f"{chemin}|{annotateur}".encode(), digest_size=8).digest()
    return int.from_bytes(graine, "big") % 1000 < TAUX_RETEST


def affichage(chemin: str, annotateur: str) -> tuple[str, str, str]:
    """Ce que l'annotateur voit : état, correction, notes."""
    if est_retest(chemin, annotateur):
        return "todo", "", ""
    row = mienne(chemin, annotateur)
    return etat_de(chemin, annotateur), row.get("corrected", ""), row.get("notes", "")


def etat_de(chemin: str, annotateur: str) -> str:
    """L'avancement affiché est celui de l'annotateur, pas celui des autres."""
    row = mienne(chemin, annotateur)
    if not row:
        return "todo"
    if row.get("inutilisable"):
        return "rebut"
    if row.get("skipped"):
        return "skip"
    return "done" if remplie(row) else "todo"


@app.get("/api/segments")
def segments(q: str = "", motif: str = "", etat: str = "tous", annotateur: str = "",
             offset: int = 0, limit: int = 200):
    """Piocher dans le corpus : recherche plein texte, motif de rejet, avancement."""
    limit = max(1, min(limit, 500))
    annotateur = nettoyer(annotateur)
    q = q.strip().lower()
    trouves = []
    for seg in SEGMENTS:
        if motif and seg.get("m", "") != motif:
            continue
        if q and q not in seg.get("t", "").lower() and q not in seg["c"].lower():
            continue
        if etat != "tous" and affichage(seg["c"], annotateur)[0] != etat:
            continue
        trouves.append(seg)
    page = injecter_temoins(trouves[offset:offset + limit], annotateur, q, motif)
    vues = {s["c"]: affichage(s["c"], annotateur) for s in page}
    return {
        "total": len(trouves),
        "offset": offset,
        "items": [
            {
                "id": s["c"],
                "texte": s.get("t", ""),
                "motif": s.get("m", ""),
                "duree": s.get("d", 0),
                "etat": vues[s["c"]][0],
                "correction": vues[s["c"]][1],
                "notes": vues[s["c"]][2],
                # Le nombre de versions des autres, jamais leur texte : afficher
                # une correction déjà écrite biaiserait toute mesure d'accord.
                "autres": sum(1 for a, v in CORRECTIONS.get(s["c"], {}).items()
                              if a != annotateur and v and remplie(v[-1])),
            }
            for s in page
        ],
    }


def injecter_temoins(page: list[dict], annotateur: str, q: str, motif: str) -> list[dict]:
    """Glisse quelques témoins non encore traités dans la page de résultats.

    Seulement en navigation libre : injectés sous un filtre par motif, ils
    dépareilleraient et se repéreraient d'un coup d'œil.
    """
    if not TEMOINS or q or motif or annotateur == "anonyme":
        return page
    deja = {s["c"] for s in page}
    candidats = [c for c in TEMOINS
                 if c in PAR_CHEMIN and c not in deja and not versions_de(c, annotateur)]
    for k, chemin in enumerate(candidats[:2]):
        page.insert(min(len(page), 3 + 7 * k), SEGMENTS[PAR_CHEMIN[chemin]])
    return page


@app.get("/api/stats")
def stats():
    versions = [r for par_ann in CORRECTIONS.values()
                for liste in par_ann.values() for r in liste]
    return {
        "segments": len(SEGMENTS),
        # Segments couverts au moins une fois, et total des versions : l'écart
        # entre les deux, c'est la double annotation déjà faite.
        "corriges": sum(1 for par_ann in CORRECTIONS.values()
                        if any(remplie(l[-1]) for l in par_ann.values() if l)),
        "versions": sum(1 for r in versions if remplie(r)),
        "reprises": sum(len(l) - 1 for par_ann in CORRECTIONS.values()
                        for l in par_ann.values() if len(l) > 1),
        "doubles": sum(1 for par_ann in CORRECTIONS.values()
                       if sum(1 for l in par_ann.values() if l and remplie(l[-1])) > 1),
        "rebuts": sum(1 for par_ann in CORRECTIONS.values()
                      if any(l and l[-1].get("inutilisable") for l in par_ann.values())),
        "temoins": len(TEMOINS),
        "annotateurs": sorted({r["annotateur"] for r in versions}),
        "stockage": ETAT_DISTANT,
        "corpus": CORPUS,
    }


@app.post("/api/corrections")
async def enregistrer(request: Request):
    charge = await request.json()
    annotateur = nettoyer(charge.get("annotateur"))
    maintenant = int(time.time() * 1000)
    with VERROU:
        for row in charge.get("rows", []):
            ident = row.get("id")
            if ident not in PAR_CHEMIN:
                continue
            poser({
                "id": ident,
                "annotateur": annotateur,
                "corrected": row.get("corrected", ""),
                "notes": row.get("notes", ""),
                "skipped": bool(row.get("skipped")),
                # « Inutilisable » n'est pas « ignoré » : le premier juge
                # l'extrait, le second l'annotateur. Les confondre pousserait à
                # inventer une transcription plutôt qu'à laisser un blanc.
                "inutilisable": bool(row.get("inutilisable")),
                # De quoi repérer une correction écrite sans avoir écouté.
                "ecoute_ms": int(row.get("ecoute_ms") or 0),
                "lectures": int(row.get("lectures") or 0),
                "duree": PAR_DUREE.get(ident, 0),
                "at": maintenant,
            })
        A_ECRIRE.add(annotateur)
    return {"ok": True, "stockage": ETAT_DISTANT}


@app.get("/api/export.csv")
def export():
    tampon = StringIO()
    # Une ligne par (segment, annotateur) : c'est ce qui permettra de comparer
    # deux versions du même extrait, donc de mesurer si le travail est bon.
    tampon.write("segment_id,whisper,motif,duree_ms,corrected,notes,annotateur,"
                 "version,etat,ecoute_ms,lectures,at\r\n")
    for ident in sorted(CORRECTIONS):
        seg = SEGMENTS[PAR_CHEMIN[ident]] if ident in PAR_CHEMIN else {}
        for annotateur in sorted(CORRECTIONS[ident]):
            for n, row in enumerate(CORRECTIONS[ident][annotateur], 1):
                cellules = [ident, seg.get("t", ""), seg.get("m", ""), str(seg.get("d", 0)),
                            row.get("corrected", ""), row.get("notes", ""), annotateur,
                            str(n), etat_de(ident, annotateur), str(row.get("ecoute_ms", 0)),
                            str(row.get("lectures", 0)), str(row.get("at", ""))]
                tampon.write(",".join(
                    '"' + c.replace('"', '""') + '"' if any(x in c for x in ',"\r\n') else c
                    for c in cellules
                ) + "\r\n")
    return PlainTextResponse(
        tampon.getvalue(), media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="corrections_gcf.csv"'},
    )


# ---------------------------------------------------------------- audio

RELAYES = ("content-type", "content-length", "content-range", "accept-ranges", "etag")


@app.get("/api/audio/{chemin:path}")
def audio(chemin: str, request: Request):
    # Seuls les chemins de l'index sont servis : sans ce garde-fou, le Space
    # deviendrait un proxy ouvert vers tout dépôt lisible par le jeton.
    if chemin not in PAR_CHEMIN:
        raise HTTPException(404, "segment inconnu")
    url = (f"https://huggingface.co/datasets/{CORPUS}/resolve/{REVISION}/"
           + quote(chemin))
    entetes = {}
    if JETON:
        entetes["Authorization"] = f"Bearer {JETON}"
    # La barre de progression envoie des requêtes Range : les relayer, sinon
    # impossible de se déplacer dans l'extrait.
    if plage := request.headers.get("range"):
        entetes["Range"] = plage
    amont = session.get(url, headers=entetes, stream=True, timeout=30)
    if amont.status_code >= 400:
        amont.close()
        raise HTTPException(amont.status_code, "audio indisponible")
    sortie = {k: v for k, v in amont.headers.items() if k.lower() in RELAYES}
    sortie["Cache-Control"] = "private, max-age=3600"
    return StreamingResponse(
        amont.iter_content(64 * 1024),
        status_code=amont.status_code,
        media_type=amont.headers.get("content-type", "audio/mpeg"),
        headers=sortie,
    )


# L'interface est servie en dernier : /api/* garde la priorité sur le montage.
app.mount("/", StaticFiles(directory=RACINE / "static", html=True), name="static")
