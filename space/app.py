"""Service d'annotation : sert l'interface, relaie l'audio gaté, garde les corrections.

Le jeton Hugging Face vit ici, dans un secret du Space. Il ne descend jamais
dans le navigateur : c'est tout l'intérêt d'avoir un serveur plutôt qu'une page
statique. Le corpus reste sous accès contrôlé, et les 40 333 segments sont
malgré tout écoutables un par un.
"""
from __future__ import annotations

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


SEGMENTS, PAR_CHEMIN = charger_index()


# ---------------------------------------------------------------- corrections

# En mémoire, et poussées vers un dataset : le disque d'un Space est éphémère,
# tout ce qui n'est pas commité disparaît à la mise en veille.
CORRECTIONS: dict[str, dict] = {}
VERROU = threading.Lock()
A_ECRIRE = False
ETAT_DISTANT = "démarrage"


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
                ancienne = CORRECTIONS.get(row["id"])
                # Dernière écriture gagnante : deux annotateurs sur le même
                # segment, c'est la plus récente qui fait foi.
                if not ancienne or row.get("at", 0) >= ancienne.get("at", 0):
                    CORRECTIONS[row["id"]] = row
                    lus += 1
    ETAT_DISTANT = f"{lus} corrections reprises depuis {ANNOTATIONS}"


def ecrire_corrections() -> None:
    """Un fichier par annotateur : deux personnes n'écrasent pas leur travail."""
    global ETAT_DISTANT
    with VERROU:
        instantane = list(CORRECTIONS.values())
    par_annotateur: dict[str, list[dict]] = {}
    for row in instantane:
        par_annotateur.setdefault(row.get("annotateur") or "anonyme", []).append(row)
    for annotateur, rows in par_annotateur.items():
        contenu = "".join(
            json.dumps(r, ensure_ascii=False) + "\n"
            for r in sorted(rows, key=lambda r: r["id"])
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
        A_ECRIRE = False
        try:
            ecrire_corrections()
        except Exception as err:
            # Un jeton en lecture seule tombe ici. On garde tout en mémoire et
            # on le dit à l'interface, plutôt que de perdre le travail en silence.
            ETAT_DISTANT = f"écriture distante impossible ({type(err).__name__}) — export CSV conseillé"


@app.on_event("startup")
def demarrage() -> None:
    charger_corrections()
    threading.Thread(target=boucle_ecriture, daemon=True).start()


# ---------------------------------------------------------------- API

def etat_de(chemin: str) -> str:
    row = CORRECTIONS.get(chemin)
    if not row:
        return "todo"
    if row.get("skipped"):
        return "skip"
    return "done" if (row.get("corrected") or "").strip() else "todo"


@app.get("/api/segments")
def segments(q: str = "", motif: str = "", etat: str = "tous",
             offset: int = 0, limit: int = 200):
    """Piocher dans le corpus : recherche plein texte, motif de rejet, avancement."""
    limit = max(1, min(limit, 500))
    q = q.strip().lower()
    trouves = []
    for seg in SEGMENTS:
        if motif and seg.get("m", "") != motif:
            continue
        if q and q not in seg.get("t", "").lower() and q not in seg["c"].lower():
            continue
        if etat != "tous" and etat_de(seg["c"]) != etat:
            continue
        trouves.append(seg)
    page = trouves[offset:offset + limit]
    return {
        "total": len(trouves),
        "offset": offset,
        "items": [
            {
                "id": s["c"],
                "texte": s.get("t", ""),
                "motif": s.get("m", ""),
                "duree": s.get("d", 0),
                "etat": etat_de(s["c"]),
                "correction": CORRECTIONS.get(s["c"], {}).get("corrected", ""),
                "notes": CORRECTIONS.get(s["c"], {}).get("notes", ""),
            }
            for s in page
        ],
    }


@app.get("/api/stats")
def stats():
    faits = sum(1 for c in CORRECTIONS.values() if (c.get("corrected") or "").strip())
    return {
        "segments": len(SEGMENTS),
        "corriges": faits,
        "ignores": sum(1 for c in CORRECTIONS.values() if c.get("skipped")),
        "annotateurs": sorted({c.get("annotateur") or "anonyme" for c in CORRECTIONS.values()}),
        "stockage": ETAT_DISTANT,
        "corpus": CORPUS,
    }


@app.post("/api/corrections")
async def enregistrer(request: Request):
    global A_ECRIRE
    charge = await request.json()
    annotateur = (charge.get("annotateur") or "anonyme").strip() or "anonyme"
    annotateur = "".join(c for c in annotateur if c.isalnum() or c in "-_")[:40] or "anonyme"
    maintenant = int(time.time() * 1000)
    with VERROU:
        for row in charge.get("rows", []):
            ident = row.get("id")
            if ident not in PAR_CHEMIN:
                continue
            CORRECTIONS[ident] = {
                "id": ident,
                "corrected": row.get("corrected", ""),
                "notes": row.get("notes", ""),
                "skipped": bool(row.get("skipped")),
                "annotateur": annotateur,
                "at": maintenant,
            }
    A_ECRIRE = True
    return {"ok": True, "stockage": ETAT_DISTANT}


@app.get("/api/export.csv")
def export():
    tampon = StringIO()
    tampon.write("segment_id,whisper,motif,corrected,notes,annotateur,at\r\n")
    for ident in sorted(CORRECTIONS):
        row = CORRECTIONS[ident]
        seg = SEGMENTS[PAR_CHEMIN[ident]] if ident in PAR_CHEMIN else {}
        cellules = [ident, seg.get("t", ""), seg.get("m", ""), row.get("corrected", ""),
                    row.get("notes", ""), row.get("annotateur", ""), str(row.get("at", ""))]
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
