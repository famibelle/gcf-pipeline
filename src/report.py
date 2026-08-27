"""
Génère `rapport.md` en agrégeant les artefacts produits par les phases 1 à 5.
Ne recalcule rien : le rapport reflète l'état réel des fichiers sur disque, et
signale explicitement les phases non exécutées plutôt que de les passer sous silence.

Usage :
    python -m src.report
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from .config import CFG
from .io_utils import log, read_json


def _safe(path, loader, default=None):
    try:
        return loader(path)
    except Exception:  # noqa: BLE001
        return default


def build_report(cfg=CFG) -> str:
    rules = _safe(cfg.rules_path, read_json)
    residuals = _safe(cfg.residuals_path, read_json)
    p2 = _safe(cfg.model_dir.parent / "phase2_stats.json", read_json)
    p3 = _safe(cfg.model_dir / "phase3_metrics.json", read_json)
    p4 = _safe(Path(cfg.metrics_path).with_name("phase4_stats.json"), read_json)

    L: list[str] = []
    add = L.append

    add("# Rapport — corpus aligné gcf pour ASR\n")
    add(f"_Généré le {datetime.now().astimezone():%Y-%m-%d %H:%M %Z}_\n")
    add(f"- Corpus source ASR : `{cfg.hf_asr_dataset}`")
    add(f"- Corpus de référence gcf : `{cfg.hf_text_dataset}`")
    add("- Norme orthographique cible : GEREC-2\n")

    # -- Phase 1 -----------------------------------------------------------
    add("## Phase 1 — Diagnostic des substitutions\n")
    if not rules:
        add("> Phase non exécutée : `substitution_rules_ht2gcf.json` absent.\n")
    else:
        meta = rules.get("meta", {})
        add(f"- Paires annotées manuellement : **{meta.get('n_pairs', '?')}**")
        add(f"- Segments identiques ht/gcf après correction : {meta.get('identical_segments', '?')}")
        add(f"- Substitutions candidates observées : {meta.get('n_candidate_substitutions', '?')}")
        add(f"- Règles retenues (support ≥ {meta.get('min_support')}, "
            f"confiance ≥ {meta.get('min_confidence')}) : **{rules.get('n_rules')}**")
        cov = meta.get("coverage_on_annotated", {})
        if cov:
            add(f"- Reconstruction exacte par les seules règles : "
                f"{cov.get('exact_match')}/{cov.get('n')} ({cov.get('exact_match_rate', 0):.1%})")
        add("\n### Règles les plus fréquentes\n")
        add("| ht | gcf | support | confiance |")
        add("|---|---|---|---|")
        top = sorted([r for r in rules["rules"] if r.get("support", 0) > 0],
                     key=lambda r: -r["support"])[:25]
        for r in top:
            add(f"| `{r['src']}` | `{r['tgt']}` | {r['support']} | {r['confidence']:.2f} |")
        unconf = [r for r in rules["rules"] if r.get("status") == "seed_unconfirmed"]
        if unconf:
            add(f"\n{len(unconf)} règles amorces n'ont **jamais été observées** dans les "
                "données annotées. Elles restent hypothétiques et devraient être retirées "
                "si un échantillon plus large ne les confirme pas.\n")
        viol = meta.get("gerec2_violations_in_targets") or {}
        if viol:
            add(f"\nCaractères hors alphabet GEREC-2 relevés dans les corrections "
                f"humaines : `{viol}` — à corriger dans l'annotation.\n")

    if residuals:
        add("\n### Cas résiduels non couverts par les règles\n")
        add(f"{residuals['n_residual_blocks']} divergences observées n'ont pas été "
            "converties en règle. Elles constituent la charge de travail du correcteur neuronal.\n")
        add("| ht | gcf | opération | occurrences |")
        add("|---|---|---|---|")
        for r in residuals.get("top_unexplained", [])[:20]:
            add(f"| `{r['src'] or '∅'}` | `{r['tgt'] or '∅'}` | {r['op']} | {r['count']} |")
        add("")

    # -- Phase 2 -----------------------------------------------------------
    add("\n## Phase 2 — Paires synthétiques\n")
    if not p2:
        add("> Phase non exécutée.\n")
    else:
        add(f"- Paires générées : **{p2.get('total')}** "
            f"({p2.get('paires_generees')})")
        add(f"- Probabilité de corruption : {p2.get('corruption_prob')} | "
            f"bruit ASR : {p2.get('noise_prob')}")
        add(f"- Découpage : {p2.get('split_par')}")
        add("\nLa corruption est stochastique et le découpage se fait par empreinte du "
            "texte cible : sans ces deux précautions, les scores de la Phase 3 seraient "
            "optimistes de plusieurs points.\n")

    # -- Phase 3 -----------------------------------------------------------
    add("\n## Phase 3 — Correcteur neuronal\n")
    if not p3:
        add("> Phase non exécutée.\n")
    else:
        add(f"- Modèle de base : `{CFG.base_model}`")
        add(f"- WER avant correction (test_pairs) : **{p3['wer_before']:.4f}**")
        add(f"- WER après correction : **{p3['wer_after']:.4f}**")
        add(f"- Réduction relative : **{p3['wer_reduction_rel']:.1%}**")
        add(f"- CER : {p3['cer_before']:.4f} → {p3['cer_after']:.4f}")
        add(f"- Segments exactement reconstruits : {p3['exact_match']}/{p3['n']} "
            f"({p3['exact_match_rate']:.1%})\n")
        add("> Ces chiffres portent sur des corruptions **synthétiques**. Ils mesurent "
            "la capacité du modèle à inverser les règles, pas sa performance sur de "
            "vraies sorties Whisper. L'écart entre les deux est la principale "
            "incertitude du projet ; seul l'audit humain de la Phase 4 le quantifie.\n")
        if p3.get("examples"):
            add("### Exemples d'erreurs résiduelles\n")
            for ex in p3["examples"][:8]:
                add(f"- ht : `{ex['source_pseudo_ht']}`")
                add(f"  - prédit : `{ex['prediction']}`")
                add(f"  - attendu : `{ex['reference_gcf']}`")
            add("")

    # -- Phase 4 -----------------------------------------------------------
    add("\n## Phase 4 — Filtrage et livrable\n")
    if not p4:
        add("> Phase non exécutée.\n")
    else:
        add(f"- Segments Whisper traités : {p4['segments_total']}")
        add(f"- Après filtrage (logprob / compression / perplexité) : "
            f"{p4['segments_apres_filtrage']}")
        add(f"- Retenus en haute confiance : **{p4['segments_haute_confiance']}** "
            f"({p4['taux_retention_global']:.1%} du total)")
        add(f"- Durée audio exploitable : **{p4['duree_totale_heures']} h**")
        add(f"- WER estimé moyen : {p4['wer_estime_moyen_haute_confiance']:.4f}")
        add(f"- Motifs de rejet : `{p4.get('motifs_rejet')}`\n")
        add("> Le WER est **estimé** par extrapolation depuis test_pairs, où la vérité "
            "terrain existe. Il n'est pas mesuré sur les données réelles. Pour le "
            "valider, ré-annoter l'échantillon `audit_humain.csv` et comparer.\n")

    # -- Livrables ---------------------------------------------------------
    add("\n## Livrables\n")
    add("| Fichier | Statut |")
    add("|---|---|")
    for label, path in [
        ("substitution_rules_ht2gcf.json", cfg.rules_path),
        ("residual_cases.json", cfg.residuals_path),
        ("train/val/test_pairs.csv", cfg.train_pairs),
        ("gcf_corrector_bytesm.pt", cfg.model_ckpt),
        ("dataset_corrected.jsonl", cfg.dataset_corrected),
        ("dataset_high_confidence.jsonl", cfg.dataset_high_conf),
    ]:
        p = Path(path)
        size = f"{p.stat().st_size / 1e6:.1f} Mo" if p.exists() else "absent"
        add(f"| `{label}` | {size} |")

    add("\n## Limites connues\n")
    add("1. **Taille de l'annotation.** Les règles reposent sur ~300 segments corrigés à "
        "la main. Les substitutions rares y sont sous-représentées et n'atteignent pas "
        "le seuil de support ; elles resteront des erreurs systématiques du correcteur.")
    add("2. **Écart synthétique/réel.** Les corruptions inverses ne reproduisent pas "
        "toutes les erreurs de Whisper (hallucinations, répétitions, segments vides). "
        "Le WER estimé est donc optimiste par construction.")
    add("3. **Substitutions dépendantes du contexte.** `mwen→an` ou `li→i` ne sont "
        "correctes qu'en certaines positions syntaxiques. Sans annotation syntaxique, "
        "les règles surgénèrent et le correcteur hérite de ces erreurs.")
    add("4. **Norme unique.** GEREC-2 est traité comme la cible, alors que le gcf écrit "
        "connaît des variantes. Le corpus produit reflète le choix normatif de "
        "PawolKreyol-gfc, pas l'ensemble des usages.")
    add("5. **Pseudo-labeling.** S'il est activé, chaque tour risque d'amplifier les "
        "biais du correcteur. Le filtre d'accord limite la dérive sans l'éliminer ; "
        "au-delà de deux tours, une ré-annotation humaine est nécessaire.\n")

    return "\n".join(L)


def main(args) -> None:
    text = build_report()
    out = Path(args.out or CFG.report_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    log.info("Rapport -> %s", out)
    print(text[:2000])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Génération du rapport")
    p.add_argument("--out", default=None)
    main(p.parse_args())
