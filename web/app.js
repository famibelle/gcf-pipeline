/* Interface d'annotation — couche données.
   Dérivée de docs/index.html : le CSS, le diff, le lecteur, les raccourcis et
   le moteur de suggestion kréyòl sont repris tels quels. Ce qui change, c'est
   d'où viennent les segments (une API au lieu d'un tableau figé) et où partent
   les corrections (un dataset partagé au lieu du seul navigateur). */
/* Deux hébergements, une seule interface. En « serveur » (Space Hugging Face)
   les segments viennent d'une API qui détient le jeton du corpus gaté ; en
   « statique » (GitHub Pages) ils viennent d'un lot construit à l'avance et
   les corrections ne quittent pas le navigateur. Tout le reste est commun. */
const MODE = window.GCF_MODE || "serveur";
const API = "api";
const CLE = "gcf-annot-v1";
let CATALOGUE = [];   // mode statique : le lot entier, chargé une fois

let DATA = [];                 // page de résultats affichée
const etat = new Map();        // id -> {corrected, notes, skipped}
let active = 0, total = 0, stats = {segments: 0, corriges: 0};
let requete = {q: "", motif: "", etat: "tous", offset: 0, limit: 200};
let annotateur = "";
const enAttente = new Map();   // corrections pas encore acceptées par le serveur
let envoiTimer = null, paintTimer = null, qTimer = null;

const el = id => document.getElementById(id);
const court = id => { const b = id.split("/").pop().replace(/\.mp3$/, ""); return b.length > 42 ? b.slice(0, 39) + "…" : b; };
const secondes = ms => ms ? (ms / 1000).toFixed(1).replace(".", ",") + " s" : "";

/* ---------- état local ---------- */
function ligne(id){
  if(!etat.has(id)) etat.set(id, {corrected: "", notes: "", skipped: false, note: 0,
                                  vote: false, jalons: {}});
  const s = etat.get(id);
  if(!s.jalons) s.jalons = {};
  return s;
}
/* Un jalon ne se pose qu'une fois : c'est la première occurrence qui informe.
   Ouverture, première écoute, première frappe, note, validation — leurs écarts
   disent le soin pris, là où un seul horodatage de fin ne dit rien. */
function jalon(id, nom){
  const j = ligne(id).jalons;
  if(!j[nom]){ j[nom] = Date.now(); return true; }
  return false;
}
function travail(id){
  const s = etat.get(id);
  return !!s && (s.corrected.trim() || s.skipped || s.note > 0);
}
function stateOf(i){
  const s = etat.get(DATA[i].id);
  if(!s) return DATA[i].etat || "todo";
  // Une étoile vaut « inexploitable » : la note absorbe l'ancien bouton.
  if(s.note === 1) return "rebut";
  if(s.skipped) return "skip";
  return s.corrected.trim() ? "done" : "todo";
}
const LABEL = {todo: "à faire", done: "corrigé", skip: "ignoré", rebut: "inexploitable"};
const ANCRAGES = {
  0: "non notée",
  1: "inexploitable",
  2: "deviné, des passages résistent",
  3: "sens bon, orthographe incertaine",
  4: "correct, un doute ponctuel",
  5: "gcf validé — fait foi",
};

/* ---------- écoute réelle ----------
   Compte les secondes d'audio effectivement entendues, pas le temps passé
   devant l'écran : une correction écrite sans avoir écouté se repère ainsi. */
const ecoute = new Map();   // id -> {ms, lectures}
function compteur(id){
  if(!ecoute.has(id)) ecoute.set(id, {ms: 0, lectures: 0});
  return ecoute.get(id);
}

function lireLocal(){
  try{
    const brut = localStorage.getItem(CLE);
    if(!brut) return;
    const j = JSON.parse(brut);
    (j.rows || []).forEach(r => etat.set(r.id, {corrected: r.corrected || "", notes: r.notes || "",
                                                skipped: !!r.skipped, note: r.note || 0,
                                                jalons: r.jalons || {}}));
    (j.ecoute || []).forEach(([id, c]) => ecoute.set(id, c));
    (j.attente || []).forEach(r => enAttente.set(r.id, r));
  }catch(e){}
}
function ecrireLocal(){
  try{
    const rows = [...etat].map(([id, s]) => ({id, ...s}));
    localStorage.setItem(CLE, JSON.stringify({at: Date.now(), rows, attente: [...enAttente.values()],
                                              ecoute: [...ecoute]}));
  }catch(e){}
}

/* ---------- envoi au serveur ---------- */
function persist(id){
  const s = ligne(id), e = compteur(id);
  enAttente.set(id, {id, corrected: s.corrected, notes: s.notes, skipped: s.skipped,
                     note: s.note, vote: !!s.vote, jalons: s.jalons,
                     ecoute_ms: Math.round(e.ms), lectures: e.lectures});
  ecrireLocal();
  el("saved").textContent = "brouillon";
  clearTimeout(envoiTimer);
  envoiTimer = setTimeout(envoyer, 2500);
}
async function envoyer(){
  if(!enAttente.size) return;
  if(MODE === "statique"){
    // Rien à joindre : le travail vit dans ce navigateur, l'export CSV le
    // fait sortir. C'est le prix d'une page sans serveur.
    enAttente.clear(); ecrireLocal();
    el("saved").textContent = "enregistré dans ce navigateur";
    majStats();
    return;
  }
  const rows = [...enAttente.values()];
  enAttente.clear();
  try{
    const r = await fetch(API + "/corrections", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({annotateur, rows}),
    });
    if(!r.ok) throw new Error(r.status);
    const j = await r.json();
    el("stockage").textContent = j.stockage || "";
    el("saved").textContent = "enregistré";
    ecrireLocal();
    majStats();
  }catch(e){
    // Rien n'est perdu : la file repart au prochain essai, et le brouillon
    // reste dans le navigateur en attendant.
    rows.forEach(r => enAttente.set(r.id, r));
    ecrireLocal();
    el("saved").textContent = "hors ligne — nouvel essai";
    clearTimeout(envoiTimer);
    envoiTimer = setTimeout(envoyer, 8000);
  }
}

/* ---------- recherche dans le corpus ---------- */
function filtrerLocal(){
  const q = requete.q.trim().toLowerCase();
  return CATALOGUE.filter(s =>
    (!requete.motif || s.m === requete.motif)
    && (!q || s.t.toLowerCase().includes(q) || s.c.toLowerCase().includes(q))
    && (requete.etat === "tous" || etatDe(s.c) === requete.etat));
}
function etatDe(id){
  const s = etat.get(id);
  if(!s) return "todo";
  if(s.note === 1) return "rebut";
  if(s.skipped) return "skip";
  return s.corrected.trim() ? "done" : "todo";
}
async function chercher(remise){
  if(remise) requete.offset = 0;
  const idAvant = DATA[active] && DATA[active].id;
  el("trouves").textContent = "recherche…";
  let j;
  if(MODE === "statique"){
    const trouves = filtrerLocal();
    j = {total: trouves.length, items: trouves.slice(0, requete.offset + requete.limit)
      .map(s => ({id: s.c, texte: s.t, motif: s.m, duree: s.d, fichier: s.f,
                  etat: etatDe(s.c), correction: (etat.get(s.c) || {}).corrected || "",
                  notes: (etat.get(s.c) || {}).notes || "", autres: 0}))};
    remise = true;   // tout est déjà en mémoire : la page se refait d'un bloc
  }else{
    const p = new URLSearchParams({q: requete.q, motif: requete.motif, etat: requete.etat,
                                   annotateur, offset: requete.offset, limit: requete.limit});
    try{ j = await (await fetch(API + "/segments?" + p)).json(); }
    catch(e){ el("trouves").textContent = "serveur injoignable"; return; }
  }
  total = j.total;
  const arrivants = j.items.filter(it => !DATA.some(d => d.id === it.id));
  DATA = remise ? j.items : DATA.concat(arrivants);
  j.items.forEach(it => {
    // Le serveur fait autorité sur ce qu'il a déjà accepté. S'il resert un
    // segment vierge, c'est délibéré : le cache local doit lâcher prise, sans
    // quoi un brouillon oublié ici réapparaîtrait et fausserait la mesure.
    if(etat.has(it.id) && !enAttente.has(it.id) && it.etat === "todo"
       && !it.correction && !it.notes){
      etat.delete(it.id);
      ecoute.delete(it.id);
    }
    if(!etat.has(it.id) && (it.correction || it.notes || it.etat !== "todo"))
      etat.set(it.id, {corrected: it.correction || "", notes: it.notes || "",
                       skipped: it.etat === "skip", note: it.note || 0, jalons: {}});
  });
  el("trouves").textContent = total.toLocaleString("fr-FR") + " segment" + (total > 1 ? "s" : "")
    + (DATA.length < total ? ` · ${DATA.length} affichés` : "");
  renderList();
  if(remise){
    active = Math.min(active, Math.max(0, DATA.length - 1));
    if(!DATA.length) videPanneau();
    // Ne pas relancer go() si l'extrait affiché n'a pas changé : cela couperait
    // la lecture en cours à chaque frappe dans la recherche.
    else if(DATA[active].id !== idAvant) go(active);
    else { paint(); renderList(); }
  }
}
function videPanneau(){
  el("segid").textContent = "—";
  el("source").textContent = "";
  el("edit").value = "";
  audio.pause(); audio.removeAttribute("src");
}

/* ---------- liste ---------- */
const rowsEl = el("rows");
function renderList(){
  const frag = document.createDocumentFragment();
  if(!DATA.length){
    const v = document.createElement("div");
    v.className = "vide"; v.textContent = "Aucun segment pour ce filtre.";
    frag.appendChild(v);
  }
  DATA.forEach((d, i) => {
    const st = stateOf(i);
    const b = document.createElement("button");
    b.className = "row"; b.dataset.state = st; b.dataset.i = i;
    b.setAttribute("aria-current", i === active ? "true" : "false");
    const dot = document.createElement("span"); dot.className = "dot";
    const box = document.createElement("span");
    const id = document.createElement("span"); id.className = "id mono";
    id.textContent = court(d.id);
    if(d.motif){
      const m = document.createElement("span");
      m.className = "motif"; m.dataset.m = d.motif; m.textContent = d.motif;
      id.appendChild(m);
    }
    const s = etat.get(d.id);
    if(s && s.note){
      const n = document.createElement("span");
      n.className = "note-mini"; n.textContent = "★".repeat(s.note);
      id.appendChild(n);
    }
    const tx = document.createElement("span"); tx.className = "txt kreyol";
    tx.textContent = (s && s.corrected.trim()) || d.texte || "—";
    box.appendChild(id); box.appendChild(document.createElement("br")); box.appendChild(tx);
    b.appendChild(dot); b.appendChild(box);
    frag.appendChild(b);
  });
  if(DATA.length < total){
    const plus = document.createElement("button");
    plus.className = "plus"; plus.id = "plus";
    plus.textContent = `Charger ${Math.min(requete.limit, total - DATA.length)} de plus`;
    frag.appendChild(plus);
  }
  rowsEl.replaceChildren(frag);
}
rowsEl.addEventListener("click", e => {
  if(e.target.closest("#plus")){ requete.offset = DATA.length; chercher(false); return; }
  const b = e.target.closest(".row");
  if(b) go(Number(b.dataset.i));
});

/* ---------- filtres et recherche ---------- */
document.querySelectorAll(".chip").forEach(c => c.addEventListener("click", () => {
  document.querySelectorAll(".chip").forEach(x => x.setAttribute("aria-pressed", "false"));
  c.setAttribute("aria-pressed", "true");
  const f = c.dataset.filter;
  requete.motif = (f && f !== "tous") ? f : "";
  requete.etat = c.dataset.etat || "tous";
  chercher(true);
}));
el("q").addEventListener("input", e => {
  requete.q = e.target.value;
  clearTimeout(qTimer);
  qTimer = setTimeout(() => chercher(true), 300);
});
el("hasard").addEventListener("click", () => {
  // Piocher au hasard : un saut dans la pagination suffit, le serveur garde
  // l'ordre stable donc on retombe sur une tranche cohérente.
  requete.offset = Math.max(0, Math.floor(Math.random() * Math.max(1, total - requete.limit)));
  DATA = [];
  chercher(false).then(() => { if(DATA.length){ active = 0; go(0); } });
});
let quiTimer = null;
el("annotateur").addEventListener("input", e => {
  annotateur = e.target.value.trim();
  try{ localStorage.setItem("gcf-annotateur", annotateur); }catch(err){}
  // L'avancement est celui de la personne : changer de nom rebat la liste.
  clearTimeout(quiTimer);
  quiTimer = setTimeout(() => { etat.clear(); chercher(true); majStats(); }, 500);
});

/* ---------- diff mot à mot ---------- */
function words(t){ return (t || "").trim().split(/\s+/).filter(Boolean); }
function diffMark(srcText, curText){
  const a = words(srcText), b = words(curText);
  if(!b.length) return srcText;
  const n = a.length, m = b.length;
  const dp = Array.from({length: n + 1}, () => new Uint16Array(m + 1));
  for(let i = n - 1; i >= 0; i--) for(let j = m - 1; j >= 0; j--)
    dp[i][j] = a[i].toLowerCase() === b[j].toLowerCase() ? dp[i+1][j+1] + 1 : Math.max(dp[i+1][j], dp[i][j+1]);
  const out = []; let i = 0, j = 0;
  while(i < n && j < m){
    if(a[i].toLowerCase() === b[j].toLowerCase()){ out.push({w: a[i], same: true}); i++; j++; }
    else if(dp[i+1][j] >= dp[i][j+1]){ out.push({w: a[i], same: false}); i++; }
    else j++;
  }
  while(i < n){ out.push({w: a[i], same: false}); i++; }
  const frag = document.createDocumentFragment();
  out.forEach((o, k) => {
    const node = o.same ? document.createTextNode(o.w) : Object.assign(document.createElement("mark"), {textContent: o.w});
    frag.appendChild(node);
    if(k < out.length - 1) frag.appendChild(document.createTextNode(" "));
  });
  return frag;
}

/* ---------- panneau ---------- */
const audio = new Audio();
function go(i){
  if(i < 0 || i >= DATA.length) return;
  // Le temps d'écoute du segment qu'on quitte n'est envoyé que s'il a donné
  // lieu à quelque chose : écouter sans corriger ne crée pas de ligne.
  const precedent = DATA[active];
  if(precedent && i !== active && travail(precedent.id)) persist(precedent.id);
  active = i;
  dernierTemps = 0;
  jalon(DATA[i].id, "ouvert");
  const d = DATA[i], s = ligne(d.id);
  el("segid").textContent = court(d.id) + " · " + (i + 1) + "/" + DATA.length
    + (d.duree ? " · " + secondes(d.duree) : "");
  el("edit").value = s.corrected;
  el("notes").value = s.notes;
  el("saved").textContent = "";
  // encodeURI garde les « / » du chemin et échappe le reste.
  audio.pause();
  audio.src = MODE === "statique" ? "audio/" + encodeURI(d.fichier)
                                  : API + "/audio/" + encodeURI(d.id);
  audio.currentTime = 0; setIcon(false);
  el("cur").textContent = "0:00"; el("dur").textContent = "0:00"; el("scrub").value = 0;
  paint(); renderList(); refreshSuggest();
  const cur = rowsEl.querySelector('[aria-current="true"]');
  if(cur) cur.scrollIntoView({block: "nearest"});
}
function paint(){
  if(!DATA.length) return;
  const d = DATA[active], s = ligne(d.id), st = stateOf(active);
  const src = el("source");
  const marked = diffMark(d.texte, s.corrected);
  if(typeof marked === "string") src.textContent = marked; else src.replaceChildren(marked);
  const tag = el("segtag"); tag.className = "tag " + st; tag.textContent = LABEL[st];
  peindreEtoiles();
  el("count").textContent = stats.corriges.toLocaleString("fr-FR") + " / " + stats.segments.toLocaleString("fr-FR");
  el("bar").style.width = (stats.segments ? 100 * stats.corriges / stats.segments : 0).toFixed(2) + "%";
}
function note(msg){
  const t = el("toast"); t.textContent = msg; t.classList.add("on");
  clearTimeout(note._t); note._t = setTimeout(() => t.classList.remove("on"), 2200);
}
async function majStats(){
  if(MODE === "statique"){
    let faits = 0, rebuts = 0;
    etat.forEach(s => { if(s.corrected.trim()) faits++; if(s.inutilisable) rebuts++; });
    stats = {segments: CATALOGUE.length, corriges: faits, versions: faits, doubles: 0, rebuts};
    el("sub").textContent = CATALOGUE.length.toLocaleString("fr-FR") + " segments du lot · "
      + faits.toLocaleString("fr-FR") + " corrigés" + (rebuts ? " · " + rebuts + " inutilisables" : "");
    el("stockage").textContent = "sauvegarde locale — exporte le CSV";
    paint();
    return;
  }
  try{
    stats = await (await fetch(API + "/stats")).json();
    el("sub").textContent = stats.segments.toLocaleString("fr-FR") + " segments · "
      + stats.versions.toLocaleString("fr-FR") + " corrections"
      + (stats.doubles ? " dont " + stats.doubles + " en double" : "")
      + (stats.rebuts ? " · " + stats.rebuts + " inutilisables" : "");
    el("stockage").textContent = stats.stockage || "";
    paint();
  }catch(e){}
}

/* ---------- édition ---------- */
el("edit").addEventListener("input", e => {
  const d = DATA[active]; if(!d) return;
  const s = ligne(d.id);
  if(e.target.value.trim()) jalon(d.id, "edite");
  if(e.target.value !== d.texte) s.vote = false;
  s.corrected = e.target.value;
  if(e.target.value.trim()){ s.skipped = false; if(s.note === 1) s.note = 0; }
  persist(d.id); refreshSuggest();
  clearTimeout(paintTimer); paintTimer = setTimeout(() => { paint(); renderList(); }, 220);
});
el("notes").addEventListener("input", e => {
  const d = DATA[active]; if(!d) return;
  ligne(d.id).notes = e.target.value; persist(d.id);
});
el("copysrc").addEventListener("click", () => {
  const d = DATA[active]; if(!d) return;
  const s = ligne(d.id);
  el("edit").value = d.texte; s.corrected = d.texte; s.skipped = false;
  if(s.note === 1) s.note = 0;
  jalon(d.id, "edite");
  persist(d.id); paint(); renderList(); el("edit").focus();
});
el("skip").addEventListener("click", () => {
  const d = DATA[active]; if(!d) return;
  const s = ligne(d.id);
  s.skipped = true; s.note = 0; s.corrected = ""; el("edit").value = "";
  persist(d.id); paint(); renderList(); note("Segment ignoré — à reprendre plus tard");
});
/* ---------- vote ---------- */
function voter(){
  const d = DATA[active]; if(!d || !d.texte.trim()) return;
  const s = ligne(d.id);
  // Le vote dit « rien à corriger » : le texte de Whisper devient la version
  // validée, en un geste au lieu de trois.
  s.corrected = d.texte;
  s.note = 5;
  s.skipped = false;
  s.vote = true;
  el("edit").value = d.texte;
  jalon(d.id, "edite"); jalon(d.id, "note"); jalon(d.id, "valide");
  persist(d.id); paint(); renderList();
  note("Validée telle quelle");
  nextTodo();
}
el("bon").addEventListener("click", voter);

/* ---------- note de confiance ---------- */
function noter(valeur){
  const d = DATA[active]; if(!d) return;
  const s = ligne(d.id);
  s.note = s.note === valeur ? 0 : valeur;   // recliquer la même étoile l'annule
  if(s.note){
    jalon(d.id, "note");
    s.skipped = false;
    // Une étoile juge l'extrait inexploitable : le texte n'a plus lieu d'être.
    if(s.note === 1){ s.corrected = ""; el("edit").value = ""; }
  }
  persist(d.id); paint(); renderList();
}
el("etoiles").addEventListener("click", e => {
  const b = e.target.closest(".etoile");
  if(b) noter(Number(b.dataset.note));
});
function peindreEtoiles(){
  const d = DATA[active];
  const valeur = d ? ligne(d.id).note : 0;
  document.querySelectorAll(".etoile").forEach(b => {
    b.dataset.on = Number(b.dataset.note) <= valeur ? "1" : "0";
  });
  const txt = el("etoile-txt");
  txt.textContent = ANCRAGES[valeur];
  txt.dataset.note = valeur;
}
el("next").addEventListener("click", () => nextTodo());
function nextTodo(){
  const d = DATA[active];
  if(d){
    const s = ligne(d.id);
    // Ne rien pré-remplir : une étoile par défaut serait l'inflation même.
    if(s.corrected.trim() && !s.note){
      note("Note la transcription — Alt+1 à Alt+5");
      el("etoiles").querySelector('[data-note="5"]').focus();
      return;
    }
    jalon(d.id, "valide");
    if(travail(d.id)) persist(d.id);
  }
  for(let k = 1; k <= DATA.length; k++){
    const i = (active + k) % DATA.length;
    if(stateOf(i) === "todo"){ go(i); el("edit").focus(); return; }
  }
  if(DATA.length < total){ requete.offset = DATA.length; chercher(false).then(() => nextTodo()); return; }
  note("Plus rien à faire dans cette sélection");
}

/* ---------- lecture ---------- */
function setIcon(playing){
  el("icon").innerHTML = playing ? '<path d="M6 5h4v14H6zM14 5h4v14h-4z"></path>' : '<path d="M8 5v14l11-7z"></path>';
}
function fmt(t){ if(!isFinite(t)) return "0:00"; const m = Math.floor(t / 60), s = Math.floor(t % 60); return m + ":" + String(s).padStart(2, "0"); }
function toggle(){ if(audio.paused){ audio.play().catch(() => {}); } else audio.pause(); }
el("play").addEventListener("click", toggle);
let dernierTemps = 0;
audio.addEventListener("play", () => {
  setIcon(true);
  const d = DATA[active];
  if(d){ compteur(d.id).lectures++; jalon(d.id, "ecoute"); }
  dernierTemps = audio.currentTime;
});
audio.addEventListener("pause", () => setIcon(false));
audio.addEventListener("ended", () => setIcon(false));
audio.addEventListener("error", () => { if(audio.getAttribute("src")) note("Audio indisponible pour ce segment"); });
audio.addEventListener("loadedmetadata", () => el("dur").textContent = fmt(audio.duration));
audio.addEventListener("timeupdate", () => {
  // Le pas est borné : un saut dans la barre ne compte pas comme de l'écoute.
  const pas = audio.currentTime - dernierTemps;
  dernierTemps = audio.currentTime;
  const d = DATA[active];
  if(d && pas > 0 && pas < 1.5) compteur(d.id).ms += pas * 1000;
  el("cur").textContent = fmt(audio.currentTime);
  if(audio.duration) el("scrub").value = Math.round(1000 * audio.currentTime / audio.duration);
});
el("scrub").addEventListener("input", e => {
  if(audio.duration) audio.currentTime = audio.duration * e.target.value / 1000;
  dernierTemps = audio.currentTime;
});
el("again").addEventListener("click", () => { audio.currentTime = 0; audio.play().catch(() => {}); });
document.querySelectorAll(".rate").forEach(r => r.addEventListener("click", () => {
  document.querySelectorAll(".rate").forEach(x => x.setAttribute("aria-pressed", "false"));
  r.setAttribute("aria-pressed", "true"); audio.playbackRate = Number(r.dataset.rate);
}));

/* ---------- prédiction kréyòl (moteur KreyolKeyb) ---------- */
async function loadLexicon(){
  const j = p => fetch(p, {cache: "force-cache"}).then(r => r.ok ? r.json() : null);
  const [dict, ngrams, french] = await Promise.all([
    j("assets/kreyol/creole_dict.json"),
    j("assets/kreyol/creole_ngrams.json"),
    j("assets/kreyol/french_simple_dict.json"),
  ]);
  return {dict, ngrams, french};
}
let engine = null;
const WORD_RE = /[\p{L}\p{M}'’-]+/gu;
const TAIL_RE = /[\p{L}\p{M}'’-]*$/u;
const HEAD_RE = /^[\p{L}\p{M}'’-]*/u;

async function bootPredict(){
  if(typeof KreyolSimulatorEngine === "undefined") return;
  let lex = null;
  try{ lex = await loadLexicon(); }catch(e){ return; }
  if(!lex || !lex.dict) return;
  const e = new KreyolSimulatorEngine.SuggestionEngine();
  e.loadDictionary(lex.dict);
  e.loadNgramModel(lex.ngrams || {});
  if(lex.french) e.loadFrenchDictionary(lex.french);
  engine = e;
  refreshSuggest();
}
function tail(){
  const ta = el("edit"), pos = ta.selectionStart;
  const before = ta.value.slice(0, pos);
  const m = before.match(TAIL_RE);
  return {before, partial: m ? m[0] : "", pos};
}
function refreshSuggest(){
  const box = el("suggest");
  if(!engine){ box.replaceChildren(); return; }
  const {before, partial} = tail();
  let items = [];
  if(partial.length >= 1){
    items = engine.generateBilingualSuggestions(partial)
      .filter(s => s.word.toLowerCase() !== partial.toLowerCase())
      .map(s => ({word: s.word, lang: s.language, ctx: false}));
  }else{
    const ctx = (before.match(WORD_RE) || []).slice(-5);
    if(ctx.length){
      engine.clearHistory();
      ctx.forEach(w => engine.addWordToHistory(w));
      items = engine.generateContextualSuggestions().map(w => ({word: w, lang: "KREYOL", ctx: true}));
    }
  }
  const frag = document.createDocumentFragment();
  items.slice(0, 5).forEach((it, i) => {
    const b = document.createElement("button");
    b.type = "button"; b.className = "sug" + (it.ctx ? " ctx" : "");
    b.dataset.lang = it.lang || "KREYOL"; b.dataset.word = it.word;
    b.appendChild(document.createTextNode(it.word));
    if(i === 0){ const k = document.createElement("span"); k.className = "k"; k.textContent = "Tab"; b.appendChild(k); }
    frag.appendChild(b);
  });
  if(items.length){
    const why = document.createElement("span");
    why.className = "why";
    why.textContent = items[0].ctx ? "mot suivant probable" : "complétion accentuée";
    frag.appendChild(why);
  }
  box.replaceChildren(frag);
}
function acceptSuggestion(word){
  const ta = el("edit"), pos = ta.selectionStart;
  const before = ta.value.slice(0, pos), after = ta.value.slice(pos);
  const m = before.match(TAIL_RE); const partial = m ? m[0] : "";
  const rest = after.replace(HEAD_RE, "");
  const cased = partial ? KreyolSimulatorEngine.applyCasingPattern(partial, word) : word;
  const head = before.slice(0, before.length - partial.length);
  const sep = (rest === "" || /^[\s.,;:!?)]/.test(rest)) ? "" : " ";
  const insert = cased + (rest === "" ? " " : sep);
  ta.value = head + insert + rest;
  const np = (head + insert).length;
  ta.setSelectionRange(np, np); ta.focus();
  const d = DATA[active]; if(!d) return;
  const s = ligne(d.id);
  s.corrected = ta.value;
  if(ta.value.trim()){ s.skipped = false; if(s.note === 1) s.note = 0; jalon(d.id, "edite"); }
  persist(d.id); paint(); renderList(); refreshSuggest();
}
el("suggest").addEventListener("click", e => {
  const b = e.target.closest(".sug"); if(b) acceptSuggestion(b.dataset.word);
});
el("edit").addEventListener("keyup", e => {
  if(e.key === "Tab") return;
  if(["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"].includes(e.key)) refreshSuggest();
});
el("edit").addEventListener("click", refreshSuggest);
el("edit").addEventListener("blur", () => setTimeout(() => {
  if(!document.activeElement || !document.activeElement.closest(".suggest")) el("suggest").replaceChildren();
}, 120));
el("edit").addEventListener("focus", refreshSuggest);

/* ---------- clavier ---------- */
document.addEventListener("keydown", e => {
  const typing = /^(INPUT|TEXTAREA)$/.test(document.activeElement.tagName);
  if(e.code === "Space" && e.ctrlKey){ e.preventDefault(); audio.currentTime = 0; audio.play().catch(() => {}); return; }
  if(e.code === "Space" && !typing){ e.preventDefault(); toggle(); return; }
  if(e.key === "Tab" && document.activeElement === el("edit")){
    const first = el("suggest").querySelector(".sug");
    if(first){ e.preventDefault(); acceptSuggestion(first.dataset.word); return; }
  }
  if(e.altKey && e.key === "Enter"){ e.preventDefault(); voter(); return; }
  if(e.altKey && e.key >= "1" && e.key <= "5"){ e.preventDefault(); noter(Number(e.key)); return; }
  if(e.key === "Enter" && (e.ctrlKey || e.metaKey)){ e.preventDefault(); nextTodo(); return; }
  if(e.altKey && e.key === "ArrowDown"){ e.preventDefault(); go(Math.min(active + 1, DATA.length - 1)); }
  if(e.altKey && e.key === "ArrowUp"){ e.preventDefault(); go(Math.max(active - 1, 0)); }
});

/* ---------- thème ---------- */
el("theme").addEventListener("click", () => {
  const cur = document.documentElement.getAttribute("data-theme");
  const dark = cur ? cur === "dark" : matchMedia("(prefers-color-scheme:dark)").matches;
  document.documentElement.setAttribute("data-theme", dark ? "light" : "dark");
});

/* ---------- export CSV (mode statique) ---------- */
/* Le statut n'est pas stocké : il se déduit de la note, seule source de
   vérité. Deux colonnes qui se contredisent, c'est une de trop. */
function statut(e){
  if(e.note === 5) return "human_validated";
  if(e.note === 1) return "unusable";
  if(e.note) return "human_reviewed";
  return e.corrected.trim() ? "draft" : "";
}
const iso = t => t ? new Date(t).toISOString() : "";
function csvCell(v){ v = (v == null ? "" : String(v)); return /[",\n\r]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v; }
function telechargerCsv(){
  const tete = ["segment_id", "whisper", "motif", "duree_ms", "corrected", "notes",
                "annotateur", "rating", "status", "vote", "ecoute_ms", "lectures",
                "ouvert_a", "ecoute_a", "edite_a", "note_a", "valide_a"];
  const lignes = [tete.join(",")];
  CATALOGUE.forEach(s => {
    const e = etat.get(s.c);
    if(!e || !(e.corrected.trim() || e.skipped || e.note)) return;
    const c = ecoute.get(s.c) || {ms: 0, lectures: 0};
    const j = e.jalons || {};
    lignes.push([s.c, s.t, s.m, s.d, e.corrected, e.notes, annotateur, e.note || "",
                 statut(e), e.vote ? "1" : "", Math.round(c.ms), c.lectures,
                 iso(j.ouvert), iso(j.ecoute), iso(j.edite), iso(j.note), iso(j.valide)]
                .map(csvCell).join(","));
  });
  if(lignes.length === 1){ note("Rien à exporter pour l'instant"); return; }
  const csv = lignes.join("\r\n") + "\r\n";
  try{
    const url = URL.createObjectURL(new Blob([csv], {type: "text/csv;charset=utf-8"}));
    const a = document.createElement("a");
    a.href = url; a.download = "corrections_gcf.csv";
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    note((lignes.length - 1) + " corrections exportées");
  }catch(e){
    navigator.clipboard?.writeText(csv).then(() => note("Téléchargement bloqué — CSV copié"));
  }
}

/* ---------- démarrage ---------- */
try{ annotateur = localStorage.getItem("gcf-annotateur") || ""; }catch(e){}
el("annotateur").value = annotateur;

async function chargerLot(){
  const params = new URLSearchParams(location.search);
  let noms = [];
  try{ noms = await (await fetch("data/lots.json", {cache: "no-store"})).json(); }catch(e){}
  const nom = params.get("lot") || noms[0];
  if(!nom){ el("trouves").textContent = "aucun lot publié"; return; }
  try{
    const j = await (await fetch("data/lot-" + nom + ".json")).json();
    CATALOGUE = j.segments || [];
    // Plusieurs lots : de quoi passer de l'un à l'autre par l'URL.
    el("mode").textContent = noms.length > 1
      ? "lot « " + nom + " » — les autres : " + noms.filter(n => n !== nom).map(n => "?lot=" + n).join(" ")
      : "Sauvegarde dans ce navigateur — exporte le CSV régulièrement.";
  }catch(e){ el("trouves").textContent = "lot « " + nom + " » introuvable"; }
}

(async () => {
  lireLocal();
  if(MODE === "statique"){
    await chargerLot();
    el("export").addEventListener("click", e => { e.preventDefault(); telechargerCsv(); });
  }else{
    el("mode").textContent = "Corrections partagées : elles repartent vers le dataset d'annotations.";
    window.addEventListener("beforeunload", () => { if(enAttente.size) navigator.sendBeacon?.(API + "/corrections",
      new Blob([JSON.stringify({annotateur, rows: [...enAttente.values()]})], {type: "application/json"})); });
  }
  majStats();
  await chercher(true);
  bootPredict();
  if(enAttente.size) envoyer();
})();
