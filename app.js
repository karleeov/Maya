// SafePlate — app logic.
// Open-source AI core: all-MiniLM-L6-v2 (Apache-2.0, open weights) running
// entirely in this browser via transformers.js + ONNX Runtime Web.
// Ingredients and allergen profiles are embedded on-device; cosine
// similarity catches hidden allergens that keyword search misses.

import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2';
import { RECIPES, ALLERGENS, SWAPS } from './recipes.js';

env.allowLocalModels = false; // always fetch open weights from the HF hub, then cache in-browser

// Thresholds tuned offline against the exact quantized ONNX build the
// browser downloads (see the DEV post write-up for the methodology).
const THRESH = {
  severe: { stop: 0.87, warn: 0.74 },
  mild:   { stop: 0.92, warn: 0.82 },
};
// Dairy-free / nut-free foods whose names collide with allergen terms
// ("peanut butter" is not dairy, "coconut milk" is not milk). Vetoed for
// both keyword and semantic matching.
const EXCEPTIONS = {
  'peanut butter': ['milk'], 'sunflower seed butter': ['milk'], 'vegan butter': ['milk'],
  'coconut milk': ['milk'], 'coconut cream': ['milk'], 'coconut yogurt': ['milk'],
  'coconut aminos': ['soy', 'gluten'], 'coconut oil': [],
  'oat milk': ['milk'], 'almond milk': ['milk'], 'soy milk': ['milk'], 'rice milk': ['milk'],
  'butter': ['peanut', 'tree-nut'],
  'tomato sauce': ['fish', 'soy', 'gluten', 'peanut', 'shellfish'],
  'eggplant': ['egg'], 'glutinous rice': ['gluten'], 'rice noodles': ['gluten', 'egg'],
  'fish sauce': ['gluten', 'soy', 'peanut', 'shellfish'], 'corn tortillas': ['gluten'],
};
const MODEL_ID = 'Xenova/all-MiniLM-L6-v2';
const DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];

const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

/* ── State ─────────────────────────────────────────────────────────── */

const savedProfile = JSON.parse(localStorage.getItem('safeplate-profile-v1') || 'null');
const state = {
  name: savedProfile?.name ?? 'Maya',
  allergens: new Set(savedProfile?.allergens ?? ['peanut', 'tree-nut', 'sesame']),
  severity: savedProfile?.severity ?? 'severe',
  filter: 'all',
  modelReady: false,
  extractor: null,
  ingVecs: new Map(),     // normalized ingredient -> Float32Array(384)
  allergenVecs: new Map() // allergen id -> { vecs: Float32Array[], terms: string[] }
};
let plan = JSON.parse(localStorage.getItem('safeplate-plan-v1') || 'null') || Object.fromEntries(DAYS.map((d) => [d, null]));

const norm = (s) => s.toLowerCase().replace(/[()]/g, ' ').replace(/[^a-z'\-\s]/g, ' ').replace(/\s+/g, ' ').trim();

/* ── Model loading ─────────────────────────────────────────────────── */

async function loadModel() {
  const fileProg = new Map();
  const onProgress = (p) => {
    if (p.status === 'progress' && p.file) {
      fileProg.set(p.file, p.progress || 0);
      const avg = [...fileProg.values()].reduce((a, b) => a + b, 0) / Math.max(fileProg.size, 1);
      $('#loadBar').style.width = `${Math.max(2, avg)}%`;
      const mb = p.loaded ? ` · ${(p.loaded / 1048576).toFixed(1)} MB` : '';
      $('#loadLabel').textContent = `Fetching open model ${Math.round(avg)}% — ${p.file}${mb}`;
    } else if (p.status === 'ready' || p.status === 'done') {
      $('#loadLabel').textContent = 'Preparing embeddings…';
    }
  };

  let lastErr;
  for (const host of [null, 'https://hf-mirror.com']) {
    try {
      if (host) { env.remoteHost = host; $('#loadLabel').textContent = 'Retrying via mirror…'; }
      state.extractor = await pipeline('feature-extraction', MODEL_ID, { progress_callback: onProgress });
      lastErr = null; break;
    } catch (e) { lastErr = e; console.warn(`Model load failed via ${host || 'huggingface.co'}`, e); }
  }
  if (lastErr) throw lastErr;

  $('#loadLabel').textContent = 'Reading every ingredient…';
  // Embed every unique ingredient once
  const uniqueIngs = [...new Set(RECIPES.flatMap((r) => r.ingredients.map(norm)))];
  for (let i = 0; i < uniqueIngs.length; i += 24) {
    const batch = uniqueIngs.slice(i, i + 24);
    const out = await state.extractor(batch, { pooling: 'mean', normalize: true });
    batch.forEach((ing, j) => state.ingVecs.set(ing, out.data.slice(j * 384, (j + 1) * 384)));
    $('#loadBar').style.width = `${92 + (8 * (i + batch.length)) / uniqueIngs.length}%`;
  }
  // Embed each allergen's vocabulary
  for (const a of ALLERGENS) {
    const texts = [a.desc, ...a.terms];
    const out = await state.extractor(texts, { pooling: 'mean', normalize: true });
    state.allergenVecs.set(a.id, {
      vecs: texts.map((_, j) => out.data.slice(j * 384, (j + 1) * 384)),
      terms: [a.label.toLowerCase(), ...a.terms],
    });
  }
  state.modelReady = true;
}

/* ── Scoring ───────────────────────────────────────────────────────── */

const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };

const escRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

function keywordHit(ing, allergen) {
  if (EXCEPTIONS[ing]?.includes(allergen.id)) return null;
  for (const t of allergen.terms) {
    // word-boundary phrase match with plural tolerance — catches
    // "eggs" but not "eggplant", "milk" but not "buttermilk"
    if (new RegExp(`\\b${escRe(t)}(e?s)?\\b`).test(ing)) return t;
  }
  return null;
}

function scoreIngredient(rawIng) {
  // Returns best match { allergen, score, term } or null
  const ing = norm(rawIng);
  let best = null;
  for (const a of ALLERGENS) {
    if (!state.allergens.has(a.id)) continue;
    const kw = keywordHit(ing, a);
    let score = kw ? 1 : 0, term = kw || '';
    if (!kw && state.modelReady && !EXCEPTIONS[ing]?.includes(a.id)) {
      const vec = state.ingVecs.get(ing) || null;
      const av = state.allergenVecs.get(a.id);
      if (vec && av) {
        av.vecs.forEach((tv, j) => {
          const s = dot(vec, tv);
          if (s > score) { score = s; term = av.terms[j]; }
        });
      }
    }
    if (!best || score > best.score) best = { allergen: a, score, term };
  }
  return best && best.score >= THRESH[state.severity].warn ? best : null;
}

function scoreRecipe(recipe) {
  const flags = [];
  for (const ing of recipe.ingredients) {
    const hit = scoreIngredient(ing);
    if (hit) {
      flags.push({ ingredient: ing, ...hit, swap: findSwap(ing, hit.term, hit.score === 1) });
    }
  }
  const t = THRESH[state.severity];
  const stops = flags.filter((f) => f.score >= t.stop).length;
  const warns = flags.length - stops;
  const verdict = stops > 0 || (state.severity === 'severe' && warns >= 2) ? 'stop'
    : flags.length > 0 ? 'warn' : 'safe';
  return { recipe, flags, verdict };
}

function findSwap(ing, term, isKeyword) {
  // Prefer swaps that match the ingredient itself; only fall back to the
  // matched term for direct (keyword) hits — semantic cousins shouldn't
  // inherit random swaps ("coconut milk" ≠ almond-truffle swap).
  const low = ing.toLowerCase();
  for (const s of SWAPS) if (s.match.some((m) => low.includes(m))) return s;
  if (isKeyword && term) {
    const lt = term.toLowerCase();
    for (const s of SWAPS) if (s.match.some((m) => lt.includes(m))) return s;
  }
  return null;
}

/* ── Rendering ─────────────────────────────────────────────────────── */

function friendName() { return state.name.trim() || 'your friend'; }

function renderAll() {
  renderSummary();
  renderTabs();
  renderGrid();
  renderPlanner();
  document.querySelectorAll('.friend-inline').forEach((el) => (el.textContent = friendName()));
  $('#footFriend').textContent = friendName();
}

function verdictLabel(v, n) {
  const name = friendName();
  if (v === 'safe') return `All clear for ${name}`;
  if (v === 'warn') return `${n} ingredient${n > 1 ? 's' : ''} to double-check`;
  return `Not safe for ${name}`;
}

function renderSummary() {
  const scored = RECIPES.map(scoreRecipe);
  const c = { safe: 0, warn: 0, stop: 0 };
  scored.forEach((s) => c[s.verdict]++);
  $('#summaryStrip').innerHTML = `
    <span class="sum-pill safe"><b>${c.safe}</b> safe plates</span>
    <span class="sum-pill warn"><b>${c.warn}</b> worth a closer look</span>
    <span class="sum-pill stop"><b>${c.stop}</b> off the table</span>
    <span class="sum-pill">cooked for <b>&nbsp;${esc(friendName())}</b></span>`;
}

function renderTabs() {
  const scored = RECIPES.map(scoreRecipe);
  const c = { all: scored.length, safe: 0, warn: 0, stop: 0 };
  scored.forEach((s) => c[s.verdict]++);
  const tabs = [['all', 'All recipes'], ['safe', 'Safe'], ['warn', 'Double-check'], ['stop', 'Avoid']];
  $('#filterTabs').innerHTML = tabs.map(([k, l]) =>
    `<button class="tab ${state.filter === k ? 'active' : ''}" data-filter="${k}">${l}<span class="n">${c[k]}</span></button>`).join('');
  document.querySelectorAll('.tab').forEach((b) =>
    b.addEventListener('click', () => { state.filter = b.dataset.filter; renderTabs(); renderGrid(); }));
}

function renderGrid() {
  const scored = RECIPES.map(scoreRecipe).filter((s) => state.filter === 'all' || s.verdict === state.filter);
  const grid = $('#recipeGrid');
  if (!scored.length) {
    grid.innerHTML = `<div class="no-results"><span class="big">Nothing here — yet.</span>Try another filter, or loosen the allergy profile in the sidebar.</div>`;
    return;
  }
  grid.innerHTML = scored.map((s, i) => {
    const r = s.recipe;
    const flags = s.flags.map((f) => {
      const lvl = f.score >= THRESH[state.severity].stop ? 'stop' : 'warn';
      const pct = state.modelReady ? `<span class="pct">${Math.round(f.score * 100)}%</span>` : '';
      const why = f.score === 1
        ? `contains <strong>${esc(f.allergen.label.toLowerCase())}</strong>`
        : `semantically close to <strong>${esc(f.allergen.label.toLowerCase())}</strong> (matched “${esc(f.term)}”)`;
      const swap = f.swap ? `<div class="f-swap">↳ swap: <b>${esc(f.swap.swap)}</b> — ${esc(f.swap.note)}</div>` : '';
      return `<div class="flag ${lvl}">${pct}<span class="f-ing">${esc(f.ingredient)}</span><div class="f-why">${why}</div>${swap}</div>`;
    }).join('');
    const addable = s.verdict === 'safe';
    const inPlan = Object.values(plan).includes(r.id);
    return `
    <article class="card" style="animation-delay:${(i % 9) * 55}ms">
      <div class="card-top">
        <div class="plate" aria-hidden="true">${r.emoji}</div>
        <div>
          <h3 class="card-title">${esc(r.name)}</h3>
          <div class="card-meta">${esc(r.cuisine)} · ${r.minutes} min · ${r.tags.join(' · ')}</div>
        </div>
      </div>
      <div class="verdict-band ${s.verdict}"><span class="vdot"></span>${verdictLabel(s.verdict, s.flags.length)}</div>
      <p class="card-blurb">${esc(r.blurb)}</p>
      ${flags ? `<div class="flag-list">${flags}</div>` : ''}
      <div class="card-foot">
        <span class="ing-count">${r.ingredients.length} ingredients scanned</span>
        <button class="btn-add ${inPlan ? 'added' : ''}" data-add="${r.id}" ${addable && !inPlan ? '' : (inPlan ? '' : 'disabled')}>
          ${inPlan ? 'In the week ✓' : 'Add to week'}
        </button>
      </div>
    </article>`;
  }).join('');
  grid.querySelectorAll('[data-add]').forEach((b) =>
    b.addEventListener('click', () => addToPlan(b.dataset.add)));
}

/* ── Scanner ───────────────────────────────────────────────────────── */

function runScan() {
  const raw = $('#scanInput').value;
  const items = raw.split(/[\n,;]+/).map((s) => s.trim()).filter(Boolean);
  const box = $('#scanResults');
  if (!items.length) return;
  if (!state.allergens.size) {
    box.innerHTML = `<p class="scan-note">Pick at least one allergy in the sidebar first — I need to know what to look for.</p>`;
    return;
  }
  const rows = [];
  let clear = 0;
  for (const item of items) {
    const hit = scoreIngredient(item);
    if (!hit) { clear++; continue; }
    const lvl = hit.score >= THRESH[state.severity].stop ? 'stop' : 'warn';
    const pct = state.modelReady ? `<span class="pct">${Math.round(hit.score * 100)}%</span>` : '';
    const why = hit.score === 1
      ? `direct match for <strong>${esc(hit.allergen.label.toLowerCase())}</strong>`
      : `semantically close to <strong>${esc(hit.allergen.label.toLowerCase())}</strong> (nearest: “${esc(hit.term)}”)`;
    const swap = findSwap(item, hit.term, hit.score === 1);
    rows.push(`<div class="flag ${lvl}">${pct}<span class="f-ing">${esc(item)}</span><div class="f-why">${why}</div>${swap ? `<div class="f-swap">↳ swap: <b>${esc(swap.swap)}</b> — ${esc(swap.note)}</div>` : ''}</div>`);
  }
  const head = rows.length
    ? `<p class="scan-note">⚠ ${rows.length} of ${items.length} ingredient${items.length > 1 ? 's' : ''} flagged for ${esc(friendName())}${clear ? ` — ${clear} looked clear` : ''}.${state.modelReady ? '' : ' (keyword mode — AI still loading)'}</p>`
    : `<p class="scan-note">✅ Nothing in that list trips ${esc(friendName())}'s profile. Still — when an allergy is severe, trust labels over any app, this one included.</p>`;
  box.innerHTML = head + rows.join('');
}

/* ── Planner ───────────────────────────────────────────────────────── */

function addToPlan(id) {
  const day = DAYS.find((d) => !plan[d]);
  if (!day) { toast('The week is full — tap a day to clear it.'); return; }
  plan[day] = id;
  localStorage.setItem('safeplate-plan-v1', JSON.stringify(plan));
  const r = RECIPES.find((x) => x.id === id);
  toast(`${r.emoji} ${r.name} added to ${day}`);
  renderGrid(); renderPlanner();
}

function renderPlanner() {
  $('#weekGrid').innerHTML = DAYS.map((d) => {
    const r = plan[d] ? RECIPES.find((x) => x.id === plan[d]) : null;
    return `<div class="day"><div class="day-name">${d.slice(0, 3)}</div>
      ${r
        ? `<div class="day-slot filled" data-day="${d}" title="Tap to remove"><span class="slot-name">${r.emoji} ${esc(r.name)}</span></div>`
        : `<div class="day-slot">empty plate</div>`}
    </div>`;
  }).join('');
  document.querySelectorAll('.day-slot.filled').forEach((el) =>
    el.addEventListener('click', () => {
      plan[el.dataset.day] = null;
      localStorage.setItem('safeplate-plan-v1', JSON.stringify(plan));
      renderGrid(); renderPlanner();
    }));
}

let toastTimer;
function toast(msg) {
  let t = $('#toast');
  if (!t) { t = document.createElement('div'); t.id = 'toast'; document.body.appendChild(t); }
  t.textContent = msg; t.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove('show'), 2400);
}

/* ── Sidebar wiring ────────────────────────────────────────────────── */

function renderChips() {
  $('#allergenChips').innerHTML = ALLERGENS.map((a) =>
    `<button class="chip ${state.allergens.has(a.id) ? 'on' : ''}" data-a="${a.id}" aria-pressed="${state.allergens.has(a.id)}">${a.emoji} ${a.label}</button>`).join('');
  document.querySelectorAll('.chip').forEach((c) =>
    c.addEventListener('click', () => {
      const id = c.dataset.a;
      state.allergens.has(id) ? state.allergens.delete(id) : state.allergens.add(id);
      saveProfile(); renderChips(); renderAll();
    }));
}

function renderSeverity() {
  $('#sevMild').classList.toggle('active', state.severity === 'mild');
  $('#sevSevere').classList.toggle('active', state.severity === 'severe');
}

function saveProfile() {
  localStorage.setItem('safeplate-profile-v1', JSON.stringify({
    name: state.name, allergens: [...state.allergens], severity: state.severity,
  }));
}

function wireSidebar() {
  $('#friendName').value = state.name;
  $('#friendName').addEventListener('input', (e) => { state.name = e.target.value; saveProfile(); renderSummary(); renderGrid(); document.querySelectorAll('.friend-inline').forEach((el) => (el.textContent = friendName())); $('#footFriend').textContent = friendName(); });
  $('#sevMild').addEventListener('click', () => { state.severity = 'mild'; saveProfile(); renderSeverity(); renderAll(); });
  $('#sevSevere').addEventListener('click', () => { state.severity = 'severe'; saveProfile(); renderSeverity(); renderAll(); });
  renderChips(); renderSeverity();
}

/* ── Network badge ─────────────────────────────────────────────────── */

function wireNet() {
  const update = () => {
    const on = navigator.onLine;
    $('#netDot').className = `dot ${on ? 'dot-ok' : 'dot-off'}`;
    $('#netStatus').textContent = on
      ? (state.modelReady ? 'online — but nothing leaves this tab' : 'online')
      : 'offline — still fully working';
  };
  window.addEventListener('online', update);
  window.addEventListener('offline', update);
  update();
}

/* ── Boot ──────────────────────────────────────────────────────────── */

wireSidebar();
wireNet();
renderAll();
$('#scanBtn').addEventListener('click', runScan);
$('#scanInput').addEventListener('keydown', (e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) runScan(); });

// Plan my week: fill empty days with a varied set of safe picks
$('#planFill').addEventListener('click', () => {
  const safe = RECIPES.filter((r) => scoreRecipe(r).verdict === 'safe' && !Object.values(plan).includes(r.id));
  const cuisines = new Set();
  for (const d of DAYS) {
    if (plan[d] || !safe.length) continue;
    const pick = safe.find((r) => !cuisines.has(r.cuisine)) || safe[0];
    plan[d] = pick.id; cuisines.add(pick.cuisine);
    safe.splice(safe.indexOf(pick), 1);
  }
  localStorage.setItem('safeplate-plan-v1', JSON.stringify(plan));
  renderGrid(); renderPlanner();
  toast('Week planned with safe plates ✨');
});

// Shareable deep links: ?friend=Ana&allergens=peanut,egg&scan=oat milk, nutella
(function deepLink() {
  const q = new URLSearchParams(location.search);
  if (q.get('friend')) { state.name = q.get('friend').slice(0, 24); $('#friendName').value = state.name; }
  if (q.get('allergens')) {
    const ids = q.get('allergens').split(',').map((s) => s.trim()).filter((id) => ALLERGENS.some((a) => a.id === id));
    if (ids.length) { state.allergens = new Set(ids); renderChips(); }
  }
  if (q.get('friend') || q.get('allergens')) { saveProfile(); renderAll(); }
  if (q.get('scan')) {
    $('#scanInput').value = q.get('scan');
    setTimeout(() => { document.querySelector('.scanner').scrollIntoView({ behavior: 'smooth' }); runScan(); }, 400);
  }
  if (q.get('plan')) setTimeout(() => $('#planFill').click(), 500);
})();

loadModel()
  .then(() => {
    $('#modelDot').className = 'dot dot-ok';
    $('#modelStatus').textContent = 'AI ready · all-MiniLM-L6-v2 · on-device';
    $('#loadBar').style.width = '100%';
    $('#loadLabel').textContent = 'Table is set.';
    setTimeout(() => $('#loader').classList.add('done'), 450);
    wireNet();
    renderAll(); // rescore with embeddings now live
    if ($('#scanInput').value.trim()) runScan(); // upgrade a keyword-mode scan to full AI
  })
  .catch((err) => {
    console.error('Model failed, falling back to keyword mode:', err);
    $('#modelDot').className = 'dot dot-err';
    $('#modelStatus').textContent = 'AI unavailable — keyword mode';
    $('#loadLabel').textContent = 'Could not reach the model hub — continuing with keyword matching.';
    $('#loadBar').style.width = '100%';
    setTimeout(() => $('#loader').classList.add('done'), 900);
    renderAll();
  });
