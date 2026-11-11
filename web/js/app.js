"use strict";

/* AI Product Authentication Using Logo & Packaging Recognition — frontend
   talks to the JSON API exposed by web/server.py */

const API = "";

const SIGNAL_META = {
  template:    { label: "Packaging layout match",        dot: "#22d3ee" },
  orb:         { label: "Logo / print feature match",    dot: "#a78bfa" },
  color:       { label: "Colour distribution",           dot: "#34d399" },
  edge_layout: { label: "Print edge structure",          dot: "#60a5fa" },
  sharpness:   { label: "Print sharpness",               dot: "#fbbf24" },
  noise:       { label: "Print surface noise",           dot: "#f87171" },
  qr:          { label: "Security QR code",              dot: "#2dd4bf" },
  siamese:     { label: "Learned visual signature",      dot: "#f472b6" },
};

const EVIDENCE = [
  { key: "ref",     label: "Original (reference)",   dot: "#34d399" },
  { key: "query",   label: "Test image (query)",     dot: "#fbbf24" },
  { key: "overlay", label: "Suspicious-region overlay", dot: "#ef4444" },
];

const QR_STATUS = {
  qr_match:     { dot: "#10b981", text: "QR matches reference" },
  missing_qr:   { dot: "#f59e0b", text: "QR missing from the query" },
  unexpected_qr:{ dot: "#f59e0b", text: "Unexpected QR on the query" },
  qr_mismatch:  { dot: "#ef4444", text: "QR content differs from the reference" },
  skipped:      { dot: "#8b96ad", text: "QR check skipped" },
};

const VERDICT_COLOR = {
  Authentic: "#34d399",
  "Likely Authentic": "#60a5fa",
  Suspicious: "#fbbf24",
  Counterfeit: "#f87171",
};

const VIEWS = {
  verify:  { title: "Verify a product",    sub: "Choose the product and upload or capture a photo to authenticate it." },
  library: { title: "Product library",     sub: "Manage the authenticated reference imagery used for verification." },
  system:  { title: "System status",       sub: "Runtime configuration for the authentication engine." },
};

const state = {
  products: [],
  weights: {},
  info: {},
  queryFile: null,
  queryUrl: null,
  libFiles: [],
  stream: null,
  busy: false,
};

/* ── dom helpers ─────────────────────────────────────────── */
const $ = (id) => document.getElementById(id);
const el = (tag, cls, htmlText) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (htmlText != null) n.textContent = htmlText;
  return n;
};
const clear = (node) => { while (node.firstChild) node.removeChild(node.firstChild); };

function chip(dot, text, style) {
  const c = el("span", "chip");
  const d = el("span", "chip__dot");
  d.style.background = dot || "var(--dim)";
  c.append(d, document.createTextNode(text));
  if (style) c.setAttribute("style", style);
  return c;
}

/* ── toasts + progress bar ───────────────────────────────── */
function toast(text, kind) {
  const t = el("div", "toast" + (kind ? " toast--" + kind : ""));
  t.append(el("span", "toast__bar"), el("span", null, text));
  $("toasts").append(t);
  setTimeout(() => {
    t.style.transition = "opacity .3s ease";
    t.style.opacity = "0";
    setTimeout(() => t.remove(), 320);
  }, kind === "err" ? 6500 : 3800);
}

function progress(on) {
  const bar = $("loadingBar");
  bar.classList.toggle("is-done", !on);
  if (on) { bar.classList.remove("is-done"); void bar.offsetWidth; bar.classList.add("is-on"); }
}

/* ── networking ──────────────────────────────────────────── */
async function api(path, options) {
  progress(true);
  try {
    const res = await fetch(API + path, options);
    const text = await res.text();
    let data = null;
    try { data = text ? JSON.parse(text) : null; } catch { data = { error: text }; }
    if (!res.ok) throw new Error((data && data.error) || `Request failed (${res.status})`);
    return data;
  } finally {
    progress(false);
  }
}

/* ── navigation ──────────────────────────────────────────── */
function go(view) {
  for (const key of Object.keys(VIEWS)) {
    $("view-" + key).hidden = key !== view;
  }
  document.querySelectorAll(".nav__item").forEach((b) => {
    if (b.dataset.nav === view) b.setAttribute("aria-current", "page");
    else b.removeAttribute("aria-current");
  });
  $("viewTitle").textContent = VIEWS[view].title;
  $("viewSub").textContent = VIEWS[view].sub;
  $("app").classList.remove("nav-open");
  $("burger").setAttribute("aria-expanded", "false");
  window.scrollTo({ top: 0, behavior: "smooth" });
  if (view === "library") loadProducts();
}

/* ── status / chips ──────────────────────────────────────── */
function renderStatus(info) {
  state.info = info;
  state.weights = info.weights || {};

  const hasModel = !!info.model_loaded;
  $("modelPulse").className = "pulse " + (hasModel ? "pulse--ok" : "pulse--warn");
  $("sideDevice").textContent = info.device || "—";
  $("sideThreshold").textContent = info.threshold != null ? info.threshold.toFixed(0) : "—";
  $("sideModelText").textContent = hasModel
    ? `Siamese model active (${info.model_mtime || "checkpoint"}).`
    : "No Siamese checkpoint found — running classical signals only.";
  $("sideCheckpoint").textContent = info.model_path || "";

  const heroChips = clear($("heroChips"));
  heroChips.append(
    chip(hasModel ? "#10b981" : "#f59e0b", `Siamese model ${hasModel ? "active" : "not trained"}`),
    chip("#22d3ee", `Device: ${(info.device || "").toUpperCase()}`),
    chip("#60a5fa", `Verdict threshold: ${info.threshold?.toFixed(0) ?? "—"} → Original`),
    chip("#a78bfa", "Signals: classical + learned"),
  );

  const top = $("topChips");
  top.classList.remove("hidden");
  $("topChipText").textContent = `${info.products ?? 0} product${info.products === 1 ? "" : "s"} · ${info.images ?? 0} ref image${info.images === 1 ? "" : "s"}`;

  $("ringThreshold").textContent = info.threshold != null ? `${info.threshold.toFixed(0)} / 100` : "—";

  renderSystem(info);
}

function renderSystem(info) {
  const tiles = clear($("systemTiles"));
  const tile = (label, value, unit, note) => {
    const c = el("div", "card stat");
    c.append(el("div", "stat__label", label));
    const v = el("div", "stat__value", value);
    if (unit) { const s = el("small", null, unit); v.append(s); }
    c.append(v);
    if (note) c.append(el("div", "stat__note", note));
    return c;
  };
  tiles.append(
    tile("Siamese model", info.model_loaded ? "Loaded" : "Missing", "", info.model_path || ""),
    tile("Device", (info.device || "—").toUpperCase()),
    tile("Products", String(info.products ?? 0), "", `${info.images ?? 0} reference images`),
  );

  const rows = clear($("weightTable"));
  for (const [key, w] of Object.entries(state.weights)) {
    const meta = SIGNAL_META[key] || { label: key, dot: "#8b96ad" };
    const tr = el("tr");
    const dotCell = el("td");
    dotCell.append(chip(meta.dot, ""));
    const wCell = el("td", null, `${(w * 100).toFixed(0)}%`);
    wCell.style.textAlign = "right";
    tr.append(dotCell, el("td", null, meta.label), wCell);
    rows.append(tr);
  }

  const a = clear($("configA")), b = clear($("configB"));
  const kv = (host, k, v) => {
    const row = el("div", "kv");
    row.append(el("span", null, k), el("b", null, String(v)));
    host.append(row);
  };
  const cfg = info.config || {};
  kv(a, "Image size", cfg.IMAGE_SIZE);
  kv(a, "Siamese input", cfg.SIAMESE_SIZE);
  kv(a, "Embedding dim", cfg.EMBED_DIM);
  kv(a, "ORB features", cfg.ORB_FEATURES);
  kv(b, "ORB ratio", cfg.ORB_RATIO_THRESHOLD);
  kv(b, "Heatmap grid", (cfg.GRID || []).join(" × "));
  kv(b, "Siamese margin", cfg.SIAMESE_MARGIN);
  kv(b, "Binary threshold", cfg.BINARY_THRESHOLD);
}

/* ── products ────────────────────────────────────────────── */
async function loadProducts() {
  let data;
  try {
    data = await api("/api/products");
  } catch (e) {
    toast(e.message, "err");
    return;
  }
  state.products = data.products || [];

  const select = $("productSelect");
  const previous = select.value;
  clear(select);
  if (!state.products.length) {
    select.append(new Option("— no products —", ""));
    select.disabled = true;
  } else {
    select.disabled = false;
    for (const p of state.products) select.append(new Option(p.name, p.name));
    if (state.products.some((p) => p.name === previous)) select.value = previous;
  }
  $("productHint").textContent = state.products.length
    ? `${state.products.length} product${state.products.length === 1 ? "" : "s"} · ${state.products.reduce((n, p) => n + p.count, 0)} reference images`
    : "—";
  $("noProducts").hidden = state.products.length > 0;
  syncRun();
  renderLibrary();
}

function renderLibrary() {
  const grid = clear($("libraryGrid"));
  $("libraryEmpty").hidden = state.products.length > 0;
  $("libCountProducts").textContent = String(state.products.length);
  $("libCountImages").textContent = String(state.products.reduce((n, p) => n + p.count, 0));
  $("libPath").textContent = state.info.genuine_dir || "—";

  for (const p of state.products) {
    const card = el("div", "pcard");
    const head = el("div", "pcard__head");
    head.append(chip("#60a5fa", ""), el("span", "pcard__name", p.name));
    const count = el("div", "pcard__count");
    count.append(el("b", null, String(p.count)), document.createTextNode(" reference image(s)"));
    const thumbs = el("div", "pcard__thumbs");
    for (const ref of p.references || []) {
      const img = new Image();
      img.src = `/api/reference?product=${encodeURIComponent(p.name)}&file=${encodeURIComponent(ref)}`;
      img.alt = `${p.name} — ${ref}`;
      img.loading = "lazy";
      img.addEventListener("click", () => window.open(img.src, "_blank", "noopener"));
      thumbs.append(img);
    }
    const foot = el("div", "pcard__foot");
    const use = el("button", "btn btn--sm", "Use for verification");
    use.addEventListener("click", () => {
      $("productSelect").value = p.name;
      go("verify");
      syncRun();
    });
    const del = el("button", "btn btn--sm btn--ghost", "Delete");
    del.addEventListener("click", () => removeProduct(p.name));
    foot.append(use, del);

    card.append(head, count);
    if ((p.references || []).length) card.append(thumbs);
    card.append(foot);
    grid.append(card);
  }
}

async function removeProduct(name) {
  if (!confirm(`Delete "${name}" and all of its reference images?`)) return;
  try {
    await api(`/api/products/${encodeURIComponent(name)}`, { method: "DELETE" });
    toast(`Deleted "${name}".`, "ok");
    await Promise.all([refreshStatus(), loadProducts()]);
  } catch (e) {
    toast(e.message, "err");
  }
}

async function refreshStatus() {
  try {
    renderStatus(await api("/api/info"));
  } catch (e) {
    toast(e.message, "err");
  }
}

/* ── upload pane ─────────────────────────────────────────── */
function setQuery(file) {
  if (!file) return;
  if (state.queryUrl) URL.revokeObjectURL(state.queryUrl);
  state.queryFile = file;
  state.queryUrl = URL.createObjectURL(file);

  const thumbs = clear($("queryThumbs"));
  const t = el("div", "thumb");
  const img = new Image();
  img.src = state.queryUrl;
  img.alt = "Selected query preview";
  t.append(img);
  const x = el("button", "thumb__x", "×");
  x.type = "button";
  x.setAttribute("aria-label", "Remove selected image");
  x.addEventListener("click", () => {
    $("queryFile").value = "";
    setQuery(null);
  });
  t.append(x);
  thumbs.append(t);
  syncRun();
}

function syncRun() {
  const ok = !!state.queryFile && !!$("productSelect").value && !state.busy;
  $("runBtn").disabled = !ok;
}

/* ── camera ──────────────────────────────────────────────── */
async function startCamera() {
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" }, audio: false });
    $("camVideo").srcObject = state.stream;
    await $("camVideo").play();
    $("camStart").textContent = "Restart camera";
    $("camSnap").disabled = false;
  } catch {
    toast("Camera unavailable — use the upload tab instead.", "err");
  }
}

function stopCamera() {
  if (state.stream) state.stream.getTracks().forEach((t) => t.stop());
  state.stream = null;
  $("camSnap").disabled = true;
  $("camStart").textContent = "Start camera";
}

function capture() {
  const video = $("camVideo");
  if (!video.videoWidth) return;
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);
  canvas.toBlob((blob) => {
    if (blob) {
      setQuery(new File([blob], `capture-${Date.now()}.jpg`, { type: "image/jpeg" }));
      toast("Frame captured.", "ok");
    }
  }, "image/jpeg", 0.92);
}

/* ── verification ────────────────────────────────────────── */
function fillColor(v) {
  if (v >= 0.7) return "#10b981";
  if (v >= 0.4) return "#f59e0b";
  return "#ef4444";
}

function renderRing(score) {
  const size = 168, stroke = 14, r = (size - stroke) / 2, circ = 2 * Math.PI * r;
  const wrap = $("scoreRing");
  clear(wrap);
  wrap.style.width = size + "px";
  wrap.style.height = size + "px";

  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("width", size);
  svg.setAttribute("height", size);
  const mk = (cls) => {
    const c = document.createElementNS(ns, "circle");
    c.setAttribute("cx", size / 2); c.setAttribute("cy", size / 2); c.setAttribute("r", r);
    c.setAttribute("fill", "none"); c.setAttribute("stroke-width", stroke);
    c.setAttribute("class", cls);
    return c;
  };
  const track = mk("ring__track");
  const fill = mk("ring__fill");
  fill.setAttribute("stroke-linecap", "round");
  fill.setAttribute("stroke", fillColor(score / 100));
  fill.style.strokeDasharray = circ;
  fill.style.strokeDashoffset = circ;
  svg.append(track, fill);

  const label = el("div", "ring__label");
  const b = el("b", null, score.toFixed(1));
  b.style.color = fillColor(score / 100);
  label.append(b, el("span", null, "out of 100"));
  wrap.append(svg, label);

  requestAnimationFrame(() => { fill.style.strokeDashoffset = circ * (1 - Math.min(1, Math.max(0, score / 100))); });
}

function renderResults(data) {
  const res = data.result;
  const score = res.score;
  const isOrig = res.binary === "Original";
  const verdict = res.verdict;
  const confidence = isOrig ? score : Math.max(0, 100 - score);

  const banner = clear($("banner"));
  const cls = isOrig ? "banner--ok" : "banner--fake";
  banner.className = "banner " + cls;

  const logo = el("div", "banner__logo", isOrig ? "OK" : "!!");
  const text = el("div", "banner__text");
  text.append(
    el("div", "banner__label", isOrig ? "ORIGINAL" : "FAKE"),
  );
  const sc = el("div", "banner__score");
  sc.append(document.createTextNode(score.toFixed(1)), el("small", null, " / 100"));
  text.append(sc, el("div", "banner__sub", isOrig
    ? "This product matches the authenticated reference for this item."
    : "This product deviates from the authenticated reference — counterfeiting is suspected."));

  const badge = (value, caption, color) => {
    const b = el("div", "banner__badge");
    const strong = el("b", null, value);
    if (color) strong.style.color = color;
    b.append(strong, el("span", null, caption));
    return b;
  };

  banner.append(logo, text,
    badge(verdict, "Detailed verdict", VERDICT_COLOR[verdict]),
    badge(confidence.toFixed(1) + "%", "Confidence"));

  // alignment + QR chips
  const m = res.metrics || {};
  const chips = clear($("alignChips"));
  const align = m.alignment_score ?? 0;
  chips.append(chip(align > 0.2 ? "#10b981" : "#f59e0b",
    `Aligned to reference · ${m.alignment_matches ?? 0} keypoint matches (confidence ${align.toFixed(2)})`));
  const qr = QR_STATUS[m.qr_status];
  if (qr) chips.append(chip(qr.dot, qr.text));

  // stats
  renderRing(score);
  $("statRef").textContent = data.reference || "—";
  $("statConf").textContent = confidence.toFixed(1) + "%";
  $("statCount").textContent = String(Object.keys(res.scores || {}).length);

  // evidence
  const grid = clear($("evidence"));
  for (const item of EVIDENCE) {
    const card = el("div", "evidence");
    const lbl = el("div", "lbl");
    lbl.append(el("span", "chip__dot"), document.createTextNode(item.label));
    lbl.firstChild.style.background = item.dot;
    const frame = el("div", "evidence__frame is-loading");
    const img = new Image();
    img.alt = item.label;
    img.src = item.key === "ref" ? (data.reference_url || "") : data.images?.[item.key] || "";
    img.addEventListener("load", () => frame.classList.remove("is-loading"));
    img.addEventListener("error", () => {
      frame.classList.remove("is-loading");
      frame.append(el("div", "empty", "Image unavailable"));
    });
    frame.append(img);
    card.append(lbl, frame);
    grid.append(card);
  }

  // signals
  const sigs = clear($("signals"));
  const scores = res.scores || {};
  for (const [key, value] of Object.entries(scores)) {
    const meta = SIGNAL_META[key] || { label: key, dot: "#8b96ad" };
    const row = el("div", "sig");
    row.style.setProperty("--dot", meta.dot);
    row.style.setProperty("--fill", fillColor(value));

    const label = el("div", "sig__label");
    const dot = el("span", "sig__dot");
    label.append(dot, document.createTextNode(meta.label));

    const track = el("div", "sig__track");
    const fill = el("div", "sig__fill");
    track.append(fill);

    const w = el("div", "sig__w", `${((state.weights[key] || 0) * 100).toFixed(0)}%`);
    const pct = el("div", "sig__pct", String(Math.round(value * 100)));
    pct.style.color = fillColor(value);

    row.append(label, track, w, pct);
    sigs.append(row);
    requestAnimationFrame(() => { fill.style.width = `${Math.max(0, Math.min(1, value)) * 100}%`; });
  }

  // reasons
  const rsn = data.reasons || [];
  $("reasonsSec").hidden = rsn.length === 0;
  const rc = clear($("reasons"));
  for (const t of rsn) rc.append(chip("#f59e0b", t));

  $("results").hidden = false;
  $("results").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function runVerification() {
  if (state.busy || !state.queryFile) return;
  state.busy = true;
  const btn = $("runBtn");
  const original = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = "";
  btn.append(el("span", "spinner"), document.createTextNode("Analysing packaging…"));
  $("results").hidden = true;

  const body = new FormData();
  body.append("product", $("productSelect").value);
  body.append("image", state.queryFile, state.queryFile.name || "capture.jpg");

  try {
    renderResults(await api("/api/verify", { method: "POST", body }));
    toast("Verification complete.", "ok");
  } catch (e) {
    toast(e.message, "err");
  } finally {
    state.busy = false;
    btn.innerHTML = original;
    syncRun();
  }
}

/* ── library upload ──────────────────────────────────────── */
function renderLibThumbs() {
  const wrap = clear($("libThumbs"));
  for (const f of state.libFiles) {
    const t = el("div", "thumb");
    const img = new Image();
    img.src = URL.createObjectURL(f);
    img.alt = f.name;
    const cap = el("span", "thumb__name", f.name);
    const x = el("button", "thumb__x", "×");
    x.type = "button";
    x.setAttribute("aria-label", `Remove ${f.name}`);
    x.addEventListener("click", () => {
      state.libFiles = state.libFiles.filter((o) => o !== f);
      $("libFiles").value = "";
      renderLibThumbs();
    });
    t.append(img, cap, x);
    wrap.append(t);
  }
  $("libAdd").disabled = !($("libName").value.trim() && state.libFiles.length);
}

async function addReferences() {
  const name = $("libName").value.trim();
  if (!name || !state.libFiles.length) return;
  const btn = $("libAdd");
  btn.disabled = true;
  const original = btn.textContent;
  btn.textContent = "Uploading…";
  const body = new FormData();
  body.append("product", name);
  for (const f of state.libFiles) body.append("files", f, f.name);
  try {
    const res = await api("/api/products", { method: "POST", body });
    toast(res.message || "References added.", "ok");
    state.libFiles = [];
    $("libName").value = "";
    $("libFiles").value = "";
    renderLibThumbs();
    await Promise.all([refreshStatus(), loadProducts()]);
  } catch (e) {
    toast(e.message, "err");
    btn.disabled = false;
    btn.textContent = original;
  }
}

/* ── wiring ──────────────────────────────────────────────── */
function init() {
  document.querySelectorAll(".nav__item").forEach((b) => b.addEventListener("click", () => go(b.dataset.nav)));
  $("burger").addEventListener("click", () => {
    const open = $("app").classList.toggle("nav-open");
    $("burger").setAttribute("aria-expanded", String(open));
  });
  $("scrim").addEventListener("click", () => {
    $("app").classList.remove("nav-open");
    $("burger").setAttribute("aria-expanded", "false");
  });

  document.querySelectorAll(".seg__btn").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll(".seg__btn").forEach((o) => o.setAttribute("aria-selected", String(o === b)));
    const camera = b.dataset.source === "camera";
    $("uploadPane").hidden = camera;
    $("cameraPane").hidden = !camera;
    if (!camera) stopCamera();
    else startCamera();
  }));

  const drop = $("drop");
  ["dragenter", "dragover"].forEach((e) => drop.addEventListener(e, (ev) => {
    ev.preventDefault(); drop.classList.add("is-over");
  }));
  ["dragleave", "drop"].forEach((e) => drop.addEventListener(e, (ev) => {
    ev.preventDefault(); drop.classList.remove("is-over");
  }));
  drop.addEventListener("drop", (ev) => {
    const f = ev.dataTransfer?.files?.[0];
    if (f) setQuery(f);
  });
  $("queryFile").addEventListener("change", (e) => setQuery(e.target.files[0]));

  $("productSelect").addEventListener("change", syncRun);
  $("runBtn").addEventListener("click", runVerification);
  $("camStart").addEventListener("click", startCamera);
  $("camSnap").addEventListener("click", capture);
  window.addEventListener("beforeunload", stopCamera);

  $("libFiles").addEventListener("change", (e) => {
    state.libFiles = Array.from(e.target.files || []);
    renderLibThumbs();
  });
  $("libName").addEventListener("input", renderLibThumbs);
  $("libAdd").addEventListener("click", addReferences);

  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => e.preventDefault());

  renderStatus({
    model_loaded: false, device: "…", threshold: null,
    weights: {}, config: {}, products: 0, images: 0,
  });
  refreshStatus();
  loadProducts();
}

document.addEventListener("DOMContentLoaded", init);
