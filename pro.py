import html
import numpy as np
import cv2
import streamlit as st

import config
from preprocess import imread_bytes
from dataset import list_products, product_images
from scoring import Assessor, reasons

APP_NAME = "AI Product Authentication"
BRAND = "AI"

st.set_page_config(page_title=APP_NAME, layout="wide", initial_sidebar_state="expanded")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root{
  --bg:#0b0f1a; --panel:#121827; --panel-2:#182136;
  --border:rgba(255,255,255,.07); --border-strong:rgba(255,255,255,.14);
  --text:#e6ecf7; --muted:#8b96ad; --dim:#5c6b85;
  --accent:#6d5dfc; --accent2:#22d3ee;
  --ok:#10b981; --warn:#f59e0b; --bad:#ef4444; --info:#60a5fa;
}

html, body, [data-testid="stAppViewContainer"], [class*="css"]{
  font-family:'Inter','Segoe UI',system-ui,-apple-system,sans-serif;
  color:var(--text);
}

[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(1100px 520px at 14% -8%, rgba(109,93,252,.16), transparent 60%),
    radial-gradient(900px 480px at 108% 2%, rgba(34,211,238,.10), transparent 55%),
    linear-gradient(180deg,#0a0e18 0%, #0c121f 100%);
}

[data-testid="stHeader"]{background:transparent;}

.block-container{padding-top:1.4rem;padding-bottom:3rem;max-width:1440px;}

h1,h2,h3,h4{color:var(--text);letter-spacing:-.01em;}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#0d1322 0%, #0a0e18 100%);
  border-right:1px solid var(--border);
}
[data-testid="stSidebar"] > div:first-child{padding-top:1.1rem;padding-bottom:1.5rem;}

.side-brand{display:flex;align-items:center;gap:.75rem;padding:.2rem .1rem 1rem;}
.side-logo{
  width:42px;height:42px;border-radius:12px;flex:0 0 auto;
  display:flex;align-items:center;justify-content:center;
  font-weight:900;font-size:1.05rem;color:#fff;letter-spacing:.06em;
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  box-shadow:0 6px 22px rgba(109,93,252,.45);
}
.side-name{font-weight:800;font-size:.95rem;line-height:1.15;color:var(--text);}
.side-name span{color:var(--muted);font-weight:600;font-size:.8rem;display:block;}

[data-testid="stSidebar"] [role="radiogroup"]{gap:.2rem !important;}
[data-testid="stSidebar"] [role="radiogroup"] label{
  display:flex;align-items:center;gap:.6rem;
  padding:.55rem .8rem;border-radius:12px;cursor:pointer;
  color:var(--muted);font-weight:500;font-size:.92rem;
  border:1px solid transparent;transition:all .15s ease;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover{background:rgba(255,255,255,.04);color:var(--text);}
[data-testid="stSidebar"] [role="radiogroup"] label div:first-child{display:none;}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){
  background:linear-gradient(120deg,rgba(109,93,252,.22),rgba(34,211,238,.08));
  border-color:rgba(109,93,252,.45);color:#fff;font-weight:700;
}

[data-testid="stSidebar"] hr{border-color:var(--border);}

.side-cap{color:var(--dim);font-size:.74rem;line-height:1.5;}

/* ---------- shared blocks ---------- */
.hero{
  padding:1.5rem 1.9rem;border-radius:20px;margin-bottom:1.4rem;
  border:1px solid var(--border);
  background:linear-gradient(120deg,rgba(109,93,252,.14),rgba(34,211,238,.07) 45%,rgba(255,255,255,.02));
  box-shadow:0 18px 44px rgba(0,0,0,.35);
}
.hero-tag{font-size:.7rem;letter-spacing:.24em;font-weight:700;color:var(--muted);text-transform:uppercase;}
.hero-title{font-size:2.05rem;font-weight:900;letter-spacing:-.025em;margin-top:.3rem;line-height:1.05;}
.hero-title .grad{
  background:linear-gradient(90deg,var(--accent),var(--accent2));
  -webkit-background-clip:text;background-clip:text;color:transparent;
}
.hero-sub{color:var(--muted);font-size:.95rem;margin-top:.45rem;max-width:780px;line-height:1.55;}
.hero-chips{margin-top:.9rem;display:flex;flex-wrap:wrap;gap:.45rem;}

.sec{margin:1.4rem 0 .7rem;}
.sec-title{font-size:1.12rem;font-weight:800;display:flex;align-items:center;gap:.6rem;letter-spacing:-.01em;}
.sec-title::before{content:"";width:6px;height:1.15rem;border-radius:3px;
  background:linear-gradient(180deg,var(--accent),var(--accent2));}
.sec-sub{color:var(--muted);font-size:.85rem;margin-top:.25rem;}

.lbl{font-size:.68rem;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);font-weight:700;margin:.6rem 0 .3rem;}

.chip{
  display:inline-flex;align-items:center;gap:.5rem;
  border:1px solid var(--border);background:rgba(255,255,255,.04);
  border-radius:999px;padding:.34rem .85rem;font-size:.8rem;color:var(--text);
}
.chip .dot{width:8px;height:8px;border-radius:50%;flex:0 0 auto;}

.tile{
  background:linear-gradient(160deg,rgba(255,255,255,.055),rgba(255,255,255,.02));
  border:1px solid var(--border);border-radius:16px;padding:1.05rem 1.2rem;
}
.tile .t-label{font-size:.7rem;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);font-weight:700;}
.tile .t-value{font-size:1.75rem;font-weight:900;margin-top:.3rem;font-variant-numeric:tabular-nums;letter-spacing:-.02em;}
.tile .t-unit{font-size:.95rem;font-weight:700;color:var(--muted);margin-left:.15rem;}

.banner{
  border-radius:18px;padding:1.5rem 1.9rem;display:flex;align-items:center;gap:1.4rem;
  border:1px solid var(--border);margin-bottom:1.1rem;
}
.banner b-logo{
  width:66px;height:66px;border-radius:16px;flex:0 0 auto;
  display:flex;align-items:center;justify-content:center;
  font-weight:900;font-size:1.35rem;color:#fff;
}
.banner-ok{background:linear-gradient(115deg,rgba(16,185,129,.20),rgba(16,185,129,.05) 55%,rgba(255,255,255,.01));
  border-color:rgba(16,185,129,.35);}
.banner-fake{background:linear-gradient(115deg,rgba(239,68,68,.22),rgba(239,68,68,.05) 55%,rgba(255,255,255,.01));
  border-color:rgba(239,68,68,.40);}
.banner .b-text{flex:1;}
.banner .b-label{font-size:.72rem;letter-spacing:.18em;text-transform:uppercase;font-weight:800;}
.banner-ok .b-label{color:#34d399;}
.banner-fake .b-label{color:#f87171;}
.banner .b-score{font-size:2.5rem;font-weight:900;line-height:1;font-variant-numeric:tabular-nums;margin-top:.25rem;letter-spacing:-.02em;}
.banner .b-score small{font-size:1rem;color:var(--muted);font-weight:700;}
.banner .b-sub{color:var(--muted);font-size:.85rem;margin-top:.5rem;}
.banner .b-badge{
  text-align:center;flex:0 0 auto;
  border:1px solid var(--border);border-radius:14px;padding:.6rem .9rem;background:rgba(0,0,0,.2);
}
.banner .b-badge .bb-v{font-weight:800;font-size:.95rem;}
.banner .b-badge .bb-l{font-size:.62rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);margin-top:.15rem;}

.img-card{
  border:1px solid var(--border);border-radius:16px;overflow:hidden;
  background:rgba(0,0,0,.25);
}
.img-card img{border-radius:0;border-top:1px solid var(--border);}
.img-cap{padding:.6rem .9rem;font-size:.8rem;color:var(--text);display:flex;align-items:center;gap:.5rem;}
.img-cap .cap{flex:1;}
.img-cap .badge{font-size:.62rem;letter-spacing:.1em;text-transform:uppercase;font-weight:800;padding:.14rem .5rem;border-radius:999px;}

.sig{display:flex;align-items:center;gap:.9rem;padding:.62rem 0;border-bottom:1px dashed rgba(255,255,255,.05);}
.sig:last-child{border-bottom:none;}
.sig-label{width:270px;color:var(--muted);font-size:.86rem;font-weight:500;}
.sig-label b{color:var(--text);font-weight:700;}
.sig-track{flex:1;height:10px;border-radius:999px;background:rgba(255,255,255,.06);overflow:hidden;}
.sig-fill{height:100%;border-radius:999px;transition:width .5s ease;}
.sig-pct{width:56px;text-align:right;font-weight:800;font-size:.9rem;font-variant-numeric:tabular-nums;}
.sig-w{width:44px;text-align:right;color:var(--dim);font-size:.75rem;font-weight:600;}

.pgrid{display:grid;grid-template-columns:repeat(2,1fr);gap:.85rem;}
.pcard{
  border:1px solid var(--border);border-radius:16px;padding:1rem 1.2rem;
  background:linear-gradient(160deg,rgba(255,255,255,.05),rgba(255,255,255,.015));
}
.pcard .p-name{font-weight:800;font-size:.98rem;display:flex;align-items:center;gap:.55rem;word-break:break-word;}
.pcard .p-name .pc-dot{width:9px;height:9px;border-radius:50%;flex:0 0 auto;
  background:linear-gradient(135deg,var(--accent),var(--accent2));}
.pcard .p-count{color:var(--muted);font-size:.82rem;margin-top:.45rem;}
.pcard .p-count b{color:var(--text);}

.empty{
  padding:1.5rem;border:1px dashed rgba(255,255,255,.16);border-radius:16px;
  color:var(--muted);font-size:.92rem;text-align:center;
}

/* ---------- streamlit widgets ---------- */
.stButton > button{
  border-radius:12px;font-weight:600;font-size:.92rem;
  border:1px solid var(--border-strong);background:rgba(255,255,255,.05);color:var(--text);
  padding:.5rem 1.05rem;transition:all .15s ease;
}
.stButton > button:hover{border-color:var(--accent);background:rgba(109,93,252,.12);color:#fff;}
.stButton > button[kind="primary"]{
  background:linear-gradient(120deg,var(--accent),#4f46e5 55%,var(--accent2));
  border:none;color:#fff;box-shadow:0 8px 24px rgba(109,93,252,.35);
}
.stButton > button[kind="primary"]:hover{filter:brightness(1.12);box-shadow:0 10px 30px rgba(109,93,252,.5);}

.stProgress > div > div{background:rgba(255,255,255,.07);border-radius:999px;}
.stProgress > div > div > div > div{
  background:linear-gradient(90deg,var(--accent),var(--accent2));
  border-radius:999px;
}

div[data-baseweb="select"] > div,
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input{
  background:rgba(255,255,255,.04);border-radius:12px;
  border:1px solid var(--border)!important;color:var(--text)!important;
}
div[data-baseweb="select"] > div:hover{border-color:var(--accent)!important;}

[data-testid="stFileUploaderDropzone"]{
  border:1px dashed rgba(255,255,255,.22);border-radius:14px;
  background:rgba(255,255,255,.02);color:var(--muted);
}
[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--accent);background:rgba(109,93,252,.07);}

[data-testid="stVerticalBlockBorderWrapper"]{
  background:linear-gradient(160deg,rgba(255,255,255,.035),rgba(255,255,255,.012));
  border-color:var(--border)!important;border-radius:16px;
}

[data-testid="stAlert"]{
  border-radius:14px;border:1px solid var(--border);
}

.stTabs [data-baseweb="tab-list"]{gap:.5rem;}
.stTabs [data-baseweb="tab"]{
  background:rgba(255,255,255,.03);border:1px solid var(--border);
  border-radius:999px;padding:.35rem 1.1rem;color:var(--muted);
}
.stTabs [aria-selected="true"]{
  background:linear-gradient(120deg,rgba(109,93,252,.22),rgba(34,211,238,.10));
  border-color:rgba(109,93,252,.5);color:#fff;font-weight:700;
}

[data-testid="stSlider"] [role="slider"]{background:var(--accent);}
</style>
"""


@st.cache_resource
def get_assessor():
    return Assessor()


def esc(text):
    return html.escape(str(text))


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def hero():
    has_model = get_assessor().model is not None
    device = get_assessor().device.upper()
    chips = []
    chips.append(
        f'<span class="chip"><span class="dot" style="background:{("#10b981" if has_model else "#f59e0b")}"></span>'
        f'Siamese model {"active" if has_model else "not trained"}</span>'
    )
    chips.append(f'<span class="chip"><span class="dot" style="background:#22d3ee"></span>Device: {device}</span>')
    chips.append(
        f'<span class="chip"><span class="dot" style="background:#60a5fa"></span>'
        f'Verdict threshold: {config.BINARY_THRESHOLD:.0f} → Original</span>'
    )
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-tag">Introduction to Computer Vision · Team Mini Project</div>
          <div class="hero-title">AI Product <span class="grad">Authentication</span></div>
          <div class="hero-sub">
            Authenticate product packaging using logo and packaging recognition — verifying the
            package against an authenticated reference library with logo/print alignment, colour
            forensics, physical print-quality analysis, embedded security QR codes and a learned
            Siamese visual signature.
          </div>
          <div class="hero-chips">{''.join(chips)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title, sub=""):
    s = f'<div class="sec"><div class="sec-title">{esc(title)}</div>'
    if sub:
        s += f'<div class="sec-sub">{esc(sub)}</div>'
    return s + "</div>"


def panel_header(label, dot):
    return (
        f'<div class="lbl" style="margin-top:0"><span style="display:inline-block;width:8px;height:8px;'
        f'border-radius:50%;background:{dot};margin-right:.5rem"></span>{esc(label)}</div>'
    )


def fill_color(v):
    if v >= 0.7:
        return "#10b981"
    if v >= 0.4:
        return "#f59e0b"
    return "#ef4444"


SIGNAL_LABELS = {
    "template": ("Packaging layout match", "#22d3ee"),
    "orb": ("Logo / print feature match", "#a78bfa"),
    "color": ("Colour distribution", "#34d399"),
    "edge_layout": ("Print edge structure", "#60a5fa"),
    "sharpness": ("Print sharpness", "#fbbf24"),
    "noise": ("Print surface noise", "#f87171"),
    "qr": ("Security QR code", "#2dd4bf"),
    "siamese": ("Learned visual signature", "#f472b6"),
}


def signal_rows(scores):
    rows = []
    weights = config.WEIGHTS
    for key, val in scores.items():
        label, dot = SIGNAL_LABELS.get(key, (key, "#8b96ad"))
        rows.append(
            f'<div class="sig">'
            f'<span class="sig-label"><span style="display:inline-block;width:8px;height:8px;'
            f'border-radius:50%;background:{dot};margin-right:.5rem"></span>{esc(label)}</span>'
            f'<div class="sig-track"><div class="sig-fill" style="width:{val*100:.0f}%;'
            f'background:linear-gradient(90deg,{dot},{fill_color(val)})"></div></div>'
            f'<span class="sig-w">{weights.get(key, 0) * 100:.0f}%</span>'
            f'<span class="sig-pct" style="color:{fill_color(val)}">{val*100:.0f}</span>'
            f'</div>'
        )
    return '<div style="border:1px solid var(--border);border-radius:16px;padding:.4rem 1.2rem;">' + "".join(rows) + "</div>"


def show_result(result):
    score = result["score"]
    verdict = result["verdict"]
    binary = result["binary"]
    is_orig = binary == "Original"
    vmap = {
        "Authentic": "#34d399",
        "Likely Authentic": "#60a5fa",
        "Suspicious": "#fbbf24",
        "Counterfeit": "#f87171",
    }
    vc = vmap.get(verdict, "#e6ecf7")
    m = result["metrics"]

    confidence = score if is_orig else max(0.0, 100 - score)
    banner_class = "banner-ok" if is_orig else "banner-fake"
    banner_label = "ORIGINAL" if is_orig else "FAKE"
    banner_sub = (
        "This product matches the authenticated reference for this item."
        if is_orig
        else "This product deviates from the authenticated reference — counterfeiting is suspected."
    )
    logo_colors = ("#059669,#10b981" if is_orig else "#dc2626,#ef4444")
    logo_text = "OK" if is_orig else "!!"

    st.markdown(
        f"""
        <div class="banner {banner_class}">
          <div class="b-logo" style="background:linear-gradient(135deg,{logo_colors})">{logo_text}</div>
          <div class="b-text">
            <div class="b-label">{banner_label}</div>
            <div class="b-score">{score:.1f}<small> / 100</small></div>
            <div class="b-sub">{banner_sub}</div>
          </div>
          <div class="b-badge">
            <div class="bb-v" style="color:{vc}">{esc(verdict)}</div>
            <div class="bb-l">Detailed verdict</div>
          </div>
          <div class="b-badge">
            <div class="bb-v">{confidence:.1f}%</div>
            <div class="bb-l">Confidence</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="lbl">Authenticity score</div>', unsafe_allow_html=True)
    st.progress(min(1.0, max(0.0, score / 100.0)))

    align_score = m["alignment_score"]
    align_dot = "#10b981" if align_score > 0.2 else "#f59e0b"
    qr_chip = ""
    qr_status = m.get("qr_status")
    qr_map = {
        "qr_match": ("#10b981", "QR matches reference"),
        "missing_qr": ("#f59e0b", "QR missing from the query"),
        "unexpected_qr": ("#f59e0b", "Unexpected QR on the query"),
        "qr_mismatch": ("#ef4444", "QR content differs from the reference"),
        "skipped": ("#8b96ad", "QR check skipped"),
    }
    if qr_status and qr_status in qr_map:
        qcolor, qtext = qr_map[qr_status]
        qr_chip = f'<span class="chip"><span class="dot" style="background:{qcolor}"></span>{qtext}</span>'

    st.markdown(
        f'<div class="hero-chips">'
        f'<span class="chip"><span class="dot" style="background:{align_dot}"></span>'
        f'Aligned to reference · {m["alignment_matches"]} keypoint matches '
        f'(confidence {align_score:.2f})</span>'
        f'{qr_chip}'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown(section("Visual evidence", "Reference versus query, plus the suspicious-region overlay after alignment."), unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(panel_header("Original (reference)", "#34d399"), unsafe_allow_html=True)
        st.image(result["ref"], channels="BGR", use_container_width=True)
    with c2:
        st.markdown(panel_header("Test image (query)", "#fbbf24"), unsafe_allow_html=True)
        st.image(result["query"], channels="BGR", use_container_width=True)
    with c3:
        st.markdown(panel_header("Suspicious-region overlay", "#ef4444"), unsafe_allow_html=True)
        st.image(result["overlay"], channels="BGR", use_container_width=True)

    st.markdown(
        section(
            "Signal breakdown",
            "Per-signal match score, weighted by each signal's contribution to the final verdict. "
            "Green ≥ 70 · amber 40–69 · red &lt; 40.",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(signal_rows(result["scores"]), unsafe_allow_html=True)

    rsn = reasons(result)
    if rsn:
        st.markdown(section("Why this verdict", "Signals that weakened (or confirmed) the authenticity decision."), unsafe_allow_html=True)
        chips = "".join(
            f'<span class="chip" style="margin:.15rem .4rem .15rem 0"><span class="dot" style="background:#f59e0b"></span>{esc(t)}</span>'
            for t in rsn
        )
        st.markdown('<div class="hero-chips">' + chips + "</div>", unsafe_allow_html=True)


def page_verify():
    st.markdown(section("Verify a product", "Choose the product and upload or capture a photo to authenticate it."), unsafe_allow_html=True)
    products = list_products()
    if not products:
        st.markdown(
            '<div class="empty">No reference products yet. Add reference images in the <b>Product Library</b> tab first.</div>',
            unsafe_allow_html=True,
        )
        return

    with st.container(border=True):
        product = st.selectbox("Product", products)
        c_up, c_cam = st.columns(2)
        up = c_up.file_uploader(
            "Upload photo of the product", type=["jpg", "jpeg", "png", "bmp", "webp"]
        )
        cam = c_cam.camera_input("…or capture with your camera")

    data = None
    if up is not None:
        data = up.getvalue()
    elif cam is not None:
        data = cam.getvalue()

    if data is None:
        st.info("Upload or capture an image to verify.")
        return

    buf = up.getvalue() if up is not None else cam.getvalue()
    query = imread_bytes(buf)
    if query is None:
        st.error("Could not decode the image.")
        return

    run = st.button("Run verification", type="primary")
    if not run:
        return

    from dataset import product_ref

    refs = product_ref(product)
    with st.spinner("Analysing packaging, logo, colour and security features…"):
        result = get_assessor().assess(refs, query)
    show_result(result)


def page_library():
    st.markdown(section("Product library", "Manage the authenticated reference imagery used for verification."), unsafe_allow_html=True)
    c_add, c_view = st.columns([1.1, 1.35], gap="large")

    with c_add:
        st.markdown(panel_header("Add reference photos", "#34d399"), unsafe_allow_html=True)
        with st.container(border=True):
            product_name = st.text_input("Product name", placeholder="e.g. Coca-Cola 330ml")
            up_refs = st.file_uploader(
                "Reference images (several angles improve robustness)",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
                accept_multiple_files=True,
            )
            cam_ref = st.camera_input("…or capture a reference photo")
            if st.button("Save reference images", type="primary"):
                from dataset import save_reference

                if not product_name.strip():
                    st.warning("Enter a product name first.")
                else:
                    name = product_name.strip()
                    n = 0
                    for f in (up_refs or []):
                        save_reference(name, f.name, f.getvalue())
                        n += 1
                    if cam_ref is not None:
                        save_reference(name, "camera_reference.png", cam_ref.getvalue())
                        n += 1
                    st.success(f"Saved {n} reference image(s) for '{name}'.")
                    st.cache_resource.clear()
                    st.rerun()

    with c_view:
        st.markdown(panel_header("Reference library", "#60a5fa"), unsafe_allow_html=True)
        known = list_products()
        if not known:
            st.markdown('<div class="empty">The library is empty.</div>', unsafe_allow_html=True)
        else:
            cards = []
            for name in known:
                cnt = len(product_images(name))
                cards.append(
                    f'<div class="pcard"><div class="p-name"><span class="pc-dot"></span>{esc(name)}</div>'
                    f'<div class="p-count"><b>{cnt}</b> reference image(s)</div></div>'
                )
            st.markdown(f'<div class="pgrid">{"".join(cards)}</div>', unsafe_allow_html=True)
            clear_sel = st.multiselect("Remove products (deletes their images)", known)
            if st.button("Remove selected") and clear_sel:
                import shutil

                for name in clear_sel:
                    shutil.rmtree(config.GENUINE_DIR / name)
                st.cache_resource.clear()
                st.rerun()


def page_train():
    st.markdown(
        section(
            "Siamese CNN training",
            "Trains a contrastive encoder that tells genuine packaging from synthetic counterfeits.",
        ),
        unsafe_allow_html=True,
    )

    has_weights = config.MODEL_PATH.exists()
    has_model = get_assessor().model is not None
    device = get_assessor().device.upper()

    c_status, c_cfg = st.columns([1.05, 1.35], gap="large")
    with c_status:
        st.markdown(panel_header("Model status", "#22d3ee"), unsafe_allow_html=True)
        chips = []
        chips.append(
            f'<span class="chip"><span class="dot" style="background:{"#10b981" if has_weights else "#8b96ad"}"></span>'
            f'Checkpoint: {"present" if has_weights else "not trained"}</span>'
        )
        chips.append(
            f'<span class="chip"><span class="dot" style="background:{"#10b981" if has_model else "#f59e0b"}"></span>'
            f'Encoder: {"loaded" if has_model else "disabled"}</span>'
        )
        chips.append(f'<span class="chip"><span class="dot" style="background:#22d3ee"></span>Device: {device}</span>')
        chips.append(
            '<span class="chip"><span class="dot" style="background:#a78bfa"></span>Signals: classical + learned</span>'
        )
        st.markdown('<div class="hero-chips">' + "".join(chips) + "</div>", unsafe_allow_html=True)
        st.markdown(
            f'<div class="sec-sub" style="margin-top:.8rem">Checkpoint path: <code>{config.MODEL_PATH}</code>. '
            f'Without a trained model the system still works using classical computer-vision signals only.</div>',
            unsafe_allow_html=True,
        )

    with c_cfg:
        st.markdown(panel_header("Training configuration", "#a78bfa"), unsafe_allow_html=True)
        with st.container(border=True):
            epochs = st.slider("Epochs", 2, 40, 10)
            batch = st.slider("Batch size", 8, 64, 32)
            if st.button("Train now", type="primary"):
                if len(list_products()) == 0:
                    st.error("Add reference images first.")
                else:
                    progress = st.progress(0.0)
                    chart = st.empty()
                    history = []
                    from train_script import train_model

                    def cb(ep, loss):
                        history.append(loss)
                        progress.progress(ep / epochs)
                        chart.line_chart(history)

                    try:
                        history, out = train_model(
                            epochs=epochs, batch=batch, image_size=config.SIAMESE_SIZE,
                            lr=5e-4, out_path=config.MODEL_PATH, log_cb=cb,
                        )
                        st.success(f"Training complete. Loss history: {[round(x, 3) for x in history]}")
                        st.cache_resource.clear()
                    except RuntimeError as e:
                        st.error(str(e))


def sidebar():
    st.sidebar.markdown(
        f'<div class="side-brand"><div class="side-logo">{BRAND}</div>'
        f'<div class="side-name">Product Authentication<span>Logo &amp; Packaging Recognition</span></div></div>',
        unsafe_allow_html=True,
    )

    page = st.sidebar.radio(
        "Navigate",
        ["Verify Product", "Product Library", "Train Siamese"],
        label_visibility="collapsed",
    )

    st.sidebar.divider()
    st.sidebar.markdown(panel_header("System status", "#22d3ee"), unsafe_allow_html=True)
    has_model = get_assessor().model is not None
    chips = [
        f'<span class="chip"><span class="dot" style="background:{"#10b981" if has_model else "#f59e0b"}"></span>'
        f'Encoder {"active" if has_model else "not trained"}</span>',
        f'<span class="chip"><span class="dot" style="background:#22d3ee"></span>{get_assessor().device.upper()}</span>',
    ]
    st.sidebar.markdown('<div class="hero-chips">' + "".join(chips) + "</div>", unsafe_allow_html=True)

    st.sidebar.divider()
    st.sidebar.markdown(
        '<div class="side-cap">Authenticates packaging via logo/print and layout recognition, colour and edge '
        'forensics, print-quality analysis, embedded QR security and a learned Siamese signature.<br><br>'
        "Team mini-project · Introduction to Computer Vision<br>Built with OpenCV · PyTorch · Streamlit</div>",
        unsafe_allow_html=True,
    )
    return page


inject_css()
hero()
page = sidebar()

if page == "Verify Product":
    page_verify()
elif page == "Product Library":
    page_library()
else:
    page_train()