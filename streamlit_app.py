import os, sys, json, html, base64, math
import numpy as np
import cv2
import streamlit as st
from PIL import Image, ImageOps

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE + "/src")
st.set_page_config(page_title="Libras GFE Lab", page_icon="🙂", layout="wide")

MAX_SIDE = 1280


@st.cache_resource(show_spinner="Loading the model (first visit takes a moment)...")
def load_models():
    import inference, cv_pipeline, overlay_view
    zall = json.load(open(BASE + "/models/zone_weights.json"))
    return inference, cv_pipeline, overlay_view, zall


inference, cv_pipeline, overlay_view, ZALL = load_models()
PER_EXPR = ZALL.get("per_expression")

# ================================================================ content
EXPRESSIONS = {
    "affirmative": ("Affirmative", "Used when making a statement, to assert that something is true. The dataset paper calls it \"assertion\". Head nodding (up and down) goes with it.",
                    "Asserts that a statement is true. Goes with head nodding."),
    "conditional": ("Conditional", "Marks the \"if\" part of a sentence: a condition that must hold for the main clause.",
                    "Marks the \"if\" part of a sentence."),
    "doubt_question": ("Doubt Question", "Looks like a question but does not expect an answer. It draws attention to the information that follows. The paper links it with lowered eyebrows and a compressed mouth.",
                       "Looks like a question but expects no answer."),
    "emphasis": ("Emphasis", "Highlights the new or important information in what is being said. The paper calls it \"focus\".",
                 "Highlights the new or important information."),
    "negative": ("Negative", "Marks a negative sentence (\"not\", \"no\"). The paper links it with lowered eyebrows, a downturned mouth and a side-to-side head shake.",
                 "Marks a negative sentence (\"not\", \"no\")."),
    "relative": ("Relative", "Marks a clause that gives more information about something, as in \"the person who...\".",
                 "Adds information: \"the person who...\"."),
    "topics": ("Topics", "Moves the topic of the sentence to the front, to say what it is about before commenting on it.",
               "Puts the topic first: what the sentence is about."),
    "wh_question": ("WH Question", "Asks who, what, when, where, how or why.",
                    "Asks who, what, when, where, how or why."),
    "yn_question": ("Yes/No Question", "Asks a question that can be answered with yes or no.",
                    "Asks a question answered with yes or no."),
}


def nice(name):
    return EXPRESSIONS.get(name, (name.replace("_", " ").capitalize(), "", ""))[0]


def h(s):
    return " ".join(s.split())


# ================================================================ assets
def data_uri(rel, mime):
    p = os.path.join(BASE, rel)
    if os.path.exists(p):
        return "data:" + mime + ";base64," + base64.b64encode(open(p, "rb").read()).decode()
    return None


BG = data_uri("assets/bg.jpg", "image/jpeg")
FACE = data_uri("assets/face.webp", "image/webp")


# ================================================================ look
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600;700;800&family=Space+Mono:wght@700&display=swap');
:root{--orange:#ff9a52; --tan:#d6975a; --ink:#ffffff; --mute:#cfd9f7; --teal:#5fd6c8; --line:rgba(255,255,255,.22);}
.stApp{background: BGLAYER linear-gradient(180deg,#02196a 0%,#1a3d9c 42%,#5b88ee 100%) fixed !important; background-blend-mode: soft-light, normal; font-family:'Figtree',system-ui,sans-serif; color:var(--ink);}
[data-testid="stHeader"]{background:transparent;}
.block-container{max-width:1180px; padding-top:4.2rem; padding-bottom:3rem;}
#MainMenu, footer{visibility:hidden;}
h1,h2,h3{font-family:'Figtree',sans-serif;}
button, button p{font-family:'Figtree',sans-serif;}
/* nav */
.brand{display:flex; align-items:center; gap:2px; font-family:'Space Mono',monospace; font-weight:700; font-size:1.35rem; color:#fff; padding-top:2px;}
.brand .logo{width:26px; height:34px; margin-right:2px;}
.brand .bar{width:2px; height:36px; background:var(--orange); margin-left:22px;}
.pill{display:inline-flex; align-items:center; gap:8px; border:1px solid rgba(255,255,255,.35); border-radius:999px; padding:7px 15px; background:rgba(10,30,120,.55); color:#fff; font-size:.78rem; font-weight:600; float:right;}
[class*="st-key-nav_"] button{background:transparent !important; border:0 !important; border-bottom:3px solid transparent !important; border-radius:0 !important; box-shadow:none !important; color:#fff !important; font-family:'Space Mono',monospace !important; font-weight:700 !important; font-size:1.02rem !important; padding:.35rem .15rem !important; min-height:0 !important;}
[class*="st-key-nav_"] button:hover{color:var(--orange) !important;}
[class*="st-key-nav_"] button p{font-size:1.05rem !important; font-weight:700 !important; color:inherit !important;}
.st-key-explore button p, .st-key-ready button p, .st-key-try button p{font-size:1.15rem !important; font-weight:800 !important; color:inherit !important;}
/* home */
.hl{font-family:'Figtree',sans-serif; font-weight:800; font-size:clamp(2.6rem,5.6vw,4.6rem); line-height:1.04; margin:34px 0 26px; letter-spacing:-.02em;}
.hl .o{color:#ff8a4c;} .hl .r{color:#b10d0d;}
.lead{font-size:1.18rem; line-height:1.6; color:#fff; max-width:600px; margin-bottom:34px;}
.face{width:100%; max-height:560px; object-fit:contain; display:block; margin-left:auto; filter:drop-shadow(0 18px 40px rgba(0,0,0,.35));}
.mesh{width:100%; max-width:480px; display:block; margin:0 auto;}
.discover{background:#dfe4ec; color:#0b0f1f; border-radius:44px 44px 0 0; padding:34px 44px 30px; font-weight:800; font-size:clamp(1.5rem,2.8vw,2.3rem); line-height:1.15; margin-top:26px; letter-spacing:-.01em;}
.st-key-explore button, .st-key-ready button, .st-key-try button{border:0 !important; border-radius:999px !important; font-weight:800 !important; letter-spacing:.02em; transition:all .2s ease;}
.st-key-explore button{background:#3d58b8 !important; color:#fff !important; padding:.8rem 2.4rem !important; font-size:1.2rem !important; box-shadow:0 6px 0 #263f93;}
.st-key-ready button, .st-key-try button{background:#ff9650 !important; color:#0a0f24 !important; padding:.85rem 2rem !important; font-size:1.15rem !important; box-shadow:0 7px 0 #c8651f;}
.st-key-explore button:hover, .st-key-ready button:hover, .st-key-try button:hover{transform:translateY(-2px);}
/* study */
.sec{display:flex; align-items:center; gap:18px; font-size:2.05rem; font-weight:500; margin:6px 0 14px; color:#fff;}
.sec:after{content:""; flex:1; height:2px; background:rgba(255,255,255,.75);}
.exg{display:grid; grid-template-columns:1fr 1fr; gap:22px 18px;}
.ex{display:flex; align-items:center; gap:14px; background:linear-gradient(180deg,#4a6fd4,#3558bf); border-radius:30px; padding:14px 20px; box-shadow:0 8px 0 rgba(12,30,110,.5); transition:transform .2s ease;}
.ex:hover{transform:translateY(-3px);}
.ex .n{font-size:1.65rem; font-weight:700; min-width:34px;}
.ex b{display:block; font-size:1.02rem; font-weight:700;}
.ex span{display:block; font-size:.82rem; color:var(--mute); line-height:1.35; margin-top:2px;}
.ex.last{grid-column:1/-1; width:52%; margin:0 auto;}
.flow{display:flex; align-items:center; gap:8px;}
.fc{flex:1; background:linear-gradient(180deg,#2b4cb0,#3a5cc2); border-radius:16px; padding:22px 12px 16px; text-align:center; min-height:196px; box-shadow:0 8px 0 rgba(12,30,110,.5);}
.fc svg{width:56px; height:56px; margin-bottom:14px;}
.fc b{display:block; font-size:.98rem; margin-bottom:6px;}
.fc span{font-size:.78rem; color:var(--mute); line-height:1.4; display:block;}
.arrow{color:#fff; font-size:1.2rem;}
.chk{display:flex; flex-direction:column; gap:10px;}
.chk div{display:flex; align-items:center; gap:12px; font-size:1.0rem;}
.chk i{width:28px; height:28px; border-radius:50%; background:var(--teal); display:inline-flex; align-items:center; justify-content:center; flex:none;}
.chk i svg{width:15px; height:15px;}
.privacy{display:flex; align-items:center; gap:18px; background:rgba(40,70,170,.75); border:1px solid rgba(255,255,255,.25); border-radius:14px; padding:18px 22px; box-shadow:0 8px 0 rgba(12,30,110,.45); margin-top:20px;}
.privacy svg{width:44px; height:44px; flex:none;}
.privacy b{display:block; color:var(--orange); font-size:1.05rem; margin-bottom:2px;}
.crop{width:100% !important; height:620px !important; max-width:none !important; object-fit:cover !important; object-position:35% 30% !important; border-radius:28px; display:block; background:linear-gradient(180deg,#2a4aad,#3a5bc1); box-shadow:9px 9px 0 rgba(8,22,90,.6);}
/* demo */
[class*="st-key-card_"]{background:linear-gradient(180deg,#2a4aad,#3a5bc1); border-radius:26px; padding:20px 22px 22px; box-shadow:9px 9px 0 rgba(8,22,90,.6); margin-bottom:22px;}
.ct{display:flex; align-items:baseline; gap:14px; margin-bottom:14px;}
.ct .n{font-size:2.1rem; font-weight:800; color:#fff;}
.ct .t{font-size:1.15rem; font-weight:700; color:var(--orange);}
.dash{border:2px dashed rgba(255,255,255,.7); border-radius:20px; padding:18px; min-height:150px; display:flex; align-items:center; justify-content:center; text-align:center; color:var(--mute); line-height:1.5;}
.tips{display:flex; flex-direction:column; gap:7px; margin:6px 0 12px; font-size:.86rem;}
.tips div{display:flex; align-items:center; gap:9px;}
.tips svg{width:15px; height:15px; flex:none;}
.tt{color:var(--orange); font-weight:700; font-size:1.05rem; margin:12px 0 6px; display:flex; align-items:center; gap:8px;}
[class*="st-key-card_"] .stButton button{border-radius:999px !important; border:0 !important; font-weight:700 !important; box-shadow:0 4px 0 rgba(0,0,0,.28); color:#fff !important; background:var(--tan) !important; padding:.55rem 1rem !important;}
[class*="st-key-card_"] .stButton button p{font-size:1.02rem !important; font-weight:700 !important; color:inherit !important;}
[class*="st-key-card_"] [class*="st-key-ex_"] button{padding:.3rem .1rem !important; min-width:0 !important;}
[class*="st-key-card_"] [class*="st-key-ex_"] button p{font-size:.85rem !important; overflow:visible !important; text-overflow:clip !important;}
[class*="st-key-card_"] .stButton button:hover{filter:brightness(1.08); transform:translateY(-1px);}
[class*="st-key-card_"] .stButton button:disabled{opacity:.5;}
[data-testid="stFileUploaderDropzone"]{background:transparent; border:2px dashed rgba(255,255,255,.7) !important; border-radius:20px;}
[data-testid="stFileUploaderDropzone"] *{color:#fff !important;}
[data-testid="stExpander"]{border:1px solid var(--line) !important; border-radius:14px !important; background:rgba(255,255,255,.06);}
[data-testid="stImage"] img{border-radius:14px; box-shadow:0 8px 28px rgba(0,0,0,.35);}
[data-testid="stCaptionContainer"]{color:var(--mute);}
.card{background:rgba(255,255,255,.10); border:1px solid rgba(255,255,255,.2); border-radius:14px; padding:14px 16px; margin-top:12px; color:#fff; animation:rise .5s ease both;}
.card.main{font-size:1.06rem; line-height:1.5; border-color:rgba(255,154,82,.7); background:linear-gradient(135deg, rgba(255,154,82,.22), rgba(255,255,255,.08) 60%);}
.card b{color:var(--orange);}
.card .ttl{font-weight:700; margin-bottom:10px;}
.sub{color:var(--mute); font-size:.85rem; margin-top:6px; line-height:1.5; display:block;}
@keyframes rise{from{opacity:0; transform:translateY(12px);}}
.row{display:flex; align-items:center; gap:10px; margin:8px 0;}
.nm{width:132px; font-size:.9rem;}
.track{flex:1; background:rgba(255,255,255,.18); border-radius:6px; height:10px; overflow:hidden;}
.fill{height:100%; background:linear-gradient(90deg,#ffb454,#ff8f6b); border-radius:6px; animation:grow 1.1s cubic-bezier(.2,.8,.2,1) both;}
@keyframes grow{from{width:0 !important;}}
.val{width:48px; text-align:right; font-size:.85rem; color:var(--mute);}
.legend{display:flex; gap:14px; flex-wrap:wrap; color:var(--mute); font-size:.82rem; margin-top:8px;}
.legend i{display:inline-block; width:11px; height:11px; border-radius:3px; margin-right:6px; vertical-align:-1px;}
.lim{margin:0; padding-left:20px; line-height:1.6; font-size:.98rem;}
.lim li{margin:6px 0;}
.foot{color:#dbe4ff; font-size:.84rem; padding:30px 0 4px; border-top:1px solid var(--line); margin-top:40px; line-height:1.6;}
.exp{margin:9px 0; line-height:1.55;}
.exp b{color:var(--orange);}
"""

BGLAYER = f"url({BG}) center/cover fixed," if BG else ""
st.markdown("<style>" + CSS.replace("BGLAYER", BGLAYER) + "</style>", unsafe_allow_html=True)

# ================================================================ small pieces
CHECK = '<svg viewBox="0 0 16 16"><path d="M3 8.5l3.2 3.2L13 4.8" fill="none" stroke="#fff" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>'
CHECK_O = '<svg viewBox="0 0 16 16"><path d="M3 8.5l3.2 3.2L13 4.8" fill="none" stroke="#ff9a52" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/></svg>'
LOGO = ('<svg class="logo" viewBox="0 0 28 36"><path d="M9 3 C4 11 5 22 9 29 C11 33 17 34 23 31" fill="none" stroke="#ff9a52" stroke-width="3" stroke-linecap="round"/>'
        '<path d="M3 27 C6 34 14 36 20 33" fill="none" stroke="#ff9a52" stroke-width="2.4" stroke-linecap="round"/></svg>')
FLASK = ('<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
         '<path d="M9 3h6M10 3v6L4.5 19a1.5 1.5 0 0 0 1.3 2.2h12.4a1.5 1.5 0 0 0 1.3-2.2L14 9V3"/><path d="M7 15h10"/></svg>')
FOOT = h("""<div class="foot">Libras GFE Lab &middot; Research demonstration. Demo only, 2D only. Photos are not saved by this app. Results may be inaccurate.
The system analyzes grammatical facial expressions, not emotions, and it is not a Libras interpreter.
Built on landmark data from two Libras signers (Freitas et al., 2014).</div>""")


def go_page(p):
    st.session_state.page = p
    st.rerun()


def nav():
    c = st.columns([2.6, 0.7, 1.0, 0.7, 1.9], vertical_alignment="center")
    c[0].markdown(h(f'<div class="brand">{LOGO}<span>ibras GFE Lab</span><span class="bar"></span></div>'), unsafe_allow_html=True)
    for col, key, label in ((c[1], "home", "Home"), (c[2], "study", "The Study"), (c[3], "demo", "Demo")):
        with col:
            if st.button(label, key="nav_" + key):
                go_page(key)
    c[4].markdown(h(f'<div class="pill">{FLASK} Research Demo</div>'), unsafe_allow_html=True)
    st.markdown("<style>.st-key-nav_" + st.session_state.page + " button{border-bottom-color:#ff9a52 !important;}</style>",
                unsafe_allow_html=True)


def mesh_svg():
    """Fallback hero picture if assets/face.webp is missing: a face of 100 dots joined by a mesh."""
    pts = []
    for cx in (170, 310):
        for i in range(10):
            t = i / 9
            pts.append((cx - 48 + 96 * t, 96 - 16 * math.sin(math.pi * t)))
        for i in range(5):
            t = i / 4
            pts.append((cx - 40 + 80 * t, 70 - 14 * math.sin(math.pi * t)))
    for cx in (175, 305):
        for i in range(8):
            a = 2 * math.pi * i / 8
            pts.append((cx + 34 * math.cos(a), 142 + 11 * math.sin(a)))
        pts.append((cx, 142))
    for i in range(4):
        pts.append((240, 150 + i * 16))
    for i in range(8):
        t = i / 7
        pts.append((206 + 68 * t, 214 + 9 * math.sin(math.pi * t)))
    pts.append((240, 206))
    for i in range(12):
        a = 2 * math.pi * i / 12
        pts.append((240 + 50 * math.cos(a), 276 + 20 * math.sin(a)))
    for i in range(8):
        a = 2 * math.pi * i / 8
        pts.append((240 + 30 * math.cos(a), 276 + 7 * math.sin(a)))
    for i in range(19):
        th = math.pi * (0.03 + 0.94 * i / 18)
        pts.append((240 + 152 * math.cos(th), 168 + 178 * math.sin(th)))
    lines = ""
    try:
        from scipy.spatial import Delaunay
        tri = Delaunay(np.array(pts))
        edges = set()
        for s in tri.simplices:
            for a, b in ((s[0], s[1]), (s[1], s[2]), (s[0], s[2])):
                edges.add((min(a, b), max(a, b)))
        lines = "".join(f'<line x1="{pts[a][0]:.1f}" y1="{pts[a][1]:.1f}" x2="{pts[b][0]:.1f}" y2="{pts[b][1]:.1f}"/>'
                        for a, b in edges
                        if math.dist(pts[a], pts[b]) < 120)
    except Exception:
        pass
    dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.6"/>' for x, y in pts)
    return h(f'<svg class="mesh" viewBox="0 0 480 360"><g stroke="#fff" stroke-opacity=".5" stroke-width="1">{lines}</g><g fill="#fff">{dots}</g></svg>')


def hero_picture():
    if FACE:
        return f'<img class="face" src="{FACE}" alt="Face with landmark mesh">'
    return mesh_svg()


def expressions_expander():
    with st.expander("What do the nine expressions mean?"):
        for key, (name, text, short) in EXPRESSIONS.items():
            st.markdown(h(f'<div class="exp"><b>{html.escape(name)}</b>: {html.escape(text)}</div>'), unsafe_allow_html=True)
        st.caption("Descriptions follow Freitas et al. (UCI Grammatical Facial Expressions dataset). They describe what each expression means. This demo uses only face points, not head movement.")


# ================================================================ model helpers
def expr_zones(name):
    if not isinstance(PER_EXPR, dict):
        return None
    d = PER_EXPR.get(name)
    if not isinstance(d, dict) or set(d) != set(ZALL["overall"]):
        return None
    if not all(isinstance(v, (int, float)) for v in d.values()):
        return None
    s = float(sum(d.values())) or 1.0
    return {k: float(v) / s for k, v in d.items()}


def show_scores(img, use_clahe=False):
    pts = cv_pipeline.get_points(img, use_clahe=use_clahe)
    if pts is None:
        return None
    v = cv_pipeline.get_vector(pts)
    if v is None or not np.isfinite(v).all():
        return None
    raw = inference.scores_for_normalized(v[0])
    z = (raw - raw.mean()) / (raw.std() + 1e-9)
    bars = np.exp(z - z.max())
    bars = bars / bars.sum()
    order = np.argsort(raw)[::-1]
    names = inference._names
    ranked = [(names[i], float(bars[i])) for i in order]
    return ranked, float(raw[order[0]] - raw[order[1]]), raw


def rgb_css(c):
    return f"rgb({c[0]},{c[1]},{c[2]})"


def rank_html(ranked, rawd):
    top = max(b for _, b in ranked)
    rows = "".join(
        f'<div class="row"><span class="nm">{i+1}. {html.escape(nice(n))}</span>'
        f'<div class="track"><div class="fill" style="width:{100*b/top:.0f}%"></div></div>'
        f'<span class="val">{rawd[n]:.1f}</span></div>'
        for i, (n, b) in enumerate(ranked))
    return h(f'<div class="card"><div class="ttl">Expression scores (relative, not probabilities)</div>{rows}'
             '<span class="sub">Numbers are the model\'s raw scores. Higher means closer to that pattern. Compare the order, not the size.</span></div>')


def zone_rows(zw):
    col = overlay_view._rank_colours(zw)
    top = max(zw.values())
    return "".join(
        f'<div class="row"><span class="nm">{html.escape(z.replace("_"," ").capitalize())}</span>'
        f'<div class="track"><div class="fill" style="width:{100*v/top:.0f}%;background:{rgb_css(col[z])}"></div></div>'
        f'<span class="val">{100*v:.0f}%</span></div>'
        for z, v in sorted(zw.items(), key=lambda kv: -kv[1]))


def main_html(lines):
    return h('<div class="card main">' + "".join(f"<div>{t}</div>" for t in lines) + "</div>")


def analyze(img, use_clahe):
    hh, ww = img.shape[:2]
    k = MAX_SIDE / max(hh, ww)
    if k < 1:
        img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    seen = cv_pipeline.enhance(img) if use_clahe else None
    ann, msg, scores, zw = inference.predict_gfe(img, use_clahe=bool(use_clahe))
    if not scores:
        return {"ok": False, "ann": ann, "seen": seen, "orig": img, "main": main_html([html.escape(msg)])}
    ranked, margin, raw = show_scores(img, use_clahe=bool(use_clahe))
    rawd = {n: float(r) for n, r in zip(inference._names, raw)}
    top_name = ranked[0][0]
    lines = [f'Closest pattern in this model: <b>{html.escape(nice(top_name))}</b>',
             '<span class="sub">This model is weak: it was tested on signers it had never seen and often misses. Demo only.</span>']
    if use_clahe:
        p0 = cv_pipeline.get_points(img)
        p1 = cv_pipeline.get_points(img, use_clahe=True)
        if p0 is None and p1 is not None:
            lines.append('<span class="sub">The contrast fix helped: a face was found only with it.</span>')
        elif p0 is not None and p1 is not None:
            shift = 100 * float(np.linalg.norm(p0 - p1, axis=1).mean()) / float(np.ptp(p0[:, 0]))
            lines.append(f'<span class="sub">Contrast fix applied: landmarks moved {shift:.1f}% of the face width.'
                         + (" Small change." if shift < 1 else "") + "</span>")
    ez = expr_zones(top_name)
    desc = EXPRESSIONS.get(top_name)
    what = ""
    if desc:
        what = h(f'<div class="card"><div class="ttl">What is "{html.escape(desc[0])}"?</div><div>{html.escape(desc[1])}</div></div>')
    return {"ok": True, "ann": ann, "seen": seen, "orig": img, "main": main_html(lines), "what": what,
            "rank": rank_html(ranked, rawd), "top": nice(top_name),
            "zexpr": zone_rows(ez) if ez else "", "zall": zone_rows({k_: float(v_) for k_, v_ in zw.items()})}


def load_uploaded(f):
    im = ImageOps.exif_transpose(Image.open(f)).convert("RGB")   # phone photos: respect rotation
    return np.array(im)


def load_path(p):
    return cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)


# ================================================================ state
st.session_state.setdefault("page", "home")
st.session_state.setdefault("example", None)
st.session_state.setdefault("result", None)
st.session_state.setdefault("clahe", False)
st.session_state.setdefault("cam_on", False)

EX_DIR = BASE + "/examples"
EXAMPLES = [os.path.join(EX_DIR, f) for f in ("01_whq.jpg", "04_cond.jpg", "07_ynq.jpg")
            if os.path.exists(os.path.join(EX_DIR, f))]

nav()

# ================================================================ HOME
if st.session_state.page == "home":
    left, right = st.columns([1.15, 0.85], gap="large", vertical_alignment="center")
    with left:
        st.markdown(h("""<div class="hl"><span class="o">The Language</span><br>Beyond <span class="r">Words</span></div>
            <div class="lead">Your face carries grammar: In the Libras dataset, these are classified as Grammatical Facial Expressions (GFE).
            This demo reads a face photo and shows which regions of the face a model relies on.</div>"""), unsafe_allow_html=True)
        if st.button("EXPLORE  →", key="explore"):
            go_page("study")
    with right:
        st.markdown(hero_picture(), unsafe_allow_html=True)
    st.markdown('<div class="discover">Discover how facial expressions<br>shape meaning in Libras</div>', unsafe_allow_html=True)
    st.markdown(FOOT, unsafe_allow_html=True)

# ================================================================ THE STUDY
elif st.session_state.page == "study":
    c1, c2, c3 = st.columns([0.8, 1.65, 1.55], gap="large")
    with c1:
        if FACE:
            st.markdown(f'<img class="crop" src="{FACE}" alt="">', unsafe_allow_html=True)
        else:
            st.markdown(mesh_svg(), unsafe_allow_html=True)
    with c2:
        cards = ""
        for i, (key, (name, text, short)) in enumerate(EXPRESSIONS.items(), 1):
            cls = "ex last" if i == 9 else "ex"
            cards += f'<div class="{cls}"><div class="n">{i:02d}</div><div><b>{html.escape(name)}</b><span>{html.escape(short)}</span></div></div>'
        st.markdown(h(f'<div class="exg">{cards}</div>'), unsafe_allow_html=True)
        st.write("")
        if st.button("I'm ready — Try the Demo  →", key="ready"):
            go_page("demo")
    with c3:
        icon_up = ('<svg viewBox="0 0 64 64" fill="none" stroke="#fff" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">'
                   '<path d="M20 44h-2a10 10 0 0 1-1-20 14 14 0 0 1 27-3 11 11 0 0 1 2 23h-3"/><path d="M32 52V32M24 40l8-8 8 8"/></svg>')
        icon_scan = ('<svg viewBox="0 0 64 64" fill="none" stroke="#fff" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">'
                     '<path d="M8 20V12a4 4 0 0 1 4-4h8M44 8h8a4 4 0 0 1 4 4v8M56 44v8a4 4 0 0 1-4 4h-8M20 56h-8a4 4 0 0 1-4-4v-8"/>'
                     '<path d="M24 26v4M40 26v4M24 40c4 5 12 5 16 0"/></svg>')
        icon_bars = ('<svg viewBox="0 0 64 64" fill="none" stroke="#fff" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">'
                     '<path d="M10 54h44M16 54V34h8v20M28 54V20h8v34M40 54V10h8v44"/></svg>')
        st.markdown(h(f"""<div class="sec">How It works</div>
            <div class="flow">
              <div class="fc">{icon_up}<b>Upload a Photo</b><span>Provide a clear, front-facing photo.</span></div><div class="arrow">▶</div>
              <div class="fc">{icon_scan}<b>Detect Landmarks</b><span>Mapping 100 facial landmarks.</span></div><div class="arrow">▶</div>
              <div class="fc">{icon_bars}<b>Classify &amp; Analyze</b><span>See the results and which regions matter.</span></div>
            </div>"""), unsafe_allow_html=True)
        items = ["Face the camera", "Use even lighting", "Keep the entire head visible", "Avoid heavy shadows or masks",
                 "Upload a JPG or PNG image", "Only upload photos with consent"]
        lst = "".join(f"<div><i>{CHECK}</i>{t}</div>" for t in items)
        icon_head = ('<svg viewBox="0 0 120 120" width="120" height="120" fill="none" stroke="#fff" stroke-width="4" stroke-linecap="round" stroke-linejoin="round">'
                     '<path d="M10 34V16a6 6 0 0 1 6-6h18M86 10h18a6 6 0 0 1 6 6v18M110 86v18a6 6 0 0 1-6 6H86M34 110H16a6 6 0 0 1-6-6V86"/>'
                     '<path d="M44 52c0-12 7-20 16-20s16 8 16 20-7 24-16 24-16-12-16-24z"/><path d="M30 96c4-10 14-14 30-14s26 4 30 14"/>'
                     '<path d="M48 64l8 8 14-16" stroke="#5fd6c8"/></svg>')
        st.markdown(h(f"""<div class="sec" style="margin-top:34px">Before You Begin</div>
            <div style="display:flex;align-items:center;justify-content:space-between;gap:14px"><div class="chk">{lst}</div>{icon_head}</div>
            <div class="privacy"><svg viewBox="0 0 48 48"><rect x="9" y="21" width="30" height="22" rx="4" fill="#ff9a52"/>
              <path d="M16 21v-6a8 8 0 0 1 16 0v6" fill="none" stroke="#ff9a52" stroke-width="4"/><circle cx="24" cy="32" r="3" fill="#2a3f8f"/></svg>
              <div><b>Your Privacy Matters</b>Photos are not saved. This demo is for research purposes only.</div></div>"""), unsafe_allow_html=True)
    st.markdown(FOOT, unsafe_allow_html=True)

# ================================================================ DEMO
else:
    if st.session_state.clahe:
        st.markdown("<style>.st-key-clahe_btn button{background:#ff9650 !important; color:#0a0f24 !important;}</style>", unsafe_allow_html=True)
    r = st.session_state.result
    top_l, top_m, top_r = st.columns([1, 1.12, 1.45], gap="large")

    with top_l:
        with st.container(key="card_up"):
            st.markdown('<div class="ct"><span class="n">01</span><span class="t">Upload a Photo</span></div>', unsafe_allow_html=True)
            up = st.file_uploader("Drag and drop a photo here", type=["jpg", "jpeg", "png"], label_visibility="visible")
            if st.button("Use Camera" if not st.session_state.cam_on else "Close Camera", key="camera_btn", width="stretch"):
                st.session_state.cam_on = not st.session_state.cam_on
                st.rerun()
            cam = st.camera_input("Camera", label_visibility="collapsed") if st.session_state.cam_on else None
            if EXAMPLES:
                st.caption("Or try an example (frames from the public dataset)")
                cols = st.columns(len(EXAMPLES))
                for c, p in zip(cols, EXAMPLES):
                    with c:
                        st.image(p, width="stretch")
                        if st.button("Use", key="ex_" + os.path.basename(p), width="stretch"):
                            st.session_state.example = p
            source, label = None, None
            if cam is not None:
                source, label = ("upload", cam), "your camera photo"
            elif up is not None:
                source, label = ("upload", up), "your uploaded photo"
            elif st.session_state.example:
                source, label = ("example", st.session_state.example), "an example photo"
            if source:
                st.caption(f"Selected: {label}")
                st.image(load_uploaded(source[1]) if source[0] == "upload" else load_path(source[1]), width="stretch")
            tips = ["Face the camera, even lighting.", "Keep the entire head visible.", "Avoid heavy shadows or masks.",
                    "CLAHE improves contrast (experimental, off by default)."]
            st.markdown(h('<div class="tt">💡 Photo Tips:</div><div class="tips">' + "".join(f"<div>{CHECK_O}{t}</div>" for t in tips) + "</div>"),
                        unsafe_allow_html=True)
            if st.button("CLAHE: ON" if st.session_state.clahe else "CLAHE: OFF", key="clahe_btn", width="stretch"):
                st.session_state.clahe = not st.session_state.clahe
                st.rerun()
            go = st.button("Classify", key="classify", width="stretch", disabled=source is None)
            if source is None:
                st.caption("Choose a photo first.")
            if go and source:
                img = load_uploaded(source[1]) if source[0] == "upload" else load_path(source[1])
                with st.spinner("Analyzing..."):
                    st.session_state.result = analyze(img, st.session_state.clahe)
                st.rerun()

    with top_m:
        with st.container(key="card_lm"):
            st.markdown('<div class="ct"><span class="n">02</span><span class="t">Facial Landmark Detection</span></div>', unsafe_allow_html=True)
            if not r:
                st.markdown('<div class="dash">Your annotated face will appear here after you press <b>&nbsp;Classify</b>.</div>', unsafe_allow_html=True)
            else:
                st.image(r["ann"], width="stretch")
                st.markdown(h('<div class="legend"><span><i style="background:rgb(230,50,50)"></i>top third of zones</span>'
                              '<span><i style="background:rgb(245,170,30)"></i>middle third</span>'
                              '<span><i style="background:rgb(60,180,90)"></i>bottom third</span></div>'
                              '<span class="sub">Colour = RF zone importance (A+B), rank-based. The same for every photo.</span>'),
                            unsafe_allow_html=True)
        if r and r.get("seen") is not None:
            with st.container(key="card_cl"):
                st.markdown('<div class="ct"><span class="t">Contrast fix (CLAHE): what the detector saw</span></div>', unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                c1.image(r["orig"], caption="Original", width="stretch")
                c2.image(r["seen"], caption="After CLAHE", width="stretch")

    with top_r:
        with st.container(key="card_res"):
            st.markdown('<div class="ct"><span class="n">03</span><span class="t">Classification Results</span></div>', unsafe_allow_html=True)
            if not r:
                st.markdown('<div class="dash" style="min-height:360px">Scores for the nine expressions will appear here.</div>', unsafe_allow_html=True)
            else:
                st.markdown(r["main"], unsafe_allow_html=True)
                if r["ok"]:
                    st.markdown(r["what"], unsafe_allow_html=True)
                    st.markdown(r["rank"], unsafe_allow_html=True)

    bot_l, bot_r = st.columns([2.1, 1], gap="large")
    with bot_l:
        with st.container(key="card_z1"):
            ttl = f'Zone Importance for the predicted output' + (f' ("{html.escape(r["top"])}")' if r and r.get("ok") else "")
            st.markdown(f'<div class="ct"><span class="n">04</span><span class="t">{ttl}</span></div>', unsafe_allow_html=True)
            if r and r.get("ok") and r.get("zexpr"):
                st.markdown(h(r["zexpr"] + '<span class="sub">Changes with the expression the model picks. A model-level number, not a measurement of your face.</span>'),
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="dash" style="min-height:90px">Appears after classification.</div>', unsafe_allow_html=True)
        with st.container(key="card_z2"):
            st.markdown('<div class="ct"><span class="n">05</span><span class="t">Zone Importance Overall</span></div>', unsafe_allow_html=True)
            if r and r.get("ok"):
                st.markdown(h(r["zall"] + '<span class="sub">The same for every photo. These are the colours on the face overlay.</span>'), unsafe_allow_html=True)
            else:
                st.markdown('<div class="dash" style="min-height:90px">Appears after classification.</div>', unsafe_allow_html=True)
    with bot_r:
        with st.container(key="card_lim"):
            st.markdown(h("""<div class="ct"><span class="t" style="color:#fff;font-size:1.4rem">ⓘ Limitations</span></div>
                <ul class="lim"><li>Trained using landmark data from two Libras signers.</li>
                <li>2D analysis only: depth information is not considered.</li>
                <li>Photo landmarks are approximately matched to Kinect landmarks.</li>
                <li>The demo has not been evaluated on photos.</li>
                <li>Results may be inaccurate.</li></ul>"""), unsafe_allow_html=True)

    expressions_expander()
    st.markdown(FOOT, unsafe_allow_html=True)
