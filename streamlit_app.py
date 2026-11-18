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
    "affirmative": ("Affirmative", "Used when making a statement, to assert that something is true. The dataset paper calls it \"assertion\". Head nodding (up and down) goes with it."),
    "conditional": ("Conditional", "Marks the \"if\" part of a sentence: a condition that must hold for the main clause."),
    "doubt_question": ("Doubt question", "Looks like a question but does not expect an answer. It draws attention to the information that follows. The paper links it with lowered eyebrows and a compressed mouth."),
    "emphasis": ("Emphasis", "Highlights the new or important information in what is being said. The paper calls it \"focus\"."),
    "negative": ("Negative", "Marks a negative sentence (\"not\", \"no\"). The paper links it with lowered eyebrows, a downturned mouth and a side-to-side head shake."),
    "relative": ("Relative", "Marks a clause that gives more information about something, as in \"the person who...\"."),
    "topics": ("Topics", "Moves the topic of the sentence to the front, to say what it is about before commenting on it."),
    "wh_question": ("WH question", "Asks who, what, when, where, how or why."),
    "yn_question": ("Yes/no question", "Asks a question that can be answered with yes or no."),
}


def nice(name):
    return EXPRESSIONS.get(name, (name.replace("_", " ").capitalize(), ""))[0]


def h(s):
    return " ".join(s.split())


# ================================================================ look
def bg_url():
    p = BASE + "/assets/bg.jpg"
    if os.path.exists(p):
        return "url(data:image/jpeg;base64," + base64.b64encode(open(p, "rb").read()).decode() + ")"
    return "none"


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&display=swap');
:root{--amber:#ffb454;--blue:#6ea8ff;--green:#4ade80;--pink:#ff6b8a;--lav:#a9a9d6;--ink:#f3f5ff;--mute:#a9b0cc;--line:rgba(255,255,255,.10);}
.stApp{background:
 radial-gradient(900px 520px at 10% 6%, rgba(255,180,84,.17), transparent 62%),
 radial-gradient(820px 620px at 92% 12%, rgba(110,168,255,.20), transparent 60%),
 radial-gradient(1100px 700px at 50% 112%, rgba(74,222,128,.08), transparent 62%),
 linear-gradient(180deg, rgba(7,10,28,.50), rgba(7,10,28,.78)),
 BGURL center/cover fixed #0b1126 !important;
 font-family:'Inter',system-ui,sans-serif; color:var(--ink);}
[data-testid="stHeader"]{background:transparent;}
.block-container{max-width:1180px; padding-top:1.1rem; padding-bottom:3rem;}
#MainMenu, footer{visibility:hidden;}
h1,h2,h3{font-family:'Space Grotesk','Inter',sans-serif; letter-spacing:-.01em;}
.nav{display:flex; justify-content:space-between; align-items:center; padding:4px 0 22px; gap:12px; flex-wrap:wrap;}
.brand{display:flex; gap:10px; align-items:center; font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.1rem; color:var(--ink);}
.logo{width:26px; height:26px;}
.badge{border:1px solid var(--line); border-radius:999px; padding:7px 15px; color:var(--mute); font-size:.84rem; background:rgba(255,255,255,.04); backdrop-filter:blur(6px);}
.eyebrow{letter-spacing:.2em; color:var(--amber); font-size:.78rem; font-weight:600; margin:18px 0 14px;}
.hl{font-family:'Space Grotesk',sans-serif; font-size:clamp(2.3rem,4.6vw,3.9rem); line-height:1.05; font-weight:700; color:var(--ink); margin:0 0 18px;}
.grad{background:linear-gradient(90deg,#ffb454,#ff8f6b,#ffd08a,#ffb454); background-size:200% auto; -webkit-background-clip:text; background-clip:text; color:transparent; animation:shine 7s linear infinite;}
@keyframes shine{to{background-position:200% center;}}
.lead{color:#c9cfe6; font-size:1.08rem; line-height:1.65; max-width:540px; margin-bottom:22px;}
.note{color:var(--mute); font-size:.88rem; margin-top:12px;}
.const{width:100%; max-width:520px; margin:0 auto; display:block; animation:floaty 8s ease-in-out infinite;}
@keyframes floaty{0%,100%{transform:translateY(0);}50%{transform:translateY(-8px);}}
.d{animation:tw 3.4s ease-in-out infinite;}
@keyframes tw{0%,100%{opacity:.5;}50%{opacity:1;}}
.legend{display:flex; gap:14px; flex-wrap:wrap; color:var(--mute); font-size:.84rem; margin-top:10px; justify-content:center;}
.legend i{display:inline-block; width:11px; height:11px; border-radius:50%; margin-right:6px; vertical-align:-1px;}
.sect{font-family:'Space Grotesk',sans-serif; font-size:1.45rem; font-weight:700; margin:44px 0 16px; color:var(--ink);}
.how{display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:16px;}
.hc{background:rgba(18,24,54,.55); border:1px solid var(--line); border-radius:18px; padding:22px; backdrop-filter:blur(10px); transition:transform .25s ease, border-color .25s ease, box-shadow .25s ease;}
.hc:hover{transform:translateY(-5px); border-color:rgba(255,180,84,.45); box-shadow:0 14px 40px rgba(0,0,0,.35);}
.hc .n{font-family:'Space Grotesk',sans-serif; font-size:1.9rem; font-weight:700;}
.hc b{display:block; font-size:1.08rem; margin:8px 0 6px; color:var(--ink);}
.hc span{color:#b9c0da; line-height:1.55; font-size:.95rem;}
.pills{display:flex; flex-wrap:wrap; gap:10px; margin-top:6px;}
.pill{position:relative; border:1px solid var(--line); border-radius:999px; padding:8px 17px; background:rgba(255,255,255,.05); color:var(--ink); font-size:.93rem; cursor:default; transition:all .2s ease;}
.pill:hover{border-color:var(--amber); transform:translateY(-2px); background:rgba(255,180,84,.12);}
.pill::after{content:attr(data-tip); position:absolute; left:50%; bottom:calc(100% + 10px); transform:translateX(-50%) translateY(4px); width:270px; background:#11173a; border:1px solid var(--line); border-radius:12px; padding:10px 12px; font-size:.82rem; line-height:1.45; color:var(--ink); opacity:0; pointer-events:none; transition:all .18s ease; z-index:60; white-space:normal; box-shadow:0 12px 36px rgba(0,0,0,.45);}
.pill:hover::after{opacity:1; transform:translateX(-50%) translateY(0);}
.foot{color:#8a91ad; font-size:.84rem; padding:34px 0 4px; border-top:1px solid var(--line); margin-top:46px; line-height:1.6;}
.sh{display:flex; align-items:center; gap:11px; margin:2px 0 12px; color:var(--ink); font-weight:600; font-size:1.12rem; font-family:'Space Grotesk',sans-serif;}
.sh i{font-style:normal; background:linear-gradient(135deg,#ffb454,#ff8f6b); color:#1a1020; border-radius:50%; width:28px; height:28px; display:inline-flex; align-items:center; justify-content:center; font-size:.92rem; box-shadow:0 0 18px rgba(255,180,84,.45);}
.title2{font-family:'Space Grotesk',sans-serif; font-size:clamp(1.8rem,3.2vw,2.6rem); font-weight:700; margin:0 0 4px;}
.sub2{color:var(--mute); margin-bottom:20px;}
.card{background:rgba(14,19,46,.62); border:1px solid var(--line); border-radius:14px; padding:14px 16px; margin-top:12px; color:var(--ink); animation:rise .55s ease both; backdrop-filter:blur(6px);}
.card.main{font-size:1.06rem; line-height:1.5; border-color:rgba(255,180,84,.38); background:linear-gradient(135deg, rgba(255,180,84,.12), rgba(14,19,46,.62) 55%);}
.card b{color:var(--amber);}
.card .ttl{font-weight:600; margin-bottom:10px; color:var(--ink);}
.card .sub{color:var(--mute); font-size:.85rem; margin-top:6px; line-height:1.5;}
@keyframes rise{from{opacity:0; transform:translateY(12px);}}
.row{display:flex; align-items:center; gap:10px; margin:7px 0;}
.nm{width:132px; font-size:.9rem; color:#dfe3f5;}
.track{flex:1; background:rgba(255,255,255,.08); border-radius:6px; height:10px; overflow:hidden;}
.fill{height:100%; background:linear-gradient(90deg,#ffb454,#ff8f6b); border-radius:6px; animation:grow 1.1s cubic-bezier(.2,.8,.2,1) both;}
@keyframes grow{from{width:0 !important;}}
.val{width:48px; text-align:right; font-size:.85rem; color:var(--mute);}
.exp{margin:9px 0; color:#dfe3f5; line-height:1.55;}
.exp b{color:var(--amber);}
.st-key-panel_left, .st-key-panel_right{background:rgba(16,22,52,.52); border:1px solid var(--line); border-radius:20px; padding:20px 22px; backdrop-filter:blur(12px); box-shadow:0 18px 50px rgba(0,0,0,.35);}
.st-key-cta button{background:#fff !important; color:#10163a !important; border:0 !important; border-radius:999px !important; padding:.7rem 1.7rem !important; font-weight:600 !important; font-size:1.02rem !important; box-shadow:0 8px 30px rgba(255,255,255,.18); transition:all .22s ease;}
.st-key-cta button:hover{transform:translateY(-2px) scale(1.03); box-shadow:0 12px 40px rgba(255,180,84,.45);}
.st-key-analyze button{background:linear-gradient(135deg,#ffb454,#ff8f6b) !important; color:#1a1020 !important; border:0 !important; border-radius:12px !important; font-weight:700 !important; padding:.6rem 1rem !important; box-shadow:0 8px 28px rgba(255,143,107,.35); transition:all .2s ease;}
.st-key-analyze button:hover{transform:translateY(-2px); box-shadow:0 12px 34px rgba(255,143,107,.55);}
.st-key-analyze button:disabled{opacity:.45; box-shadow:none;}
.stButton button{border-radius:12px;}
[data-testid="stFileUploaderDropzone"]{background:rgba(255,255,255,.04); border:1.5px dashed rgba(255,180,84,.55); border-radius:14px;}
[data-testid="stExpander"]{border:1px solid var(--line) !important; border-radius:14px !important; background:rgba(255,255,255,.035);}
[data-testid="stImage"] img{border-radius:14px; box-shadow:0 10px 38px rgba(0,0,0,.45), 0 0 0 1px rgba(255,255,255,.09);}
[data-testid="stCaptionContainer"]{color:var(--mute);}
"""

st.markdown("<style>" + CSS.replace("BGURL", bg_url()) + "</style>", unsafe_allow_html=True)

ZONE_COL = {"eyebrows": "#ffb454", "eyes": "#6ea8ff", "nose": "#4ade80", "mouth": "#ff6b8a", "face_contour": "#a9a9d6"}


def constellation_svg():
    """A drawn face made of 100 dots (30 eyebrows, 18 eyes, 13 nose, 20 mouth, 19 contour). Decorative only."""
    pts = []                                              # (x, y, zone)
    for cx in (170, 310):                                 # eyebrows: 10 on the brow + 5 above, per side
        for i in range(10):
            t = i / 9
            pts.append((cx - 48 + 96 * t, 96 - 16 * math.sin(math.pi * t), "eyebrows"))
        for i in range(5):
            t = i / 4
            pts.append((cx - 40 + 80 * t, 70 - 14 * math.sin(math.pi * t), "eyebrows"))
    for cx in (175, 305):                                 # eyes: 8 around + 1 centre, per side
        for i in range(8):
            a = 2 * math.pi * i / 8
            pts.append((cx + 34 * math.cos(a), 142 + 11 * math.sin(a), "eyes"))
        pts.append((cx, 142, "eyes"))
    for i in range(4):                                    # nose: bridge, base, tip
        pts.append((240, 150 + i * 16, "nose"))
    for i in range(8):
        t = i / 7
        pts.append((206 + 68 * t, 214 + 9 * math.sin(math.pi * t), "nose"))
    pts.append((240, 206, "nose"))
    for i in range(12):                                   # mouth: 12 outer + 8 inner
        a = 2 * math.pi * i / 12
        pts.append((240 + 50 * math.cos(a), 276 + 20 * math.sin(a), "mouth"))
    for i in range(8):
        a = 2 * math.pi * i / 8
        pts.append((240 + 30 * math.cos(a), 276 + 7 * math.sin(a), "mouth"))
    for i in range(19):                                   # contour: a U shape (jaw only)
        th = math.pi * (0.03 + 0.94 * i / 18)
        pts.append((240 + 152 * math.cos(th), 168 + 178 * math.sin(th), "face_contour"))
    dots = "".join(
        f'<circle class="d" cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="{ZONE_COL[z]}" '
        f'style="animation-delay:{(k * 0.37) % 3.4:.2f}s;filter:drop-shadow(0 0 5px {ZONE_COL[z]})"/>'
        for k, (x, y, z) in enumerate(pts))
    return h(f'<svg class="const" viewBox="0 0 480 360" xmlns="http://www.w3.org/2000/svg">{dots}</svg>')


LOGO = h("""<svg class="logo" viewBox="0 0 26 26"><circle cx="6" cy="7" r="2.6" fill="#ffb454"/><circle cx="13" cy="5" r="2.6" fill="#6ea8ff"/>
<circle cx="20" cy="8" r="2.6" fill="#4ade80"/><circle cx="8" cy="17" r="2.6" fill="#ff6b8a"/><circle cx="18" cy="18" r="2.6" fill="#a9a9d6"/></svg>""")
NAV = h(f'<div class="nav"><div class="brand">{LOGO} Libras GFE Lab</div><div class="badge">Research demo &middot; grammatical facial expressions</div></div>')
FOOT = h("""<div class="foot">Demo only, 2D only. Photos are not saved by this app. Results may be inaccurate.
Built on landmark data from two Libras signers (Freitas et al., 2014). This is not an interpreter and does not read emotion: it looks at grammar on the face.</div>""")


def expressions_expander():
    with st.expander("What do the nine expressions mean?"):
        for key, (name, text) in EXPRESSIONS.items():
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
    return h(f'<div class="card"><div class="ttl">Relative ranking (not probabilities)</div>{rows}'
             '<div class="sub">Numbers are the model\'s raw scores. Higher means closer to that pattern. Compare the order, not the size.</div></div>')


def zones_html(zw, title, sub):
    col = overlay_view._rank_colours(zw)
    top = max(zw.values())
    rows = "".join(
        f'<div class="row"><span class="nm">{html.escape(z.replace("_"," ").capitalize())}</span>'
        f'<div class="track"><div class="fill" style="width:{100*v/top:.0f}%;background:{rgb_css(col[z])}"></div></div>'
        f'<span class="val">{100*v:.0f}%</span></div>'
        for z, v in sorted(zw.items(), key=lambda kv: -kv[1]))
    return h(f'<div class="card"><div class="ttl">{title}</div>{rows}<div class="sub">{sub}</div></div>')


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
    zcards = ""
    ez = expr_zones(top_name)
    if ez:
        zcards += zones_html(ez, f'Zone importance for "{html.escape(nice(top_name))}" (RF, A+B)',
                             "Changes with the expression the model picks. Still a model-level number, not a measurement of your face.")
    zcards += zones_html({k_: float(v_) for k_, v_ in zw.items()}, "Zone importance, all expressions (RF, A+B)",
                         "The same for every photo. These are the colours on the overlay.")
    desc = EXPRESSIONS.get(top_name)
    what = ""
    if desc:
        what = h(f'<div class="card"><div class="ttl">What is "{html.escape(desc[0])}"?</div>'
                 f'<div>{html.escape(desc[1])}</div></div>')
    return {"ok": True, "ann": ann, "seen": seen, "orig": img, "main": main_html(lines),
            "rank": rank_html(ranked, rawd), "zones": zcards, "what": what}


def load_uploaded(f):
    im = ImageOps.exif_transpose(Image.open(f)).convert("RGB")   # phone photos: respect rotation
    return np.array(im)


def load_path(p):
    return cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)


# ================================================================ state
st.session_state.setdefault("page", "welcome")
st.session_state.setdefault("example", None)
st.session_state.setdefault("result", None)

EX_DIR = BASE + "/examples"
EXAMPLES = [os.path.join(EX_DIR, f) for f in ("01_whq.jpg", "04_cond.jpg", "07_ynq.jpg")
            if os.path.exists(os.path.join(EX_DIR, f))]

# ================================================================ welcome page
if st.session_state.page == "welcome":
    st.markdown(NAV, unsafe_allow_html=True)
    left, right = st.columns([1.08, 0.92], gap="large")
    with left:
        st.markdown(h("""<div class="eyebrow">LIBRAS &middot; NINE GRAMMATICAL EXPRESSIONS</div>
            <div class="hl">A still face in sign language isn't calm <span class="grad">&mdash; it's a grammatical error.</span></div>
            <div class="lead">In Libras, the eyebrows, eyes, mouth and head carry grammar: questions, conditions, negation.
            This demo reads a face photo and shows which regions of the face a model relies on.</div>"""),
                    unsafe_allow_html=True)
        if st.button("Try the demo  →", key="cta"):
            st.session_state.page = "demo"
            st.rerun()
        st.markdown('<div class="note">Bring a face photo of someone who agreed to be in it.</div>', unsafe_allow_html=True)
    with right:
        st.markdown(constellation_svg(), unsafe_allow_html=True)
        st.markdown(h("""<div class="legend"><span><i style="background:#ffb454"></i>Eyebrows &middot; 30</span>
            <span><i style="background:#6ea8ff"></i>Eyes &middot; 18</span><span><i style="background:#4ade80"></i>Nose &middot; 13</span>
            <span><i style="background:#ff6b8a"></i>Mouth &middot; 20</span><span><i style="background:#a9a9d6"></i>Face contour &middot; 19</span></div>"""),
                    unsafe_allow_html=True)

    st.markdown('<div class="sect">How it works</div>', unsafe_allow_html=True)
    st.markdown(h("""<div class="how">
        <div class="hc"><div class="n" style="color:#6ea8ff">01</div><b>Upload a photo</b><span>A front-facing face, fully in the frame, with some background around the head.</span></div>
        <div class="hc"><div class="n" style="color:#ffb454">02</div><b>We map 100 landmarks</b><span>MediaPipe finds the face. We keep 100 points that line up, roughly, with the Kinect points the model learned from.</span></div>
        <div class="hc"><div class="n" style="color:#4ade80">03</div><b>See the regions that matter</b><span>The 100 points become 200 numbers for the model. The page shows the scores and which facial zones carry the most weight.</span></div>
        </div>"""), unsafe_allow_html=True)

    st.markdown('<div class="sect">The nine expressions in the data</div>', unsafe_allow_html=True)
    pills = "".join(f'<span class="pill" data-tip="{html.escape(text, quote=True)}">{html.escape(name)}</span>'
                    for name, text in EXPRESSIONS.values())
    st.markdown(h(f'<div class="pills">{pills}</div>'), unsafe_allow_html=True)
    st.caption("Hover over (or tap) a pill to see what it means.")
    expressions_expander()
    st.markdown(FOOT, unsafe_allow_html=True)

# ================================================================ demo page
else:
    nav_l, nav_r = st.columns([4, 1])
    with nav_l:
        st.markdown(NAV, unsafe_allow_html=True)
    with nav_r:
        if st.button("Back to start", width="stretch", key="back"):
            st.session_state.page = "welcome"
            st.rerun()
    st.markdown(h("""<div class="title2">Try the <span class="grad">demo</span></div>
        <div class="sub2">Choose a photo, press Analyze, and read which facial zones the model weighs.</div>"""),
                unsafe_allow_html=True)

    left, right = st.columns([1, 1], gap="large")
    with left:
        with st.container(key="panel_left"):
            st.markdown('<div class="sh"><i>1</i> Choose a photo</div>', unsafe_allow_html=True)
            up = st.file_uploader("Upload a face photo", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
            with st.expander("Or take a photo with your camera"):
                cam = st.camera_input("Camera", label_visibility="collapsed")
            if EXAMPLES:
                st.caption("Or try an example (frames from the public dataset)")
                cols = st.columns(len(EXAMPLES))
                for c, p in zip(cols, EXAMPLES):
                    with c:
                        st.image(p, width="stretch")
                        if st.button("Use this", key="ex_" + os.path.basename(p), width="stretch"):
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

            st.markdown('<div class="sh" style="margin-top:14px"><i>2</i> Analyze</div>', unsafe_allow_html=True)
            clahe = st.checkbox("Contrast fix (CLAHE): try it for dark or low-contrast photos", value=False)
            go = st.button("Analyze", key="analyze", width="stretch", disabled=source is None)
            if source is None:
                st.caption("Choose a photo first.")
            if go and source:
                img = load_uploaded(source[1]) if source[0] == "upload" else load_path(source[1])
                with st.spinner("Analyzing..."):
                    st.session_state.result = analyze(img, clahe)

    with right:
        with st.container(key="panel_right"):
            st.markdown('<div class="sh"><i>3</i> Read the result</div>', unsafe_allow_html=True)
            r = st.session_state.result
            if not r:
                st.markdown(h('<div class="card">Your result will appear here after you press <b>Analyze</b>.</div>'),
                            unsafe_allow_html=True)
            else:
                st.image(r["ann"], width="stretch", caption="Zones (colour = RF zone importance (A+B))")
                st.markdown(h('<div class="legend"><span><i style="background:rgb(230,50,50)"></i>top third of zones</span>'
                              '<span><i style="background:rgb(245,170,30)"></i>middle third</span>'
                              '<span><i style="background:rgb(60,180,90)"></i>bottom third</span></div>'),
                            unsafe_allow_html=True)
                st.markdown(r["main"], unsafe_allow_html=True)
                if r.get("seen") is not None:
                    st.markdown('<div class="sh" style="margin-top:14px">Contrast fix: what the detector saw</div>', unsafe_allow_html=True)
                    c1, c2 = st.columns(2)
                    c1.image(r["orig"], caption="Original", width="stretch")
                    c2.image(r["seen"], caption="After CLAHE", width="stretch")
                if r["ok"]:
                    st.markdown(r["what"], unsafe_allow_html=True)
                    st.markdown(r["rank"], unsafe_allow_html=True)
                    st.markdown(r["zones"], unsafe_allow_html=True)

    st.write("")
    expressions_expander()
    st.markdown(FOOT, unsafe_allow_html=True)
