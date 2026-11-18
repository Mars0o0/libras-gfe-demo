import os, sys, json, html
import numpy as np
import cv2
import streamlit as st
from PIL import Image, ImageOps

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE + "/src")
st.set_page_config(page_title="Libras facial expressions", page_icon="🙂", layout="wide")

MAX_SIDE = 1280


@st.cache_resource(show_spinner="Loading the model (first visit takes a moment)...")
def load_models():
    import inference, cv_pipeline, overlay_view
    zall = json.load(open(BASE + "/models/zone_weights.json"))
    return inference, cv_pipeline, overlay_view, zall


inference, cv_pipeline, overlay_view, ZALL = load_models()
PER_EXPR = ZALL.get("per_expression")

# ---------------------------------------------------------------- content
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


# ---------------------------------------------------------------- style
CSS = """
.block-container {max-width: 1150px; padding-top: 1.5rem;}
#MainMenu, footer {visibility: hidden;}
.hero {text-align:center; padding:56px 12px 8px;}
.hero h1 {font-size:2.7rem; color:#f4f4f8; margin:0 0 6px;}
.hero .quote {font-size:1.35rem; color:#f5aa1e; font-style:italic; margin:22px auto; max-width:720px; line-height:1.5;}
.hero p {color:#b8bccf; max-width:640px; margin:0 auto; line-height:1.6;}
.steps {display:flex; gap:14px; flex-wrap:wrap; justify-content:center; margin:30px auto 8px; max-width:900px;}
.step {flex:1; min-width:210px; background:#171b2e; border:1px solid #262b45; border-radius:12px; padding:16px; color:#e6e8f2;}
.step b {color:#f5aa1e;}
.step span {display:block; color:#9aa0b8; font-size:0.9rem; margin-top:4px;}
.sh {display:flex; align-items:center; gap:10px; margin:6px 0 10px; color:#f4f4f8; font-weight:600; font-size:1.1rem;}
.sh i {font-style:normal; background:#f5aa1e; color:#0f1220; border-radius:50%; width:26px; height:26px; display:inline-flex; align-items:center; justify-content:center; font-size:0.9rem;}
.card {background:#171b2e; border:1px solid #262b45; border-radius:12px; padding:14px 16px; margin-top:12px; color:#e6e8f2;}
.card.main {font-size:1.05rem; line-height:1.5;}
.card .ttl {font-weight:600; margin-bottom:10px; color:#f4f4f8;}
.card .sub {color:#9aa0b8; font-size:0.85rem; margin-top:6px;}
.row {display:flex; align-items:center; gap:10px; margin:6px 0;}
.nm {width:130px; font-size:0.9rem;}
.track {flex:1; background:#232844; border-radius:6px; height:10px; overflow:hidden;}
.fill {height:100%; background:#f5aa1e; border-radius:6px;}
.val {width:48px; text-align:right; font-size:0.85rem; color:#b8bccf;}
.legend {display:flex; gap:14px; flex-wrap:wrap; color:#b8bccf; font-size:0.85rem; margin-top:8px;}
.legend i {display:inline-block; width:12px; height:12px; border-radius:3px; margin-right:5px; vertical-align:-1px;}
.foot {text-align:center; color:#8a8fa6; font-size:0.85rem; padding:26px 0 6px;}
.exp {margin:8px 0; color:#e6e8f2; line-height:1.5;}
.exp b {color:#f5aa1e;}
"""


def h(s):
    return " ".join(s.split())


st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------- model helpers
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
        return {"ok": False, "ann": ann, "seen": seen, "main": main_html([html.escape(msg)])}
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
    return {"ok": True, "ann": ann, "seen": seen, "main": main_html(lines),
            "rank": rank_html(ranked, rawd), "zones": zcards, "what": what}


def load_uploaded(f):
    im = ImageOps.exif_transpose(Image.open(f)).convert("RGB")   # phone photos: respect rotation
    return np.array(im)


def load_path(p):
    return cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------- state
st.session_state.setdefault("page", "welcome")
st.session_state.setdefault("example", None)
st.session_state.setdefault("result", None)

EX_DIR = BASE + "/examples"
EXAMPLES = [os.path.join(EX_DIR, f) for f in ("01_whq.jpg", "04_cond.jpg", "07_ynq.jpg")
            if os.path.exists(os.path.join(EX_DIR, f))]

# ---------------------------------------------------------------- welcome page
if st.session_state.page == "welcome":
    st.markdown(h("""<div class="hero"><h1>Libras facial expressions</h1>
        <div class="quote">&ldquo;A still face in sign language isn't calm&mdash;it's a grammatical error.&rdquo;</div>
        <p>In Libras (Brazilian Sign Language) the face carries grammar: questions, conditions, negation.
        Upload a face photo and see which facial regions a machine learning model weighs most.</p></div>"""),
                unsafe_allow_html=True)
    st.markdown(h("""<div class="steps">
        <div class="step"><b>1. Choose a photo</b><span>Upload your own face photo, or pick one of the examples.</span></div>
        <div class="step"><b>2. Analyze</b><span>The app finds the face, maps 100 landmark points and runs the model.</span></div>
        <div class="step"><b>3. Read the result</b><span>See the zones the model weighs, the ranking, and what each expression means.</span></div>
        </div>"""), unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 1, 1])
    with mid:
        if st.button("Try the demo", type="primary", width="stretch"):
            st.session_state.page = "demo"
            st.rerun()
    with st.expander("What do the nine expressions mean?"):
        for key, (name, text) in EXPRESSIONS.items():
            st.markdown(h(f'<div class="exp"><b>{html.escape(name)}</b>: {html.escape(text)}</div>'), unsafe_allow_html=True)
        st.caption("Descriptions follow Freitas et al. (UCI Grammatical Facial Expressions dataset). They describe what each expression means. This demo uses only face points, not head movement.")
    st.markdown('<div class="foot">Demo only, 2D only. Photos are not saved by this app. Results may be inaccurate.</div>',
                unsafe_allow_html=True)

# ---------------------------------------------------------------- demo page
else:
    top_l, top_r = st.columns([4, 1])
    with top_l:
        st.markdown('<h2 style="margin:0;color:#f4f4f8">Libras facial expressions</h2>', unsafe_allow_html=True)
    with top_r:
        if st.button("Back to start", width="stretch"):
            st.session_state.page = "welcome"
            st.rerun()

    left, right = st.columns([1, 1], gap="large")
    with left:
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
            st.image(load_uploaded(source[1]) if source[0] == "upload" else load_path(source[1]),
                     width="stretch")

        st.markdown('<div class="sh"><i>2</i> Analyze</div>', unsafe_allow_html=True)
        clahe = st.checkbox("Contrast fix (CLAHE): try it for dark or low-contrast photos", value=False)
        go = st.button("Analyze", type="primary", width="stretch", disabled=source is None)
        if source is None:
            st.caption("Choose a photo first.")
        if go and source:
            img = load_uploaded(source[1]) if source[0] == "upload" else load_path(source[1])
            with st.spinner("Analyzing..."):
                st.session_state.result = analyze(img, clahe)

    with right:
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
            if r["ok"]:
                st.markdown(r["what"], unsafe_allow_html=True)
                st.markdown(r["rank"], unsafe_allow_html=True)
                st.markdown(r["zones"], unsafe_allow_html=True)
            if r.get("seen") is not None:
                with st.expander("What the detector saw (contrast fix on)"):
                    st.image(r["seen"], width="stretch")

    with st.expander("What do the nine expressions mean?"):
        for key, (name, text) in EXPRESSIONS.items():
            st.markdown(h(f'<div class="exp"><b>{html.escape(name)}</b>: {html.escape(text)}</div>'), unsafe_allow_html=True)
        st.caption("Descriptions follow Freitas et al. (UCI Grammatical Facial Expressions dataset). They describe what each expression means. This demo uses only face points, not head movement.")
    st.markdown('<div class="foot">Demo only, 2D only. Photos are not saved by this app. Results may be inaccurate.</div>',
                unsafe_allow_html=True)
