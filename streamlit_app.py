import os, sys, json, html, base64
import numpy as np
import cv2
import streamlit as st
from PIL import Image, ImageOps

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE + "/src")
st.set_page_config(page_title="Libras GFE Lab", page_icon="🙂", layout="wide", initial_sidebar_state="collapsed")

MAX_SIDE = 1280
LIVE_URL = "https://libras-gfe-demo-evvt6cxhgmxhmhxfsxudfx.streamlit.app/"


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


def esc(s):
    return html.escape(str(s))


def data_uri(rel):
    p = os.path.join(BASE, rel)
    if os.path.exists(p):
        return "data:image/jpeg;base64," + base64.b64encode(open(p, "rb").read()).decode()
    return ""


# ================================================================ look (copied from Libras_GFE_Lab.html)
CSS = r'''@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Space+Mono:wght@400;700&display=swap');
:root{--bg:#060B1A;--card:#101C3D;--line:#25396B;--tx:#F4F7FF;--mu:#9FB0D3;--or:#FF8F3F;--teal:#2DD4BF;--bl:#4C7BF4;--red:#FF5A5F}
html,body,.stApp{font-family:"Plus Jakarta Sans",system-ui,sans-serif !important;color:var(--tx)}
.stApp{background:radial-gradient(900px 500px at 85% -5%,#2147b055,transparent),radial-gradient(700px 500px at 0% 60%,#1a3a8a30,transparent),linear-gradient(#060B1A,#0A1632 55%,#0E2050) fixed !important}
[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stDecoration"],#MainMenu,footer{display:none !important}
[data-testid="stMainBlockContainer"],.block-container{padding:0 !important;max-width:none !important}
[data-testid="stMainBlockContainer"] [data-testid="stVerticalBlock"]{gap:0}
[data-testid="stMain"]{overflow-x:hidden}
.lx{font-family:"Plus Jakarta Sans",system-ui,sans-serif}
.lx div,.lx span,.lx b,.lx em,.lx i,.lx small{font-family:"Plus Jakarta Sans",system-ui,sans-serif}
/* header */
.st-key-hdr{position:sticky;top:0;z-index:50;padding:14px 5vw;background:#060B1Acc;backdrop-filter:blur(12px);border-bottom:1px solid #ffffff12;gap:0 !important}
.st-key-hdr [data-testid="stHorizontalBlock"]{flex-wrap:nowrap !important;align-items:center;gap:0 !important}
.st-key-hdr [data-testid="stColumn"]{flex:0 0 auto !important;width:auto !important;min-width:0 !important}
.st-key-hdr [data-testid="stColumn"]:nth-child(6){flex:1 1 auto !important}
.logo{display:flex;align-items:center;gap:12px}
.mk{width:44px;height:44px;flex:none;filter:drop-shadow(0 6px 14px #2447A899)}
.wm{display:flex;flex-direction:column;line-height:1}
.wm b{font:800 25px "Plus Jakarta Sans",sans-serif;letter-spacing:-.03em;color:#fff}
.wm small{font:700 10px "Space Mono",monospace;letter-spacing:.36em;color:var(--or);margin-top:6px}
[class*="st-key-nav_"] button{background:transparent !important;border:0 !important;border-bottom:3px solid transparent !important;border-radius:0 !important;box-shadow:none !important;color:#cfd9f5 !important;font:700 15px "Space Mono",monospace !important;padding:6px 0 !important;min-height:0 !important;height:auto !important}
[class*="st-key-nav_"] button p{font:700 15px "Space Mono",monospace !important;color:inherit !important;margin:0 !important}
[class*="st-key-nav_"] button:hover{color:var(--or) !important}
.st-key-nav_home{margin-left:48px;padding-left:28px;border-left:2px solid var(--or);margin-right:28px}
.st-key-nav_study,.st-key-nav_demo{margin-right:28px}
.pill{display:flex;align-items:center;gap:8px;font:600 12px "Plus Jakarta Sans",system-ui,sans-serif;padding:8px 14px;border:1px solid var(--line);border-radius:99px;background:#0f1d44;color:#fff;width:max-content;margin-left:auto}
.pill svg{width:16px;height:16px}
/* buttons */
[class*="st-key-bl_"] button,[class*="st-key-or_"] button,[class*="st-key-gh_"] button,[class*="st-key-ol_"] button,[class*="st-key-dk_"] button{display:inline-flex;align-items:center;justify-content:center;gap:12px;font:800 16px "Plus Jakarta Sans",sans-serif !important;letter-spacing:.04em;padding:16px 28px !important;border-radius:99px !important;border:0 !important;color:#fff !important;background:linear-gradient(135deg,#3B63D6,#2A4AA8) !important;box-shadow:0 10px 30px -10px #3B63D6,inset 0 1px #ffffff30 !important;transition:.2s;min-height:0 !important;height:auto !important}
[class*="st-key-bl_"] button p,[class*="st-key-or_"] button p,[class*="st-key-gh_"] button p,[class*="st-key-ol_"] button p,[class*="st-key-dk_"] button p{font:inherit !important;color:inherit !important;margin:0 !important}
[class*="st-key-bl_"] button:hover,[class*="st-key-or_"] button:hover,[class*="st-key-gh_"] button:hover,[class*="st-key-dk_"] button:hover{transform:translateY(-2px)}
[class*="st-key-or_"] button{background:linear-gradient(135deg,#FF9A4D,#F27A1F) !important;color:#1a0c00 !important;box-shadow:0 12px 30px -10px #FF8F3F,inset 0 1px #ffffff55 !important}
[class*="st-key-gh_"] button{background:#1b2c5e !important;box-shadow:none !important;border:1px solid var(--line) !important}
[class*="st-key-sm_"] button{padding:11px 22px !important;font-size:14px !important}
[class*="st-key-sm_"] button p{font-size:14px !important}
[class*="st-key-dk_"] button{background:#0B1220 !important;color:#fff !important;box-shadow:0 12px 30px -12px #000 !important}
[class*="st-key-ol_"] button{background:transparent !important;color:#1a0c00 !important;border:2px solid #1a0c00 !important;box-shadow:none !important}
[class*="st-key-on_"] button{border:1px solid var(--teal) !important;color:var(--teal) !important}
[class*="st-key-bl_"],[class*="st-key-or_"],[class*="st-key-gh_"],[class*="st-key-ol_"],[class*="st-key-dk_"]{width:fit-content}
button:disabled{opacity:.45 !important}
/* hero */
.st-key-hero{position:relative;overflow:hidden;padding:7vh 5vw 150px;min-height:78vh}
.st-key-hero [data-testid="stElementContainer"]{position:static}
.hero-face{position:absolute;right:0;top:0;bottom:0;width:min(48vw,640px);background:var(--f1) center/cover;-webkit-mask-image:radial-gradient(ellipse 80% 85% at 62% 48%,#000 55%,transparent 100%);mask-image:radial-gradient(ellipse 80% 85% at 62% 48%,#000 55%,transparent 100%)}
.hero-tx{position:relative;max-width:min(720px,56%)}
.h1{font-size:clamp(46px,7.4vw,104px);line-height:.98;margin:0 0 28px;font-weight:800;letter-spacing:-.03em;color:#fff}
.h1 .a,.big .a{background:linear-gradient(90deg,#FF8F3F,#FFB56E);-webkit-background-clip:text;background-clip:text;color:transparent}
.h1 .b{color:var(--red)}
.lead{font-size:clamp(16px,1.5vw,20px);line-height:1.65;color:#c5d2f0;margin:0 0 36px;max-width:560px}
.st-key-hero [class*="st-key-bl_"]{position:relative}
.panel{position:relative;z-index:2;margin-top:-60px;background:#E9EEF8;color:#0B1220;border-radius:44px 44px 0 0;padding:44px 5vw 56px;display:flex;flex-wrap:wrap;gap:28px;align-items:center;justify-content:space-between}
.panel .ph{font-size:clamp(26px,3.4vw,44px);line-height:1.08;margin:0;letter-spacing:-.02em;max-width:640px;font-weight:800;color:#0B1220}
.stats{display:flex;gap:14px;flex-wrap:wrap}
.stats>div{background:#fff;border-radius:18px;padding:14px 20px;box-shadow:0 8px 24px -14px #0b1b4a88}
.stats b{display:block;font-size:28px;color:#2A4AA8}.stats span{font-size:13px;color:#51608a}
/* study */
.study{display:grid;grid-template-columns:230px 1fr 440px;gap:30px;padding:30px 5vw 20px 0}
.sface{background:var(--f2) right center/cover;-webkit-mask-image:linear-gradient(90deg,#000 55%,transparent);mask-image:linear-gradient(90deg,#000 55%,transparent);min-height:640px}
.t{font-size:clamp(26px,2.6vw,34px);margin:0 0 18px;font-weight:600;display:flex;align-items:center;gap:20px;letter-spacing:-.01em;color:#fff}
.t::after{content:"";flex:1;height:1px;background:linear-gradient(90deg,#ffffff70,transparent)}
.card,[class*="st-key-card_"]{background:linear-gradient(160deg,#15264F,#0F1B3C);border:1px solid var(--line);border-radius:20px;box-shadow:0 24px 40px -24px #000c,inset 0 1px #ffffff12}
.gfe{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.g{display:flex;gap:14px;align-items:center;padding:16px 18px;transition:.2s}
.g:hover{border-color:var(--or);transform:translateY(-2px)}
.g i{font:700 15px "Space Mono",monospace;font-style:normal;color:var(--or);min-width:40px;height:40px;display:grid;place-items:center;border-radius:12px;background:#FF8F3F1c}
.g b{display:block;font-size:16px;color:#fff}.g span{font-size:12.5px;color:var(--mu);line-height:1.45;display:block;margin-top:2px}
.g.w{grid-column:1/-1;justify-self:center;width:calc(50% - 8px)}
.steps{display:grid;grid-template-columns:repeat(3,1fr);gap:22px;margin-bottom:34px}
.st{padding:18px 12px;text-align:center;position:relative}
.st:not(:last-child)::after{content:"";position:absolute;right:-17px;top:50%;border:7px solid transparent;border-left:10px solid var(--or);transform:translateY(-50%)}
.st svg{width:46px;height:46px;stroke:#fff;fill:none;stroke-width:1.6;stroke-linecap:round;stroke-linejoin:round;margin-bottom:10px}
.st b{display:block;font-size:14px;color:#fff}.st span{font-size:11.5px;color:var(--mu);display:block;margin-top:6px;line-height:1.4}
.ck{display:flex;flex-direction:column;gap:11px;flex:1}
.ck>div{display:flex;align-items:center;gap:12px;font-size:14.5px;color:#fff}
.ck svg{width:24px;height:24px;flex:none}
.bb{display:flex;gap:18px;align-items:center;margin-bottom:22px}
.bb>svg{width:132px;flex:none;stroke:#fff;fill:none;stroke-width:3.5;stroke-linecap:round;stroke-linejoin:round}
.priv{display:flex;gap:16px;align-items:center;padding:18px 22px}
.priv svg{width:40px;flex:none}.priv b{display:block;color:var(--or);font-size:15px;margin-bottom:3px}.priv span{font-size:13px;color:var(--mu)}
.st-key-cta{padding:8px 5vw 60px;align-items:center}
/* demo */
.st-key-dwrap{padding:34px 5vw 70px;max-width:1500px;margin:0 auto;width:100%;gap:22px !important}
.st-key-dwrap [data-testid="stColumn"]{display:flex;flex-direction:column}
.st-key-dwrap [data-testid="stColumn"]>[data-testid="stVerticalBlock"]{flex:1}
.st-key-dwrap [data-testid="stLayoutWrapper"]:has(>[class*="st-key-card_"]){flex:1;display:flex;flex-direction:column}
.st-key-dwrap [class*="st-key-card_"]{flex:1}
[class*="st-key-gh_sm_ex_"] button{padding:8px 6px !important;font-size:12px !important;letter-spacing:0}
[class*="st-key-gh_sm_ex_"] button p{font-size:12px !important}
.dh{display:flex;flex-wrap:wrap;gap:20px;align-items:flex-end;justify-content:space-between;margin-bottom:4px}
.dh .dt{margin:0;font-size:clamp(30px,3.6vw,46px);letter-spacing:-.02em;font-weight:800;color:#fff}.dh .dp{margin:8px 0 0;color:var(--mu);max-width:520px;line-height:1.55}
.stp{display:flex;gap:8px;margin:0;padding:0;flex-wrap:wrap}
.stp>div{display:flex;align-items:center;gap:10px;padding:7px 16px 7px 7px;border:1px solid var(--line);border-radius:99px;font-size:13px;color:var(--mu);background:#0f1d44}
.stp i{width:26px;height:26px;border-radius:50%;display:grid;place-items:center;background:#1b2c5e;font:700 12px "Space Mono",monospace;font-style:normal}
.stp .on{color:#fff;border-color:var(--or)}.stp .on i{background:var(--or);color:#1a0c00}.stp .done{color:#fff}.stp .done i{background:var(--teal);color:#062a3a}
.st-key-dwrap [data-testid="stHorizontalBlock"]{gap:22px !important;align-items:stretch}
[class*="st-key-card_"]{padding:22px;gap:.8rem !important;margin-bottom:0}
.dg2row{margin-top:22px}
.h4{margin:0 0 4px;font-size:17px;color:var(--or);display:flex;gap:12px;align-items:baseline;font-weight:700}
.h4 em{font:700 26px "Space Mono",monospace;font-style:normal;color:#fff}
.dz{border:1.5px dashed #3E5AA0;border-radius:14px;display:grid;place-items:center;text-align:center;color:var(--mu);font-size:13px;padding:16px;background:#0b17382e;min-height:120px}
.st-key-view{border:1.5px dashed #3E5AA0;border-radius:14px;padding:12px;background:#0b17382e;min-height:400px;justify-content:center}
.st-key-view [data-testid="stImage"] img{border-radius:10px}
.tips{margin:6px 0 4px}.tips .th{margin:0 0 10px;color:var(--or);font-size:15px;font-weight:700;display:flex;gap:8px;align-items:center}
.tips .ck>div{font-size:12.5px;color:#d6e0fa}.tips .ck svg{width:18px;height:18px}
.bars{display:flex;flex-direction:column;gap:11px;justify-content:center;border:1.5px dashed #3E5AA0;border-radius:14px;padding:20px;background:#0b17382e}
.bar{display:grid;grid-template-columns:112px 1fr 44px;gap:10px;align-items:center;font-size:12.5px;color:var(--mu)}
.bar.top{color:#fff;font-weight:700}
.bar s{display:block;height:8px;border-radius:99px;background:#1a2a5c;overflow:hidden;text-decoration:none}
.bar u{display:block;height:100%;border-radius:99px;background:linear-gradient(90deg,#FF8F3F,#FFB56E);animation:grow 1s cubic-bezier(.2,.8,.2,1) both;text-decoration:none}
@keyframes grow{from{width:0 !important}}
.bar em{font:12px "Space Mono",monospace;font-style:normal;text-align:right}
.hint{font-size:12px;color:var(--mu);margin:2px 0 0;line-height:1.5}
.note{border-left:2px solid var(--or);padding:2px 0 2px 14px;font-size:13.5px;color:#d6e0fa;line-height:1.55;margin-top:6px}
.note b{color:#fff}
.legend{display:flex;gap:14px;flex-wrap:wrap;color:var(--mu);font-size:12px}
.legend i{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:6px;vertical-align:-1px}
/* uploader / camera */
[data-testid="stFileUploaderDropzone"]{border:1.5px dashed #3E5AA0 !important;border-radius:14px !important;background:#0b17382e !important;min-height:150px;justify-content:center}
[data-testid="stFileUploaderDropzone"] *{color:var(--mu) !important}
[data-testid="stFileUploaderDropzone"] button{background:linear-gradient(135deg,#FF9A4D,#F27A1F) !important;color:#1a0c00 !important;border:0 !important;border-radius:99px !important;font-weight:800 !important}
[data-testid="stFileUploaderDropzone"] button *{color:#1a0c00 !important}
[data-testid="stFileUploader"] label{display:none}
[data-testid="stFileUploaderDropzone"]::before{content:"Drag and drop a photo here";color:#fff;font-weight:700;font-size:14px;margin-bottom:8px}
[data-testid="stFileUploaderDropzone"]{flex-direction:column;text-align:center}
[data-testid="stImage"] img{border-radius:12px}
[data-testid="stCaptionContainer"]{color:var(--mu)}
/* tabs */
.st-key-tabs{flex-direction:row !important;gap:6px !important;margin:-2px 0 6px;background:#0a1534;border-radius:12px;padding:4px;width:max-content}
.st-key-tabs button{border:0 !important;background:none !important;color:var(--mu) !important;font:600 13px "Plus Jakarta Sans",sans-serif !important;padding:7px 14px !important;border-radius:9px !important;box-shadow:none !important;min-height:0 !important;height:auto !important}
.st-key-tabs button p{font:inherit !important;color:inherit !important;margin:0 !important}
.st-key-tabs button:hover{color:#fff !important}
.limb{margin-top:22px;display:grid;grid-template-columns:auto 1fr 1fr 1fr;gap:20px;align-items:center;padding:22px}
.limb .lt{display:flex;gap:12px;align-items:center;font-size:21px;font-weight:700;color:#fff}
.limb .lp{margin:0;font-size:13.5px;color:#d6e0fa;line-height:1.5;padding-left:16px;border-left:2px solid var(--or)}
.foot2{margin-top:26px;color:var(--mu);font-size:12px;line-height:1.6;text-align:center}
.more{text-align:center;margin-top:30px;color:var(--or);font-weight:700}
.st-key-more{align-items:center;margin-top:18px}
/* take action */
.camp{display:grid;grid-template-columns:1.1fr .9fr;gap:50px;align-items:center;padding:7vh 5vw 30px}
.tag{font:700 12px "Space Mono",monospace;letter-spacing:.3em;color:var(--or);display:inline-block;margin-bottom:20px;padding:7px 14px;border:1px solid #FF8F3F66;border-radius:99px}
.big{font-size:clamp(44px,6.6vw,92px);line-height:1;margin:0 0 24px;letter-spacing:-.03em;font-weight:800;color:#fff}
.cp{font-size:clamp(16px,1.4vw,19px);line-height:1.7;color:#c5d2f0;max-width:560px;margin:0 0 30px}
.cf{position:relative;aspect-ratio:3/4;max-height:560px;border-radius:30px;background:var(--f1) center/cover;border:1px solid var(--line);box-shadow:0 40px 80px -30px #000;justify-self:center;width:100%}
.cf::before,.cf::after{content:"";position:absolute;width:64px;height:64px;border:5px solid var(--or)}
.cf::before{top:-14px;left:-14px;border-right:0;border-bottom:0;border-radius:20px 0 0 0}
.cf::after{bottom:-14px;right:-14px;border-left:0;border-top:0;border-radius:0 0 20px 0}
.st-key-camp{padding:7vh 5vw 30px}
.st-key-camp .camp{padding:0}
.facts{display:grid;grid-template-columns:repeat(3,1fr);gap:20px;padding:30px 5vw 0}
.fc{padding:28px}.fc b{font-size:clamp(46px,5vw,70px);color:var(--or);line-height:1;display:block;letter-spacing:-.03em}.fc span{color:var(--mu);font-size:14px;line-height:1.55;display:block;margin-top:10px}
.sec{padding:64px 5vw 0}
.acts{display:grid;grid-template-columns:repeat(4,1fr);gap:18px}
.ac{padding:26px;display:flex;flex-direction:column;gap:12px}.ac i{font:700 14px "Space Mono",monospace;font-style:normal;color:var(--or)}.ac b{font-size:20px;color:#fff}.ac .ap{margin:0;color:var(--mu);font-size:14px;line-height:1.6}
.st-key-close{margin:64px 5vw 0;width:auto !important;border-radius:32px;padding:60px 5vw;text-align:center;background:linear-gradient(135deg,#FF9A4D,#E8661A);color:#1a0c00;align-items:center}
.close-h{font-size:clamp(30px,4.6vw,60px);line-height:1.05;margin:0 0 14px;letter-spacing:-.03em;font-weight:800;color:#1a0c00}
.close-p{margin:0 auto 28px;max-width:560px;font-weight:500;line-height:1.6;color:#1a0c00}
.st-key-prow,.st-key-crow{gap:14px !important}
.thx{margin:16px 0 0;color:var(--teal);font-weight:600;font-size:14px}
.foot{padding:34px 5vw 40px;text-align:center;color:var(--mu);font-size:13px;line-height:1.7}
@media(max-width:1100px){.study{grid-template-columns:1fr;padding:30px 5vw}.sface{display:none}.hero-tx{max-width:100%}.st-key-hero [data-testid="stElementContainer"]{position:static}
.hero-face{opacity:.35;width:100%}.camp{grid-template-columns:1fr}.cf{max-height:420px;aspect-ratio:4/3}.acts{grid-template-columns:1fr 1fr}.limb{grid-template-columns:1fr}.limb .lp{border-left:0;padding-left:0}}
@media(max-width:700px){.gfe,.steps,.acts,.facts{grid-template-columns:1fr}.g.w{width:100%}.st:not(:last-child)::after{display:none}.bb{flex-direction:column;align-items:flex-start}.wm b{font-size:21px}.pill{display:none}.st-key-hdr [data-testid="stHorizontalBlock"]{flex-wrap:wrap !important}.st-key-nav_home{margin-left:12px;padding-left:12px;margin-right:12px}.st-key-nav_study,.st-key-nav_demo{margin-right:12px}[class*="st-key-nav_"] button,[class*="st-key-nav_"] button p{font-size:13px !important}.st-key-dwrap [data-testid="stHorizontalBlock"]{flex-wrap:wrap}.st-key-dwrap>[data-testid="stLayoutWrapper"]>[data-testid="stHorizontalBlock"]>[data-testid="stColumn"]{min-width:100% !important}.wm{display:none}.st-key-nav_home{margin-left:6px;padding-left:10px;margin-right:10px}.st-key-nav_study,.st-key-nav_demo{margin-right:10px}[class*="st-key-nav_"] button,[class*="st-key-nav_"] button p{font-size:11.5px !important}.st-key-hdr{padding:10px 4vw}.mk{width:38px;height:38px}}
.st-key-card_up [data-testid="stHorizontalBlock"]{flex-wrap:nowrap !important;gap:10px !important}.st-key-card_up [data-testid="stColumn"]{min-width:0 !important;flex:1 1 0 !important}
'''
st.markdown("<style>" + CSS + "</style>", unsafe_allow_html=True)


def md(s):
    st.markdown(h(s), unsafe_allow_html=True)


def face_vars(f1=False, f2=False):
    v = ""
    if f1:
        v += f"--f1:url({data_uri('assets/design_f1.jpg')});"
    if f2:
        v += f"--f2:url({data_uri('assets/design_f2.jpg')});"
    if v:
        st.markdown("<style>:root{" + v + "}</style>", unsafe_allow_html=True)


# ================================================================ small pieces
CK = '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="12" fill="#2DD4BF"/><path d="M6.5 12.5l3.8 3.8 7.2-8" fill="none" stroke="#062a3a" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/></svg>'
LOGO = ('<div class="lx logo"><svg class="mk" viewBox="0 0 48 48"><defs><linearGradient id="lg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#5B87F7"/><stop offset="1" stop-color="#2447A8"/></linearGradient></defs>'
        '<rect width="48" height="48" rx="14" fill="url(#lg)"/><g fill="none" stroke-linecap="round" stroke-width="3.4"><path d="M10 17Q15 10 20 16" stroke="#FF8F3F"/><path d="M28 16Q33 10 38 17" stroke="#FF8F3F"/><path d="M16 32Q24 38 32 32" stroke="#fff"/></g>'
        '<circle cx="17" cy="25" r="2.8" fill="#fff"/><circle cx="31" cy="25" r="2.8" fill="#fff"/></svg><span class="wm"><b>Libras</b><small>GFE LAB</small></span></div>')
PILL = ('<div class="lx pill"><svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M9 3h6M10 3v6l-5 9a2 2 0 0 0 2 3h10a2 2 0 0 0 2-3l-5-9V3"/><path d="M7.5 15h9"/></svg>Research Demo</div>')
UP_ICON = '<svg viewBox="0 0 48 48"><path d="M14 34a8 8 0 0 1-1-16 11 11 0 0 1 21-2 9 9 0 0 1 1 18"/><path d="M24 40V24M18 29l6-6 6 6"/></svg>'
FOOT = ('<div class="lx foot">Libras GFE Lab &middot; Research demo &middot; Photos are not saved<br>'
        'Demo only, 2D only. Results may be inaccurate. The system analyzes grammatical facial expressions, not emotions, and it is not a Libras interpreter. '
        'Built on landmark data from two Libras signers (Freitas et al., 2014).</div>')

PAGES = ("home", "study", "demo", "act")


def go_page(p):
    st.session_state.page = p
    st.rerun()


def header():
    with st.container(key="hdr"):
        c = st.columns([1, 1, 1, 1, 1, 1, 1])
        with c[0]:
            md(LOGO)
        for col, key, label in ((c[1], "home", "Home"), (c[2], "study", "The Study"), (c[3], "demo", "Demo"), (c[4], "act", "Take Action")):
            with col:
                if st.button(label, key="nav_" + key):
                    go_page(key)
        with c[6]:
            md(PILL)
    st.markdown("<style>.st-key-nav_" + st.session_state.page + " button{color:#fff !important;border-bottom-color:#FF8F3F !important;}</style>",
                unsafe_allow_html=True)


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


def bars_html(rows, empty_names=None):
    """rows: list of (label, width 0-100, value text, colour or None, is_top)."""
    if rows is None:
        rows_h = "".join(f'<div class="bar"><span>{esc(n)}</span><s></s><em>–</em></div>' for n in empty_names)
    else:
        rows_h = "".join(
            f'<div class="bar{" top" if top else ""}"><span>{esc(n)}</span><s><u style="width:{w:.0f}%' + (f';background:{c}' if c else '') + f'"></u></s><em>{esc(v)}</em></div>'
            for n, w, v, c, top in rows)
    return f'<div class="lx bars">{rows_h}</div>'


def zone_rows(zw):
    col = overlay_view._rank_colours(zw)
    top = max(zw.values())
    items = sorted(zw.items(), key=lambda kv: -kv[1])
    return [(z.replace("_", " ").capitalize(), 100 * v / top, f"{100*v:.0f}%", rgb_css(col[z]), i == 0) for i, (z, v) in enumerate(items)]


def analyze(img, use_clahe):
    hh, ww = img.shape[:2]
    k = MAX_SIDE / max(hh, ww)
    if k < 1:
        img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    seen = cv_pipeline.enhance(img) if use_clahe else None
    ann, msg, scores, zw = inference.predict_gfe(img, use_clahe=bool(use_clahe))
    if not scores:
        return {"ok": False, "ann": ann, "seen": seen, "orig": img, "msg": msg}
    ranked, margin, raw = show_scores(img, use_clahe=bool(use_clahe))
    rawd = {n: float(r) for n, r in zip(inference._names, raw)}
    top_name = ranked[0][0]
    notes = []
    if use_clahe:
        p0 = cv_pipeline.get_points(img)
        p1 = cv_pipeline.get_points(img, use_clahe=True)
        if p0 is None and p1 is not None:
            notes.append("The contrast fix helped: a face was found only with it.")
        elif p0 is not None and p1 is not None:
            shift = 100 * float(np.linalg.norm(p0 - p1, axis=1).mean()) / float(np.ptp(p0[:, 0]))
            notes.append(f"Contrast fix applied: landmarks moved {shift:.1f}% of the face width." + (" Small change." if shift < 1 else ""))
    top = max(b for _, b in ranked)
    rank_rows = [(nice(n), 100 * b / top, f"{rawd[n]:.1f}", None, i == 0) for i, (n, b) in enumerate(ranked)]
    ez = expr_zones(top_name)
    return {"ok": True, "ann": ann, "seen": seen, "orig": img, "top_name": top_name, "top": nice(top_name),
            "notes": notes, "rank": rank_rows,
            "zexpr": zone_rows(ez) if ez else None, "zall": zone_rows({k_: float(v_) for k_, v_ in zw.items()})}


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
st.session_state.setdefault("view", "o")
st.session_state.setdefault("pledged", False)
st.session_state.setdefault("share", False)

EX_DIR = BASE + "/examples"
EXAMPLES = [os.path.join(EX_DIR, f) for f in ("01_whq.jpg", "04_cond.jpg", "07_ynq.jpg")
            if os.path.exists(os.path.join(EX_DIR, f))]

header()
page = st.session_state.page

# ================================================================ HOME
if page == "home":
    face_vars(f1=True)
    with st.container(key="hero"):
        md('<div class="lx hero-face"></div><div class="lx hero-tx"><div class="h1">The <span class="a">Language</span><br>Beyond <span class="b">Words</span></div>'
           '<div class="lead">Your face carries grammar: in the Libras dataset, these are classified as Grammatical Facial Expressions (GFE). '
           'This demo reads a face photo and shows which regions of the face a model relies on.</div></div>')
        if st.button("EXPLORE  →", key="bl_explore"):
            go_page("study")
    md('<div class="lx panel"><div class="ph">Discover how facial expressions shape meaning in Libras</div>'
       '<div class="stats"><div><b>9</b><span>expressions</span></div><div><b>100</b><span>facial landmarks</span></div><div><b>0</b><span>photos saved</span></div></div></div>')
    md(FOOT)

# ================================================================ THE STUDY
elif page == "study":
    face_vars(f2=True)
    cards = ""
    for i, (key, (name, text, short)) in enumerate(EXPRESSIONS.items(), 1):
        cards += f'<div class="card g{" w" if i == 9 else ""}"><i>{i:02d}</i><div><b>{esc(name)}</b><span>{esc(short)}</span></div></div>'
    steps = (f'<div class="card st">{UP_ICON}<b>Upload a Photo</b><span>Provide a clear, front-facing photo</span></div>'
             '<div class="card st"><svg viewBox="0 0 48 48"><path d="M5 15V8h7M36 8h7v7M43 33v7h-7M12 40H5v-7"/><circle cx="18" cy="21" r="1.6" fill="#fff"/><circle cx="30" cy="21" r="1.6" fill="#fff"/><path d="M17 29q7 6 14 0"/></svg><b>Detect Landmarks</b><span>Mapping 100 facial landmarks</span></div>'
             '<div class="card st"><svg viewBox="0 0 48 48"><path d="M7 41h36M12 41V28h6v13M21 41V18h6v23M30 41V7h6v34"/></svg><b>Classify &amp; Analyze</b><span>See the results and which regions matter</span></div>')
    items = ["Face the camera", "Use even lighting", "Keep the entire head visible", "Avoid heavy shadows or masks",
             "Upload a JPG or PNG image", "Only upload photos with consent"]
    ck = "".join(f"<div>{CK}{t}</div>" for t in items)
    md(f'''<div class="lx study"><div class="sface"></div>
      <div><div class="t">The Expressions</div><div class="gfe">{cards}</div></div>
      <div><div class="t">How It Works</div><div class="steps">{steps}</div>
      <div class="t">Before You Begin</div>
      <div class="bb"><div class="ck">{ck}</div>
      <svg viewBox="0 0 140 140"><path d="M10 38V10h28M102 10h28v28M130 102v28h-28M38 130H10v-28"/><path d="M45 62c0-22 12-34 25-34s25 12 25 34c0 14-6 26-25 34-19-8-25-20-25-34z" stroke-width="3"/><path d="M52 66l14 14 26-30" stroke="#2DD4BF" stroke-width="5"/></svg></div>
      <div class="card priv"><svg viewBox="0 0 40 44" fill="none" stroke="#FF8F3F" stroke-width="3.5" stroke-linecap="round"><rect x="6" y="19" width="28" height="22" rx="5" fill="#FF8F3F"/><path d="M12 19v-6a8 8 0 0 1 16 0v6"/><circle cx="20" cy="30" r="2.6" fill="#1a0c00" stroke="none"/></svg>
      <div><b>Your Privacy Matters</b><span>Photos are not saved. This demo is for research purposes only.</span></div></div></div></div>''')
    with st.container(key="cta"):
        if st.button("I’m ready — Try the Demo →", key="or_ready"):
            go_page("demo")
    md(FOOT)

# ================================================================ DEMO
elif page == "demo":
    r = st.session_state.result
    with st.container(key="dwrap"):
        # --- source (needed for the step pills)
        up = cam = None
        source = None
        # header (pills filled after we know the state; computed below via placeholder)
        head = st.empty()
        c1, c2, c3 = st.columns([340, 700, 380])

        with c1:
            with st.container(key="card_up"):
                md('<div class="lx h4"><em>01</em>Upload</div>')
                up = st.file_uploader("Drag and drop a photo here", type=["jpg", "jpeg", "png"])
                if st.button("Use Camera" if not st.session_state.cam_on else "Close Camera", key="or_sm_camera"):
                    st.session_state.cam_on = not st.session_state.cam_on
                    st.rerun()
                cam = st.camera_input("Camera", label_visibility="collapsed") if st.session_state.cam_on else None
                if EXAMPLES:
                    st.caption("Or try an example (frames from the public dataset)")
                    cols = st.columns(len(EXAMPLES))
                    for cc, p in zip(cols, EXAMPLES):
                        with cc:
                            st.image(p, width="stretch")
                            if st.button("Use", key="gh_sm_ex_" + os.path.basename(p)):
                                st.session_state.example = p
                                st.session_state.result = None
                                st.rerun()
                if cam is not None:
                    source, label = ("upload", cam), "your camera photo"
                elif up is not None:
                    source, label = ("upload", up), "your uploaded photo"
                elif st.session_state.example:
                    source, label = ("example", st.session_state.example), "an example photo"
                tips = ["Face the camera, even lighting", "Keep the entire head visible", "Avoid heavy shadows or masks", "Improve contrast (CLAHE), experimental"]
                md('<div class="lx tips"><div class="th"><svg width="18" height="18" viewBox="0 0 24 24" fill="#FF8F3F"><path d="M12 2a7 7 0 0 0-4 12.7V18h8v-3.3A7 7 0 0 0 12 2zM9 20h6v2H9z"/></svg>Photo Tips</div><div class="ck">'
                   + "".join(f"<div>{CK}{t}</div>" for t in tips) + "</div></div>")
                b1, b2 = st.columns(2)
                with b1:
                    ckey = ("on_" if st.session_state.clahe else "") + "gh_sm_clahe"
                    if st.button("CLAHE ✓" if st.session_state.clahe else "CLAHE", key=ckey):
                        st.session_state.clahe = not st.session_state.clahe
                        st.rerun()
                with b2:
                    go = st.button("Classify", key="or_sm_classify", disabled=source is None)
                if source is None:
                    st.caption("Choose a photo first.")
                if go and source:
                    img = load_uploaded(source[1]) if source[0] == "upload" else load_path(source[1])
                    with st.spinner("Analyzing..."):
                        st.session_state.result = analyze(img, st.session_state.clahe)
                    st.session_state.view = "o"
                    st.rerun()

        r = st.session_state.result
        n = 4 if (r and r["ok"]) else (1 if source else 0)
        names = ["Upload", "Detect", "Classify", "Analyze"]
        pills = "".join(f'<div class="{"done" if i < n else "on" if i == n else ""}"><i>{i+1}</i>{t}</div>' for i, t in enumerate(names))
        head.markdown(h(f'<div class="lx dh"><div><div class="dt">Try the Demo</div><div class="dp">Upload a front-facing photo to see how the model reads the face. Photos are never saved.</div></div>'
                        f'<div class="stp">{pills}</div></div>'), unsafe_allow_html=True)

        with c2:
            with st.container(key="card_lm"):
                md('<div class="lx h4"><em>02</em>Facial Landmark Detection</div>')
                if st.session_state.clahe:
                    with st.container(key="tabs", horizontal=True):
                        if st.button("Original", key="tab_o"):
                            st.session_state.view = "o"
                            st.rerun()
                        if st.button("CLAHE", key="tab_c"):
                            st.session_state.view = "c"
                            st.rerun()
                    st.markdown("<style>.st-key-tab_" + st.session_state.view + " button{background:#1f3472 !important;color:#fff !important;}</style>", unsafe_allow_html=True)
                view = st.session_state.view if st.session_state.clahe else "o"
                with st.container(key="view"):
                    if view == "c":
                        if r and r.get("seen") is not None:
                            a, b = st.columns(2)
                            a.image(r["orig"], caption="Original", width="stretch")
                            b.image(r["seen"], caption="After CLAHE (what the detector saw)", width="stretch")
                        else:
                            md('<div class="lx" style="text-align:center;color:#9FB0D3;font-size:13px">Enhanced version appears here after Classify</div>')
                    elif r:
                        st.image(r["ann"], width="stretch")
                    elif source:
                        st.image(load_uploaded(source[1]) if source[0] == "upload" else load_path(source[1]), width="stretch")
                    else:
                        md('<div class="lx" style="text-align:center;color:#9FB0D3;font-size:13px">Your photo and detected landmarks appear here</div>')
                if r and r["ok"] and view == "o":
                    md('<div class="lx legend"><span><i style="background:rgb(230,50,50)"></i>top third of zones</span><span><i style="background:rgb(245,170,30)"></i>middle third</span>'
                       '<span><i style="background:rgb(60,180,90)"></i>bottom third</span></div>'
                       '<div class="lx hint">Colour = zone importance (RF, rank-based). The same for every photo.</div>')

        with c3:
            with st.container(key="card_res"):
                md('<div class="lx h4"><em>03</em>Classification Results</div>')
                if r and r["ok"]:
                    md(bars_html(r["rank"]))
                    md(f'<div class="lx note">Closest pattern in this model: <b>{esc(r["top"])}</b>. This model is weak: it was tested on signers it had never seen and often misses. Demo only.'
                       + "".join(f" {esc(t)}" for t in r["notes"]) + '</div>')
                    d = EXPRESSIONS.get(r["top_name"])
                    if d:
                        md(f'<div class="lx note"><b>What is “{esc(d[0])}”?</b> {esc(d[1])}</div>')
                    md('<div class="lx hint">Numbers are the model’s raw scores, not probabilities. Compare the order, not the size.</div>')
                elif r:
                    md(bars_html(None, [nice(x) for x in EXPRESSIONS]))
                    md(f'<div class="lx note">{esc(r["msg"])}</div>')
                else:
                    md(bars_html(None, [v[0] for v in EXPRESSIONS.values()]))
                    md('<div class="lx hint">Scores for each expression appear here after Classify.</div>')

        z1, z2 = st.columns(2)
        ok = bool(r and r["ok"])
        with z1:
            with st.container(key="card_z1"):
                ttl = "Zone Importance: Predicted Output" + (f' (“{esc(r["top"])}”)' if ok else "")
                md(f'<div class="lx h4"><em>04</em>{ttl}</div>')
                if ok and r.get("zexpr"):
                    md(bars_html(r["zexpr"]))
                    md('<div class="lx hint">Changes with the expression the model picks. A model-level number, not a measurement of your face.</div>')
                else:
                    md(bars_html(None, ["Eyebrows", "Eyes", "Nose", "Mouth", "Face contour"]))
        with z2:
            with st.container(key="card_z2"):
                md('<div class="lx h4"><em>05</em>Zone Importance: Overall</div>')
                if ok:
                    md(bars_html(r["zall"]))
                    md('<div class="lx hint">The same for every photo. These are the colours on the face overlay.</div>')
                else:
                    md(bars_html(None, ["Eyebrows", "Eyes", "Nose", "Mouth", "Face contour"]))

        md('''<div class="lx card limb"><div class="lt"><svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="1.8" stroke-linecap="round"><circle cx="12" cy="12" r="10"/><path d="M12 11v6M12 7.5v.5"/></svg>Limitations</div>
           <div class="lp">Trained using landmark data from two Libras signers.</div>
           <div class="lp">2D analysis only: depth information is not considered.</div>
           <div class="lp">Photo landmarks are approximately matched to Kinect landmarks, and the demo has not been evaluated on photos. Results may be inaccurate.</div></div>''')
        with st.container(key="more"):
            if st.button("Done exploring? See how you can help →", key="gh_sm_more"):
                go_page("act")
    md(FOOT)

# ================================================================ TAKE ACTION
else:
    face_vars(f1=True)
    with st.container(key="camp"):
        a, b = st.columns([1.1, 0.9], gap="large", vertical_alignment="center")
        with a:
            md('<div class="lx"><span class="tag">OUR CAUSE</span><div class="big">Every <span class="a">Voice</span><br>Belongs Here</div>'
               '<div class="cp">Many Deaf people share ideas, jokes and everyday life through sign language, yet much of the world is still built without them in mind. '
               'This study is one small step toward technology that listens with its eyes, face included. You can be part of the story too.</div></div>')
            with st.container(horizontal=True, key="prow"):
                if st.button("Thank you ✓" if st.session_state.pledged else "Add my voice", key="or_pledge"):
                    st.session_state.pledged = True
                    st.rerun()
                if st.button("Share with a friend", key="gh_share"):
                    st.session_state.share = True
                    st.rerun()
            if st.session_state.pledged:
                md('<div class="lx thx">Thanks for listening. Small gestures add up.</div>')
            if st.session_state.share:
                st.caption("Copy this link and send it to a friend:")
                st.code(LIVE_URL, language=None)
        with b:
            md('<div class="lx cf" role="img" aria-label="Face"></div>')
    md('''<div class="lx facts"><div class="card fc"><b>9</b><span>grammatical facial expressions the study classifies</span></div>
      <div class="card fc"><b>100</b><span>landmarks mapped on every face to see what the model relies on</span></div>
      <div class="card fc"><b>2</b><span>signers in the training data. Many more voices deserve a place.</span></div></div>
      <div class="lx sec"><div class="t">Small Ways to Help</div><div class="acts">
      <div class="card ac"><i>01</i><b>Say Hello</b><div class="ap">Learn a few signs, like “hello” and “thank you”. Even a little effort helps people feel welcome.</div></div>
      <div class="card ac"><i>02</i><b>Listen</b><div class="ap">Follow Deaf creators and educators, and let their voices lead the conversation.</div></div>
      <div class="card ac"><i>03</i><b>Make Space</b><div class="ap">When you host a call or event, keep the interpreter’s face clear and well lit so everyone can follow along.</div></div>
      <div class="card ac"><i>04</i><b>Share the Story</b><div class="ap">Pass this page to a friend, classmate or teacher. Awareness starts with one conversation.</div></div></div></div>''')
    with st.container(key="close"):
        md('<div class="lx close-h">Everyone belongs<br>in the conversation.</div><div class="lx close-p">Thanks for stopping by. Explore the demo, share what you learned, and help us make room for every voice.</div>')
        with st.container(horizontal=True, horizontal_alignment="center", key="crow"):
            if st.button("Try the Demo →", key="dk_try"):
                go_page("demo")
            if st.button("Back to The Study", key="ol_back"):
                go_page("study")
    md(FOOT)
