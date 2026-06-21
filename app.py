import streamlit as st
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from model.gan               import Generator, Discriminator
from model.train             import train_gan, NOISE_DIM, DEVICE
from model.classifier        import PneumoniaClassifier
from model.train_classifier  import train_classifier, predict_single_image
from attacks.fgsm            import fgsm_attack
from attacks.pgd             import pgd_attack
from defense.defend          import gaussian_denoise, median_filter_defense, detect_adversarial

import os

# ── Cloud detection ───────────────────────────────────────────────────
IS_CLOUD = os.path.exists("/mount/src")

def setup_dataset():
    """Check dataset exists. On cloud, weights are pre-loaded so skip."""
    if IS_CLOUD:
        # On cloud we use pre-trained weights — no dataset needed for inference
        return
    if not os.path.exists("data/chest_xray/train"):
        st.warning(
            "Dataset not found. Download from Kaggle: "
            "https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia "
            "and place in data/chest_xray/"
        )
        st.stop()

# ─────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedShield AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

setup_dataset()

# ─────────────────────────────────────────────────────────────────────
# Session state bootstrap
# ─────────────────────────────────────────────────────────────────────
for k, v in [("dark_mode", True), ("active_tab", 0)]:
    if k not in st.session_state:
        st.session_state[k] = v

dark = st.session_state["dark_mode"]

# ─────────────────────────────────────────────────────────────────────
# Design tokens — two complete palettes
# ─────────────────────────────────────────────────────────────────────
if dark:
    BG        = "#070b14"
    BG2       = "#0d1424"
    BG3       = "#111827"
    BORDER    = "#1a2640"
    BORDER2   = "#243351"
    TEXT1     = "#f0f4ff"
    TEXT2     = "#8b9dc3"
    TEXT3     = "#3d5175"
    ACCENT    = "#0ea5e9"
    ACCENT_DK = "#0284c7"
    ACCENT_GL = "rgba(14,165,233,0.12)"
    GLOW      = "rgba(14,165,233,0.25)"
    CHART_BG  = "#070b14"
    CHART_PL  = "#0d1424"
    XRAY_CM   = "bone"
    TAG_BG    = "rgba(14,165,233,0.10)"
    TAG_CLR   = "#38bdf8"
else:
    BG        = "#f7f9fc"
    BG2       = "#ffffff"
    BG3       = "#eef2f8"
    BORDER    = "#dde3ee"
    BORDER2   = "#c8d2e6"
    TEXT1     = "#0d1424"
    TEXT2     = "#3d5175"
    TEXT3     = "#8b9dc3"
    ACCENT    = "#0284c7"
    ACCENT_DK = "#0369a1"
    ACCENT_GL = "rgba(2,132,199,0.08)"
    GLOW      = "rgba(2,132,199,0.18)"
    CHART_BG  = "#f7f9fc"
    CHART_PL  = "#ffffff"
    XRAY_CM   = "gray"
    TAG_BG    = "rgba(2,132,199,0.08)"
    TAG_CLR   = "#0284c7"

# ─────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

*,*::before,*::after{{box-sizing:border-box;}}
html,body,[class*="css"]{{font-family:'Inter',sans-serif;}}
::-webkit-scrollbar{{width:5px;height:5px;}}
::-webkit-scrollbar-track{{background:{BG};}}
::-webkit-scrollbar-thumb{{background:{BORDER2};border-radius:10px;}}

/* ── Streamlit chrome ── */
.stApp{{background:{BG};color:{TEXT1};}}
.block-container{{
    padding:0 2.5rem 3rem 2.5rem !important;
    max-width:1280px;
    /* push content down enough so header isn't clipped */
    padding-top:3.5rem !important;
}}
header[data-testid="stHeader"]{{background:{BG} !important;border-bottom:1px solid {BORDER};}}
[data-testid="stSidebar"]{{background:{BG2} !important;border-right:1px solid {BORDER};}}
[data-testid="stSidebar"]>div:first-child{{padding:1.5rem 1.2rem;}}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"]{{
    background:{BG2};border-radius:14px;padding:5px;gap:3px;
    border:1px solid {BORDER};box-shadow:0 2px 12px rgba(0,0,0,0.15);
}}
.stTabs [data-baseweb="tab"]{{
    background:transparent;color:{TEXT2};border-radius:10px;
    padding:9px 18px;font-size:13px;font-weight:500;border:none;
    transition:all 0.2s ease;letter-spacing:0.01em;
}}
.stTabs [aria-selected="true"]{{
    background:linear-gradient(135deg,{ACCENT_DK},{ACCENT}) !important;
    color:#fff !important;box-shadow:0 2px 10px {GLOW};
}}
.stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]){{
    color:{TEXT1} !important;background:{BORDER} !important;
}}
.stTabs [data-baseweb="tab-panel"]{{padding-top:28px;}}

/* ── Buttons ── */
.stButton>button{{
    background:linear-gradient(135deg,{ACCENT_DK},{ACCENT});
    color:#fff !important;border:none !important;border-radius:10px;
    padding:11px 28px;font-weight:600;font-size:14px;letter-spacing:0.02em;
    transition:all 0.22s ease;width:100%;box-shadow:0 3px 14px {GLOW};
    position:relative;overflow:hidden;
}}
.stButton>button:hover{{
    transform:translateY(-2px);box-shadow:0 8px 24px {GLOW};filter:brightness(1.08);
}}
.stButton>button:active{{transform:translateY(0);}}

/* theme pill */
.theme-btn .stButton>button{{
    background:{BG3} !important;color:{TEXT2} !important;
    border:1px solid {BORDER2} !important;border-radius:20px !important;
    padding:5px 14px !important;font-size:12px !important;font-weight:500 !important;
    width:auto !important;box-shadow:none !important;letter-spacing:0;
}}
.theme-btn .stButton>button:hover{{
    background:{BORDER} !important;transform:none !important;
    box-shadow:none !important;filter:none !important;
}}
/* next-step button — outlined style */
.next-btn .stButton>button{{
    background:transparent !important;
    color:{ACCENT} !important;
    border:1.5px solid {ACCENT} !important;
    border-radius:10px !important;
    padding:9px 20px !important;
    font-size:13px !important;
    font-weight:600 !important;
    width:auto !important;
    box-shadow:none !important;
    letter-spacing:0.01em;
}}
.next-btn .stButton>button:hover{{
    background:{ACCENT_GL} !important;
    transform:none !important;
    box-shadow:0 0 0 3px {GLOW} !important;
    filter:none !important;
}}
/* download button */
.dl-btn .stDownloadButton>button{{
    background:linear-gradient(135deg,#065f46,#059669) !important;
    color:#fff !important;border:none !important;border-radius:10px;
    padding:11px 28px;font-weight:600;font-size:14px;letter-spacing:0.02em;
    transition:all 0.22s ease;width:100%;box-shadow:0 3px 14px rgba(5,150,105,0.35);
}}
.dl-btn .stDownloadButton>button:hover{{
    transform:translateY(-2px);box-shadow:0 8px 24px rgba(5,150,105,0.45);
    filter:brightness(1.08);
}}

/* ── Metrics ── */
[data-testid="stMetric"]{{
    background:{BG2};border:1px solid {BORDER};border-radius:12px;
    padding:16px 20px;transition:border-color 0.2s;
}}
[data-testid="stMetric"]:hover{{border-color:{BORDER2};}}
[data-testid="stMetricLabel"]{{color:{TEXT3};font-size:11px;letter-spacing:0.07em;text-transform:uppercase;}}
[data-testid="stMetricValue"]{{color:{TEXT1};font-size:22px;font-weight:700;font-family:'JetBrains Mono',monospace;}}
[data-testid="stMetricDelta"]{{font-size:12px;}}

/* ── Misc ── */
.stAlert{{border-radius:10px;border-left-width:3px;font-size:13px;background:{BG2} !important;}}
[data-testid="stSlider"] label,
[data-testid="stSelectbox"] label,
.stRadio>label{{color:{TEXT2};font-size:12px;font-weight:500;letter-spacing:0.04em;text-transform:uppercase;}}
div[data-baseweb="select"]>div{{
    background:{BG3} !important;border-color:{BORDER2} !important;
    color:{TEXT1} !important;border-radius:8px !important;
}}
.stRadio [data-testid="stMarkdownContainer"] p{{color:{TEXT2};font-size:13px;}}
[data-testid="stFileUploader"]{{
    background:{BG2};border:1.5px dashed {BORDER2};border-radius:12px;
    padding:20px;transition:border-color 0.2s;
}}
[data-testid="stFileUploader"]:hover{{border-color:{ACCENT};}}
.stProgress>div>div{{
    background:linear-gradient(90deg,{ACCENT_DK},{ACCENT},#38bdf8);
    border-radius:6px;transition:width 0.3s ease;
}}
.stProgress>div{{background:{BORDER};border-radius:6px;}}
.stSpinner>div{{border-top-color:{ACCENT} !important;}}
[data-testid="stImage"] img{{
    border-radius:10px;border:1px solid {BORDER};
    transition:border-color 0.2s,box-shadow 0.2s;
}}
[data-testid="stImage"] img:hover{{border-color:{ACCENT};box-shadow:0 0 0 3px {ACCENT_GL};}}
.stCaption{{color:{TEXT3} !important;font-size:11px !important;letter-spacing:0.02em;}}
[data-testid="stPyplotGlobalUse"]{{border-radius:12px;overflow:hidden;border:1px solid {BORDER};}}

/* ── Custom components ── */
.ms-card{{
    background:{BG2};border:1px solid {BORDER};border-radius:14px;
    padding:22px 26px;margin-bottom:18px;
    transition:border-color 0.2s,box-shadow 0.2s;
}}
.ms-card:hover{{border-color:{BORDER2};box-shadow:0 4px 20px rgba(0,0,0,0.12);}}
.ms-card-accent{{
    background:{BG2};border:1px solid {BORDER};border-top:3px solid {ACCENT};
    border-radius:14px;padding:22px 26px;margin-bottom:18px;
}}
.ms-eyebrow{{
    font-size:10px;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;
    color:{ACCENT};margin-bottom:6px;
}}
.ms-section-label{{
    font-size:10px;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;
    color:{TEXT3};margin-bottom:14px;padding-bottom:8px;border-bottom:1px solid {BORDER};
}}
.ms-rule{{border:none;border-top:1px solid {BORDER};margin:28px 0;}}
.ms-tag{{
    display:inline-block;background:{TAG_BG};color:{TAG_CLR};
    border:1px solid {TAG_CLR}33;border-radius:5px;padding:2px 8px;
    font-size:10px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;
    margin-bottom:10px;margin-right:4px;
}}
.ms-mono{{font-family:'JetBrains Mono',monospace;font-size:12px;color:{TEXT2};}}
.diag-normal{{
    display:inline-flex;align-items:center;gap:7px;
    background:{"#022c22" if dark else "#dcfce7"};
    color:{"#34d399" if dark else "#16a34a"};
    border:1px solid {"#065f46" if dark else "#86efac"};
    border-radius:25px;padding:8px 22px;font-size:14px;font-weight:700;letter-spacing:0.06em;
}}
.diag-pneumonia{{
    display:inline-flex;align-items:center;gap:7px;
    background:{"#2d0a0a" if dark else "#fee2e2"};
    color:{"#f87171" if dark else "#dc2626"};
    border:1px solid {"#7f1d1d" if dark else "#fca5a5"};
    border-radius:25px;padding:8px 22px;font-size:14px;font-weight:700;letter-spacing:0.06em;
}}
.score-bar-track{{height:6px;background:{BORDER};border-radius:6px;margin:8px 0 4px;}}
.hero-pulse{{
    display:inline-block;width:8px;height:8px;border-radius:50%;
    background:{ACCENT};box-shadow:0 0 0 0 {GLOW};
    animation:pulse 2s ease-in-out infinite;margin-right:6px;vertical-align:middle;
}}
@keyframes pulse{{
    0%,100%{{box-shadow:0 0 0 0 {GLOW};}}
    50%{{box-shadow:0 0 0 7px rgba(14,165,233,0);}}
}}
.footer-wrap{{
    margin-top:64px;border-top:1px solid {BORDER};
    padding:32px 0 20px;display:flex;align-items:center;
    justify-content:space-between;flex-wrap:wrap;gap:16px;
}}
.footer-li-btn{{
    display:inline-flex;align-items:center;gap:8px;
    background:{BG2};border:1px solid {BORDER2};color:{ACCENT};
    border-radius:10px;padding:10px 20px;font-size:13px;font-weight:600;
    text-decoration:none;transition:all 0.2s;letter-spacing:0.01em;
}}
.footer-li-btn:hover{{
    background:{ACCENT_DK};color:#fff;border-color:{ACCENT_DK};
    box-shadow:0 4px 14px {GLOW};
}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────
def style_ax(ax):
    ax.set_facecolor(CHART_BG)
    ax.tick_params(colors=TEXT3, labelsize=9)
    ax.xaxis.label.set_color(TEXT3)
    ax.yaxis.label.set_color(TEXT3)
    for s in ax.spines.values():
        s.set_edgecolor(BORDER)

def tensor_to_np(t):
    a = t.squeeze().detach().cpu().numpy()
    return (a - a.min()) / (a.max() - a.min() + 1e-8)

def show_xray(col, tensor, caption):
    col.image(tensor_to_np(tensor), caption=caption, clamp=True, use_container_width=True)

def score_bar(val, label=""):
    pct = int(min(max(val, 0), 1) * 100)
    clr = "#34d399" if val > 0.75 else "#f59e0b" if val > 0.4 else ACCENT
    return (f'<div class="ms-mono" style="margin-bottom:3px;">'
            f'{label}: <b style="color:{TEXT1};">{val:.3f}</b></div>'
            f'<div class="score-bar-track"><div style="width:{pct}%;height:100%;'
            f'background:{clr};border-radius:6px;"></div></div>')

def diag_card(col, title, pred, conf):
    conf_d = conf if pred == "PNEUMONIA" else 1 - conf
    pill   = "diag-pneumonia" if pred == "PNEUMONIA" else "diag-normal"
    dot    = "🔴" if pred == "PNEUMONIA" else "🟢"
    col.markdown(
        f'<div class="ms-card" style="text-align:center;padding:20px 14px;">'
        f'<div class="ms-eyebrow">{title}</div>'
        f'<div style="margin:10px 0 8px;"><span class="{pill}">{dot} {pred}</span></div>'
        f'<div class="ms-mono" style="color:{TEXT2};">conf <b style="color:{TEXT1};">'
        f'{conf_d:.1%}</b></div></div>',
        unsafe_allow_html=True)

def load_classifier():
    import os
    if not os.path.exists("classifier.pth"):
        return None
    from model.classifier import PneumoniaClassifier
    m = PneumoniaClassifier().to(DEVICE)
    m.load_state_dict(torch.load("classifier.pth", map_location=DEVICE))
    m.eval()
    return m

def load_generator():
    import os
    if not os.path.exists("generator.pth"):
        return None
    g = Generator(noise_dim=NOISE_DIM).to(DEVICE)
    g.load_state_dict(torch.load("generator.pth", map_location=DEVICE))
    g.eval()
    return g

def load_discriminator():
    import os
    if not os.path.exists("discriminator.pth"):
        return None
    d = Discriminator().to(DEVICE)
    d.load_state_dict(torch.load("discriminator.pth", map_location=DEVICE))
    d.eval()
    return d


def next_step_btn(label, target_tab):
    """Renders a styled 'next step' button that sets the target tab."""
    st.markdown('<div class="next-btn" style="margin-top:24px;display:flex;justify-content:flex-end;">', unsafe_allow_html=True)
    if st.button(f"➜  {label}", key=f"next_{target_tab}"):
        st.session_state["active_tab"] = target_tab
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────
# ── Header row: logo left, theme toggle right ────────────────────────
_hc1, _hc2 = st.columns([8, 1])
with _hc1:
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:16px;padding-bottom:10px;">
        <div style="background:linear-gradient(135deg,{ACCENT_DK},{ACCENT});
                    border-radius:14px;padding:13px 16px;font-size:26px;line-height:1;
                    box-shadow:0 4px 18px {GLOW};flex-shrink:0;">🩺</div>
        <div>
            <div style="font-size:26px;font-weight:700;color:{TEXT1};
                        letter-spacing:-0.03em;line-height:1.15;">MedShield AI</div>
            <div style="display:flex;align-items:center;gap:8px;margin-top:5px;">
                <span class="hero-pulse"></span>
                <span style="font-size:11px;color:{TEXT3};letter-spacing:0.05em;
                             text-transform:uppercase;">
                    Adversarially Robust Generative AI · Chest X-Ray Analysis
                </span>
            </div>
        </div>
    </div>
    <div style="display:flex;gap:6px;flex-wrap:wrap;padding-bottom:14px;">
        <span class="ms-tag">PyTorch</span>
        <span class="ms-tag">DCGAN</span>
        <span class="ms-tag">FGSM · PGD</span>
        <span class="ms-tag">CNN Classifier</span>
        <span class="ms-tag">Apple M4 · MPS</span>
    </div>
    """, unsafe_allow_html=True)

with _hc2:
    # Theme toggle — stores value in session_state so models survive the rerun
    # All trained models are already in session_state so rerun only changes CSS
    st.markdown('<div style="padding-top:8px;display:flex;justify-content:flex-end;">', unsafe_allow_html=True)
    st.markdown('<div class="theme-btn">', unsafe_allow_html=True)
    if st.button("☀️ Light" if dark else "🌙 Dark", key="theme_toggle"):
        st.session_state["dark_mode"] = not dark
        st.rerun()
    st.markdown('</div></div>', unsafe_allow_html=True)

st.markdown(f'<div style="height:1px;background:linear-gradient(90deg,{ACCENT}66,{BORDER},transparent);margin-bottom:6px;"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    # ── Title + device ───────────────────────────────────────────────
    _dev = ("⚡ MPS (M4)" if torch.backends.mps.is_available()
            else "🔥 CUDA" if torch.cuda.is_available() else "💻 CPU")
    st.markdown(f"""
    <div style="display:flex;align-items:center;justify-content:space-between;
                margin-bottom:16px;padding-bottom:12px;border-bottom:1px solid {BORDER};">
        <div>
            <div style="font-size:14px;font-weight:700;color:{TEXT1};">Configuration</div>
            <div style="font-size:10px;color:{TEXT3};margin-top:2px;">Set before each step</div>
        </div>
        <div style="background:{BG3};border:1px solid {BORDER};border-radius:8px;
                    padding:4px 9px;font-family:JetBrains Mono,monospace;
                    font-size:10px;font-weight:600;color:{ACCENT};">{_dev}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── GAN ──────────────────────────────────────────────────────────
    st.markdown(f'<div style="font-size:9px;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:{ACCENT};margin-bottom:4px;">GAN Training</div>', unsafe_allow_html=True)
    epochs = st.slider("Epochs", 5, 50, 10)

    # ── Attack ───────────────────────────────────────────────────────
    st.markdown(f'<div style="height:1px;background:{BORDER};margin:10px 0 8px;"></div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:9px;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:{ACCENT};margin-bottom:4px;">Attack</div>', unsafe_allow_html=True)
    epsilon   = st.slider("Strength ε", 0.01, 0.5, 0.1)
    pgd_steps = st.slider("PGD steps", 5, 50, 20)

    # ── Defense ──────────────────────────────────────────────────────
    st.markdown(f'<div style="height:1px;background:{BORDER};margin:10px 0 8px;"></div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:9px;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:{ACCENT};margin-bottom:4px;">Defense</div>', unsafe_allow_html=True)
    defense = st.selectbox("Method", ["Gaussian denoising", "Median filter", "Both"])

    # ── Pipeline status ───────────────────────────────────────────────
    st.markdown(f'<div style="height:1px;background:{BORDER};margin:10px 0 10px;"></div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:9px;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:{ACCENT};margin-bottom:8px;">Pipeline</div>', unsafe_allow_html=True)
    _steps = [
        ("generator"  in st.session_state, "01", "GAN trained"),
        ("classifier" in st.session_state, "02", "Classifier ready"),
        ("adv_img"    in st.session_state, "03", "Attack run"),
        ("defended"   in st.session_state, "04", "Defense applied"),
    ]
    _pipeline_html = ""
    for _done, _num, _lbl in _steps:
        _nb = ACCENT_DK if _done else BORDER
        _nt = "#ffffff"  if _done else TEXT3
        _lt = "#34d399"  if _done else TEXT2
        _ck = "✓"        if _done else _num
        _pipeline_html += (
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">'
            f'<div style="width:20px;height:20px;border-radius:5px;background:{_nb};'
            f'display:flex;align-items:center;justify-content:center;flex-shrink:0;'
            f'font-size:9px;font-weight:700;color:{_nt};">{_ck}</div>'
            f'<span style="font-size:12px;color:{_lt};">{_lbl}</span></div>'
        )
    st.markdown(_pipeline_html, unsafe_allow_html=True)

    # ── Cloud notice ──────────────────────────────────────────────────────
if IS_CLOUD:
    st.markdown(f"""
    <div style="background:{BG2};border:1px solid {BORDER};
                border-left:4px solid {ACCENT};border-radius:10px;
                padding:12px 18px;margin-bottom:12px;">
        <div style="font-size:13px;font-weight:600;color:{TEXT1};margin-bottom:4px;">
            ☁️ Running on Streamlit Cloud
        </div>
        <div style="font-size:12px;color:{TEXT2};">
            Pre-trained models are loaded. Use
            <b style="color:{ACCENT};">📂 Load Saved GAN</b> and
            <b style="color:{ACCENT};">📂 Load Saved Classifier</b>
            buttons to skip training and go straight to Tab 5 to upload an X-ray.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "  📊  Train GAN  ",
    "  🫁  Classifier  ",
    "  ⚔️  Attacks  ",
    "  🛡️  Defense  ",
    "  📂  Diagnose  ",
])

# ════════════════════════════════════════════════════
# TAB 1  —  TRAIN GAN
# ════════════════════════════════════════════════════
with tab1:
    ci, ca = st.columns([3, 2], gap="large")
    with ci:
        st.markdown(f'<div class="ms-tag">Step 01 · Generative Model</div>', unsafe_allow_html=True)
        st.markdown(f'<h2 style="color:{TEXT1};font-size:22px;font-weight:700;margin:0 0 14px;letter-spacing:-0.02em;">Train the GAN</h2>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="ms-card-accent">
            <div class="ms-eyebrow">What happens here</div>
            <div style="font-size:13px;color:{TEXT2};line-height:1.8;">
                A <b style="color:{TEXT1};">Deep Convolutional GAN</b> is trained on real chest X-ray images.
                The <b style="color:{TEXT1};">Generator</b> learns to create convincing fake X-rays from noise.
                The <b style="color:{TEXT1};">Discriminator</b> learns to tell them apart from real ones —
                and becomes the target of adversarial attacks in Step 3.
            </div>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
            <div style="background:{BG3};border:1px solid {BORDER};border-radius:10px;padding:14px;">
                <div class="ms-eyebrow">Architecture</div>
                <div style="font-size:13px;color:{TEXT1};font-weight:600;">DCGAN</div>
                <div style="font-size:11px;color:{TEXT3};">Conv + BatchNorm layers</div>
            </div>
            <div style="background:{BG3};border:1px solid {BORDER};border-radius:10px;padding:14px;">
                <div class="ms-eyebrow">Input size</div>
                <div style="font-size:13px;color:{TEXT1};font-weight:600;">64 × 64 px</div>
                <div style="font-size:11px;color:{TEXT3};">Grayscale X-ray</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with ca:
        st.markdown(f'<div style="height:26px;"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="ms-card" style="padding:18px 20px;">
            <div class="ms-eyebrow">Ready</div>
            <div style="font-size:13px;color:{TEXT2};margin:6px 0 14px;line-height:1.6;">
                Epochs: <b style="color:{TEXT1};">{epochs}</b> &nbsp;·&nbsp;
                Device: <b style="color:{ACCENT};">
                {"M4 GPU" if torch.backends.mps.is_available() else "CPU"}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)
        import os as _os1
        if _os1.path.exists("generator.pth") and "generator" not in st.session_state:
            st.markdown(f'<div style="background:{BG3};border:1px solid {BORDER};'
                f'border-radius:10px;padding:10px 14px;margin-bottom:10px;">'
                f'<div style="font-size:9px;font-weight:700;letter-spacing:0.1em;'
                f'text-transform:uppercase;color:{ACCENT};margin-bottom:3px;">Saved model found</div>'
                f'<div style="font-size:11px;color:{TEXT2};">Load to skip retraining</div></div>',
                unsafe_allow_html=True)
            if st.button("📂  Load Saved GAN", key="load_gan_btn"):
                _gen = load_generator(); _disc = load_discriminator()
                if _gen and _disc:
                    st.session_state["generator"]     = _gen
                    st.session_state["discriminator"] = _disc
                    st.success("✅  GAN loaded from disk")

        if IS_CLOUD:
            st.info("☁️ Training is disabled on Streamlit Cloud — no dataset available.\nUse **📂 Load Saved GAN** above to load the pre-trained model.")
        elif st.button("🚀  Start GAN Training", key="btn_train_gan"):

            prog = st.progress(0); stat = st.empty()
            def upd(v):
                prog.progress(v)
                stat.markdown(f'<div class="ms-mono" style="color:{ACCENT};">Training · {int(v*100)}%</div>', unsafe_allow_html=True)
            with st.spinner(""):
                gen, disc, g_losses, d_losses = train_gan(
                    progress_callback=upd, epochs_override=epochs)
                st.session_state.update({
                    "generator": gen, "discriminator": disc,
                    "_g_losses": g_losses, "_d_losses": d_losses
                })
            stat.empty()
            st.success("✅  Training complete — `generator.pth` saved")

    # Loss + samples
    if "generator" in st.session_state:
        g_l = st.session_state.get("_g_losses", [])
        d_l = st.session_state.get("_d_losses", [])
        if g_l:
            st.markdown('<hr class="ms-rule">', unsafe_allow_html=True)
            st.markdown(f'<div class="ms-section-label">Loss Curve</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(9, 2.8), facecolor=CHART_PL)
            style_ax(ax)
            ax.plot(g_l, color=ACCENT,    lw=2, label="Generator",     marker="o", ms=3.5)
            ax.plot(d_l, color="#f59e0b", lw=2, label="Discriminator", marker="o", ms=3.5)
            ax.legend(framealpha=0, labelcolor=TEXT2, fontsize=9)
            ax.grid(True, color=BORDER, alpha=0.4, lw=0.5)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown('<hr class="ms-rule">', unsafe_allow_html=True)
        st.markdown(f'<div class="ms-section-label">Generated X-ray Samples</div>', unsafe_allow_html=True)
        gen = st.session_state["generator"]
        gen.eval()
        with torch.no_grad():
            samples = gen(torch.randn(16, NOISE_DIM).to(DEVICE)).cpu()
        fig2, axes = plt.subplots(2, 8, figsize=(15, 4), facecolor=CHART_BG)
        fig2.subplots_adjust(wspace=0.04, hspace=0.04)
        for i, ax in enumerate(axes.flatten()):
            ax.imshow(tensor_to_np(samples[i]), cmap=XRAY_CM); ax.axis("off")
        st.pyplot(fig2); plt.close()

        # next_step_btn("Continue to Classifier →", 1)

# ════════════════════════════════════════════════════
# TAB 2  —  CLASSIFIER
# ════════════════════════════════════════════════════
with tab2:
    ci, ca = st.columns([3, 2], gap="large")
    with ci:
        st.markdown(f'<div class="ms-tag">Step 02 · Medical Diagnosis</div>', unsafe_allow_html=True)
        st.markdown(f'<h2 style="color:{TEXT1};font-size:22px;font-weight:700;margin:0 0 14px;letter-spacing:-0.02em;">Pneumonia Classifier</h2>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="ms-card-accent">
            <div class="ms-eyebrow">What happens here</div>
            <div style="font-size:13px;color:{TEXT2};line-height:1.8;">
                A <b style="color:{TEXT1};">CNN trained exclusively on real X-rays</b>
                learns to distinguish <span style="color:#34d399;">NORMAL</span> from
                <span style="color:#f87171;">PNEUMONIA</span>.
                This is your actual diagnostic model — and the one adversarial attacks
                will attempt to fool in Step 5.<br><br>
                <b style="color:{TEXT1};">Training tip:</b>
                <span style="color:{TEXT3};font-size:12px;">
                10–15 epochs is enough to reach 70–80%+ validation accuracy
                on the Kaggle chest X-ray dataset.
                </span>
            </div>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;">
            <div style="background:{BG3};border:1px solid {BORDER};border-radius:10px;padding:14px;">
                <div class="ms-eyebrow">Optimizer</div>
                <div style="font-size:13px;color:{TEXT1};font-weight:600;">Adam</div>
                <div style="font-size:11px;color:{TEXT3};">lr = 0.001</div>
            </div>
            <div style="background:{BG3};border:1px solid {BORDER};border-radius:10px;padding:14px;">
                <div class="ms-eyebrow">Loss fn</div>
                <div style="font-size:13px;color:{TEXT1};font-weight:600;">BCE</div>
                <div style="font-size:11px;color:{TEXT3};">Binary cross-entropy</div>
            </div>
            <div style="background:{BG3};border:1px solid {BORDER};border-radius:10px;padding:14px;">
                <div class="ms-eyebrow">Regularisation</div>
                <div style="font-size:13px;color:{TEXT1};font-weight:600;">Dropout</div>
                <div style="font-size:11px;color:{TEXT3};">p = 0.5</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with ca:
        st.markdown(f'<div style="height:26px;"></div>', unsafe_allow_html=True)
        import os as _os2
        if _os2.path.exists("classifier.pth") and "classifier" not in st.session_state:
            st.markdown(f'<div style="background:{BG3};border:1px solid {BORDER};'
                f'border-radius:10px;padding:10px 14px;margin-bottom:10px;">'
                f'<div style="font-size:9px;font-weight:700;letter-spacing:0.1em;'
                f'text-transform:uppercase;color:{ACCENT};margin-bottom:3px;">Saved model found</div>'
                f'<div style="font-size:11px;color:{TEXT2};">Load to skip retraining</div></div>',
                unsafe_allow_html=True)
            if st.button("📂  Load Saved Classifier", key="load_clf_btn"):
                _clf = load_classifier()
                if _clf:
                    st.session_state["classifier"] = _clf
                    st.success("✅  Classifier loaded from disk")

        clf_ep = st.slider("Epochs", 5, 30, 10, key="clf_ep")
        if IS_CLOUD:
            st.info("☁️ Training is disabled on Streamlit Cloud.\nUse **📂 Load Saved Classifier** above to load the pre-trained model.")
        elif st.button("🫁  Train Classifier", key="btn_train_clf"):

            prog = st.progress(0); stat = st.empty()
            def cu(v):
                prog.progress(v)
                stat.markdown(f'<div class="ms-mono" style="color:{ACCENT};">Training · {int(v*100)}%</div>', unsafe_allow_html=True)
            with st.spinner(""):
                clf, tl, va = train_classifier(
                    progress_callback=cu, epochs_override=clf_ep)
                st.session_state.update({
                    "classifier": clf, "_tlosses": tl, "_vaccs": va
                })
            stat.empty()
            st.success("✅  Classifier trained — `classifier.pth` saved")
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Val accuracy",  f"{va[-1]:.1f}%")
            mc2.metric("Train loss",    f"{tl[-1]:.4f}")
            mc3.metric("Epochs",        clf_ep)

    if "_tlosses" in st.session_state:
        tl = st.session_state["_tlosses"]
        va = st.session_state["_vaccs"]
        st.markdown('<hr class="ms-rule">', unsafe_allow_html=True)
        st.markdown(f'<div class="ms-section-label">Training Curves</div>', unsafe_allow_html=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3), facecolor=CHART_PL)
        for ax in (ax1, ax2):
            style_ax(ax); ax.grid(True, color=BORDER, alpha=0.4, lw=0.5)
        ax1.plot(tl, color=ACCENT,    lw=2, marker="o", ms=3.5)
        ax1.set_title("Training loss",       color=TEXT2, fontsize=10, pad=8)
        ax2.plot(va, color="#34d399", lw=2, marker="o", ms=3.5)
        ax2.set_title("Validation accuracy", color=TEXT2, fontsize=10, pad=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        if va[-1] < 65:
            st.warning("⚠️  Accuracy below 65% — try training for more epochs (15–20) for better diagnosis results.")

        # next_step_btn("Continue to Adversarial Attacks →", 2)

# ════════════════════════════════════════════════════
# TAB 3  —  ATTACKS
# ════════════════════════════════════════════════════
with tab3:
    st.markdown(f'<div class="ms-tag">Step 03 · Adversarial Robustness</div>', unsafe_allow_html=True)
    st.markdown(f'<h2 style="color:{TEXT1};font-size:22px;font-weight:700;margin:0 0 4px;letter-spacing:-0.02em;">Adversarial Attacks</h2>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:22px;">
        <div class="ms-card" style="margin:0;border-left:3px solid #f59e0b;">
            <div class="ms-eyebrow" style="color:#f59e0b;">FGSM</div>
            <div style="font-size:13px;color:{TEXT1};font-weight:600;margin-bottom:4px;">Fast Gradient Sign Method</div>
            <div style="font-size:12px;color:{TEXT2};line-height:1.6;">
                One large step in the gradient direction scaled by ε.
                Fast to compute, good for testing model vulnerability.
            </div>
        </div>
        <div class="ms-card" style="margin:0;border-left:3px solid #ef4444;">
            <div class="ms-eyebrow" style="color:#ef4444;">PGD</div>
            <div style="font-size:13px;color:{TEXT1};font-weight:600;margin-bottom:4px;">Projected Gradient Descent</div>
            <div style="font-size:12px;color:{TEXT2};line-height:1.6;">
                Many small FGSM steps projected back into the ε-ball.
                The strongest first-order attack — harder to defend against.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if "discriminator" not in st.session_state:
        st.warning("⚠️  Train the GAN in Step 01 first.")
    else:
        a1, a2 = st.columns([1, 2], gap="large")
        with a1:
            attack_type = st.radio("Algorithm", ["FGSM", "PGD", "Both"])
            st.markdown(f"""
            <div style="background:{BG3};border:1px solid {BORDER};border-radius:10px;padding:12px 16px;margin-top:12px;">
                <div class="ms-eyebrow" style="margin-bottom:6px;">Current settings</div>
                <div class="ms-mono">ε = {epsilon}</div>
                <div class="ms-mono">PGD steps = {pgd_steps}</div>
            </div>
            """, unsafe_allow_html=True)
            run_atk = st.button("⚔️  Run Attack", key="btn_run_attack")

        with a2:
            if run_atk:
                disc = st.session_state["discriminator"]
                gen  = st.session_state["generator"]
                noise    = torch.randn(1, NOISE_DIM).to(DEVICE)
                fake_img = gen(noise).detach()
                label    = torch.zeros(1, 1).to(DEVICE)
                results  = {}
                with st.spinner("Crafting adversarial examples…"):
                    if attack_type in ["FGSM", "Both"]:
                        results["FGSM"] = fgsm_attack(disc, fake_img, label, epsilon=epsilon)
                    if attack_type in ["PGD", "Both"]:
                        results["PGD"]  = pgd_attack(disc, fake_img, label,
                                                      epsilon=epsilon, num_steps=pgd_steps)

                disc.eval()
                with torch.no_grad():
                    orig_sc = disc(fake_img.to(DEVICE)).item()

                cols = st.columns(1 + len(results), gap="small")
                with cols[0]:
                    show_xray(cols[0], fake_img, "Original (GAN)")
                    st.markdown(score_bar(orig_sc, "Discriminator score"), unsafe_allow_html=True)
                    st.markdown(f'<div style="font-size:11px;color:{TEXT3};margin-top:3px;">1.0 = real · 0.0 = fake</div>', unsafe_allow_html=True)

                for i, (name, adv_img) in enumerate(results.items()):
                    with cols[i + 1]:
                        show_xray(cols[i + 1], adv_img, f"{name} adversarial")
                        with torch.no_grad():
                            adv_sc = disc(adv_img).item()
                        delta = adv_sc - orig_sc
                        dc    = "#34d399" if delta > 0.05 else "#f87171" if delta < -0.05 else TEXT2
                        st.markdown(score_bar(adv_sc, f"Score after {name}"), unsafe_allow_html=True)
                        st.markdown(f'<div style="font-size:11px;color:{dc};margin-top:3px;">{"▲" if delta>0 else "▼"} {abs(delta):.3f} shift</div>', unsafe_allow_html=True)

                adv_img = list(results.values())[-1]
                adv_score = disc(adv_img).item() if adv_img is not None else None

                st.session_state.update({
                    "fake_img": fake_img,
                    "adv_img":  list(results.values())[-1],
                    "label":    label,
                    "_atk_type":  attack_type,
                    "_atk_orig_sc": orig_sc,
                    "_atk_adv_sc": (
                    disc(list(results.values())[-1]).item()
                    if list(results.values())[-1] is not None
                    else None
                    )
                })
                # next_step_btn("Continue to Defense →", 3)
            else:
                st.markdown(f"""
                <div style="height:200px;border:1.5px dashed {BORDER2};border-radius:12px;
                            display:flex;align-items:center;justify-content:center;">
                    <div style="text-align:center;color:{TEXT3};">
                        <div style="font-size:30px;margin-bottom:8px;">⚔️</div>
                        <div style="font-size:13px;">Attack results will appear here</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════
# TAB 4  —  DEFENSE
# ════════════════════════════════════════════════════
with tab4:
    st.markdown(f'<div class="ms-tag">Step 04 · Robustness Framework</div>', unsafe_allow_html=True)
    st.markdown(f'<h2 style="color:{TEXT1};font-size:22px;font-weight:700;margin:0 0 14px;letter-spacing:-0.02em;">Defense Mechanisms</h2>', unsafe_allow_html=True)

    if "adv_img" not in st.session_state:
        st.warning("⚠️  Run an attack in Step 03 first.")
    else:
        d1, d2 = st.columns([1, 2], gap="large")
        with d1:
            st.markdown(f"""
            <div class="ms-card-accent">
                <div class="ms-eyebrow">Method: {defense}</div>
                <div style="font-size:13px;color:{TEXT2};line-height:1.7;margin-top:6px;">
                    {"Gaussian noise injection + clamp — removes high-frequency adversarial perturbations."
                     if defense=="Gaussian denoising"
                     else "Average pooling smooths pixel-level noise effectively."
                     if defense=="Median filter"
                     else "Combines both methods for maximum noise removal."}
                </div>
            </div>
            """, unsafe_allow_html=True)
            apply_def = st.button("🛡️  Apply Defense", key="btn_apply_def")

        with d2:
            if apply_def:
                adv_img = st.session_state["adv_img"]
                disc    = st.session_state["discriminator"]
                with st.spinner("Applying defense…"):
                    if defense == "Gaussian denoising":
                        defended = gaussian_denoise(adv_img)
                    elif defense == "Median filter":
                        defended = median_filter_defense(adv_img)
                    else:
                        defended = median_filter_defense(gaussian_denoise(adv_img))

                disc.eval()
                with torch.no_grad():
                    adv_sc = disc(adv_img).item()
                    def_sc = disc(defended).item()

                ic1, ic2 = st.columns(2)
                show_xray(ic1, adv_img,  "Before defense")
                show_xray(ic2, defended, "After defense")

                st.markdown('<hr class="ms-rule" style="margin:14px 0;">', unsafe_allow_html=True)
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Score before", f"{adv_sc:.4f}")
                mc2.metric("Score after",  f"{def_sc:.4f}", delta=f"{def_sc-adv_sc:+.4f}")
                mc3.metric("Recovery",     f"{abs(def_sc-adv_sc):.4f}")

                det    = detect_adversarial(disc, adv_img)
                is_adv = det["is_adversarial"]
                bg_d   = "#1a0a00" if is_adv else "#001a0f"
                bd_d   = "#7f1d1d" if is_adv else "#065f46"
                co_d   = "#f87171" if is_adv else "#34d399"
                ic_d   = "⚠️" if is_adv else "✅"
                lb_d   = "ADVERSARIAL DETECTED" if is_adv else "IMAGE APPEARS CLEAN"

                st.markdown(f"""
                <div style="background:{bg_d};border:1px solid {bd_d};border-left:4px solid {co_d};
                            border-radius:12px;padding:18px 22px;margin-top:8px;">
                    <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
                        <span style="font-size:20px;">{ic_d}</span>
                        <span style="font-size:14px;font-weight:700;color:{co_d};">{lb_d}</span>
                    </div>
                    <div style="display:flex;gap:24px;flex-wrap:wrap;">
                        <div><div class="ms-eyebrow" style="color:{TEXT3};">Orig score</div>
                             <div class="ms-mono" style="color:{TEXT1};">{det["original_score"]}</div></div>
                        <div><div class="ms-eyebrow" style="color:{TEXT3};">Smoothed</div>
                             <div class="ms-mono" style="color:{TEXT1};">{det["smoothed_score"]}</div></div>
                        <div><div class="ms-eyebrow" style="color:{TEXT3};">Conf. drop</div>
                             <div class="ms-mono" style="color:{co_d};">{det["confidence_drop"]}</div></div>
                        <div><div class="ms-eyebrow" style="color:{TEXT3};">Threshold</div>
                             <div class="ms-mono" style="color:{TEXT1};">{det["threshold"]}</div></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.session_state.update({
                    "defended": defended,
                    "_def_adv_sc": adv_sc,
                    "_def_def_sc": def_sc,
                    "_def_detection": det,
                    "_def_method": defense
                })
                # next_step_btn("Continue to Diagnose →", 4)
            else:
                st.markdown(f"""
                <div style="height:220px;border:1.5px dashed {BORDER2};border-radius:12px;
                            display:flex;align-items:center;justify-content:center;">
                    <div style="text-align:center;color:{TEXT3};">
                        <div style="font-size:30px;margin-bottom:8px;">🛡️</div>
                        <div style="font-size:13px;">Click "Apply Defense" to see results</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════
# TAB 5  —  DIAGNOSE  (classifier fix: use the real classifier)
# ════════════════════════════════════════════════════
with tab5:
    st.markdown(f'<div class="ms-tag">Step 05 · Clinical Demo</div>', unsafe_allow_html=True)
    st.markdown(f'<h2 style="color:{TEXT1};font-size:22px;font-weight:700;margin:0 0 4px;letter-spacing:-0.02em;">Upload & Diagnose</h2>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:13px;color:{TEXT3};margin-bottom:20px;">Upload a chest X-ray to run the full adversarial pipeline and see whether the diagnosis holds under attack.</div>', unsafe_allow_html=True)

    missing = []
    if "discriminator" not in st.session_state: missing.append("GAN (Tab 1)")
    if "classifier"    not in st.session_state: missing.append("Classifier (Tab 2)")
    if missing:
        st.warning(f"Complete first: {', '.join(missing)}")
    else:
        # ── accuracy reminder ──
        va = st.session_state.get("_vaccs", [])
        if va and va[-1] < 70:
            st.info(f"ℹ️  Classifier accuracy is {va[-1]:.1f}%. For reliable diagnosis, retrain for more epochs to push above 75%.")

    up1, up2 = st.columns([1, 2], gap="large")
    with up1:
        uploaded = st.file_uploader("Drop a chest X-ray — JPG or PNG",
                                    type=["jpg", "jpeg", "png"])
        if uploaded:
            st.image(uploaded, caption="Uploaded X-ray", use_container_width=True)

    with up2:
        if uploaded and "classifier" in st.session_state:
            img        = Image.open(uploaded).convert("L").resize((64, 64))
            tfm        = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5,), (0.5,))])
            img_tensor = tfm(img).unsqueeze(0)
            label      = torch.zeros(1, 1)
            clf        = st.session_state["classifier"]

            pred, conf = predict_single_image(clf, img_tensor)
            conf_d     = conf if pred == "PNEUMONIA" else 1 - conf
            pill       = "diag-pneumonia" if pred == "PNEUMONIA" else "diag-normal"
            dot        = "🔴" if pred == "PNEUMONIA" else "🟢"

            st.markdown(f"""
            <div class="ms-card-accent">
                <div class="ms-eyebrow">Initial Diagnosis</div>
                <div style="margin:12px 0;">
                    <span class="{pill}">{dot} {pred}</span>
                </div>
                {score_bar(conf_d, "Confidence")}
                <div style="font-size:11px;color:{TEXT3};margin-top:8px;">
                    Raw model output: <span style="font-family:JetBrains Mono;color:{ACCENT};">{conf:.4f}</span>
                    &nbsp;·&nbsp; Threshold: 0.50
                </div>
            </div>
            """, unsafe_allow_html=True)

            if not missing:
                if st.button("🚀  Run Full Adversarial Pipeline", key="btn_run_pipeline"):
                    disc = st.session_state["discriminator"]

                    with st.spinner("Running pipeline…"):
                        adv     = fgsm_attack(disc, img_tensor, label, epsilon=epsilon)
                        def_img = gaussian_denoise(adv)

                    st.markdown('<hr class="ms-rule">', unsafe_allow_html=True)
                    st.markdown(f'<div class="ms-section-label">Visual Comparison</div>', unsafe_allow_html=True)
                    vc1, vc2, vc3 = st.columns(3, gap="small")
                    show_xray(vc1, img_tensor, "Original")
                    show_xray(vc2, adv,        f"FGSM (ε={epsilon})")
                    show_xray(vc3, def_img,    "Defended")

                    st.markdown('<hr class="ms-rule">', unsafe_allow_html=True)
                    st.markdown(f'<div class="ms-section-label">Diagnosis Under Attack</div>', unsafe_allow_html=True)

                    pred_orig, co = predict_single_image(clf, img_tensor)
                    pred_adv,  ca = predict_single_image(clf, adv)
                    pred_def,  cd = predict_single_image(clf, def_img)

                    dc1, dc2, dc3 = st.columns(3, gap="small")
                    diag_card(dc1, "Original",      pred_orig, co)
                    diag_card(dc2, "After attack",  pred_adv,  ca)
                    diag_card(dc3, "After defense", pred_def,  cd)

                    st.markdown('<hr class="ms-rule">', unsafe_allow_html=True)
                    if pred_orig != pred_adv:
                        st.markdown(f"""
                        <div style="background:#200a00;border:1px solid #7c2d12;
                                    border-left:4px solid #ef4444;border-radius:12px;padding:20px 24px;">
                            <div style="font-size:15px;font-weight:700;color:#f87171;margin-bottom:6px;">
                                ⚠️  Attack changed the diagnosis
                            </div>
                            <div style="font-size:13px;color:{TEXT2};line-height:1.7;">
                                <b style="color:#fbbf24;">{pred_orig}</b> →
                                <b style="color:#f87171;">{pred_adv}</b>
                                — An invisible perturbation flipped the clinical outcome.
                                This demonstrates why adversarial robustness is non-negotiable
                                in medical AI deployment.
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background:#001a0f;border:1px solid #065f46;
                                    border-left:4px solid #34d399;border-radius:12px;padding:20px 24px;">
                            <div style="font-size:15px;font-weight:700;color:#34d399;margin-bottom:6px;">
                                ✅  Model held its diagnosis under attack
                            </div>
                            <div style="font-size:13px;color:{TEXT2};line-height:1.7;">
                                Remained <b style="color:#34d399;">{pred_orig}</b> despite
                                adversarial perturbation — demonstrating robust behaviour.
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    det = detect_adversarial(disc, adv)
                    st.markdown('<hr class="ms-rule">', unsafe_allow_html=True)
                    st.markdown(f'<div class="ms-section-label">Detection Report</div>', unsafe_allow_html=True)
                    r1, r2, r3, r4 = st.columns(4)
                    r1.metric("Adversarial",     "YES" if det["is_adversarial"] else "NO")
                    r2.metric("Original score",  det["original_score"])
                    r3.metric("Smoothed score",  det["smoothed_score"])
                    r4.metric("Confidence drop", det["confidence_drop"])

# ─────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="footer-wrap">
    <div>
        <div style="font-size:10px;font-weight:700;letter-spacing:0.12em;
                    text-transform:uppercase;color:{TEXT3};margin-bottom:8px;">Built by</div>
        <div style="font-size:18px;font-weight:700;color:{TEXT1};letter-spacing:-0.01em;margin-bottom:4px;">
            Himanshu Ranjan</div>
        <div style="font-size:12px;color:{TEXT3};line-height:1.7;">
            Adversarially Robust Generative AI · Chest X-Ray Analysis<br>
            PyTorch · Streamlit · Apple Silicon M4
        </div>
    </div>
    <div style="display:flex;flex-direction:column;align-items:flex-end;gap:10px;">
        <a class="footer-li-btn"
           href="https://www.linkedin.com/in/himanshuranjan1241" target="_blank">
            <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15"
                 viewBox="0 0 24 24" fill="currentColor">
                <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037
                         -1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046
                         c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286z
                         M5.337 7.433a2.062 2.062 0 01-2.063-2.065 2.064 2.064 0 112.063 2.065zm1.782
                         13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542
                         C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729
                         C24 .774 23.2 0 22.222 0h.003z"/>
            </svg>
            Connect on LinkedIn
        </a>
        <div style="font-size:10px;color:{TEXT3};text-align:right;">
            For educational &amp; research use only<br>Not for clinical deployment
        </div>
    </div>
</div>
""", unsafe_allow_html=True)