import base64
import html
import io

import streamlit as st
from PIL import Image

from disease_detection import predict_plant
from similarity import search_similar_images
from recommendations import generate_smart_recommendations
from localization import class_label, localize_recommendation


# Translation helper. Defined first so even the browser-tab title can be localized.
def tr(english, arabic):
    return arabic if st.session_state.get("language", "English") == "العربية" else english


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title=tr("GreenLens - Plant Disease Detection", "GreenLens - تشخيص أمراض النباتات"),
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAV_HOME = "🏠  Home"
NAV_ABOUT = "ℹ️  About"
NAV_SETTINGS = "⚙️  Settings"
NAV_ITEMS = [NAV_HOME, NAV_ABOUT, NAV_SETTINGS]


# --------------------------------------------------
# Session State
# --------------------------------------------------
defaults = {
    "page": NAV_HOME,
    "prediction_result": None,   # نتيجة التشخيص (صفحة Home)
    "home_similar": None,        # الصور المشابهة اللي بتظهر في Home
    "home_sig": None,            # بصمة الصورة الحالية (عشان نمسح النتايج لو الصورة اتغيرت)
    "show_all_similar": False,   # View All / Show Less
    "top_k": 12,                 # عدد الصور المشابهة اللي بنجيبها
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Allowed range for the number of similar images. st.session_state.top_k is the
# ONE value used by the similarity search AND shown in Settings.
TOP_K_MIN, TOP_K_MAX = 3, 12


def clamp_top_k(value) -> int:
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = defaults["top_k"]
    return max(TOP_K_MIN, min(TOP_K_MAX, value))


def change_top_k(delta: int) -> None:
    """Button callback: runs before the rerun, so the UI always shows the new value."""
    st.session_state.top_k = clamp_top_k(st.session_state.top_k + delta)


st.session_state.top_k = clamp_top_k(st.session_state.top_k)


# --------------------------------------------------
# Custom CSS
# --------------------------------------------------
st.markdown(
    """
    <style>
    .arabic-ui { direction: rtl; text-align: right; }
    .arabic-ui .tb-left, .arabic-ui .card-head, .arabic-ui .pred-left,
    .arabic-ui .care-item { direction: rtl; }
    .arabic-ui [data-testid="stFileUploader"] { direction: rtl; }
    .arabic-ui .ring-wrap { direction: ltr; }

    /* ---------- Settings: similar-images stepper ---------- */
    .stepper-label { font-weight: 700; color: var(--ink); margin: .2rem 0 .6rem; }
    .stApp .st-key-top_k_stepper { max-width: 360px; }
    /* Keep [ - ] 12 [ + ] in the same order in English AND Arabic (like digits, never mirrored)
       and never let Streamlit stack the three columns on narrow screens. */
    .stApp .st-key-top_k_stepper [data-testid="stHorizontalBlock"] {
        direction: ltr;
        flex-wrap: nowrap !important;
        align-items: center;
        gap: .5rem;
        background: #fbfefc;
        border: 1px solid #cfe5d4;
        border-radius: 18px;
        padding: .45rem;
        box-shadow: 0 2px 8px rgba(20, 83, 45, .05);
    }
    .stApp .st-key-top_k_stepper [data-testid="stHorizontalBlock"] > div { min-width: 0 !important; }
    /* descendant selector (not ">") so it still matches if Streamlit wraps the button in another element */
    .stApp .st-key-top_k_stepper .stButton button {
        min-height: 3rem;
        border-radius: 12px;
        color: var(--green-800) !important;
        background: var(--green-50);
        border: 1px solid #cfe5d4;
        box-shadow: none;
    }
    .stApp .st-key-top_k_stepper .stButton button p { font-size: 1.7rem; font-weight: 800; line-height: 1; }
    .stApp .st-key-top_k_stepper .stButton button:hover:not(:disabled) {
        background: var(--green-100); border-color: var(--green-600);
    }
    .stApp .st-key-top_k_stepper .stButton button:disabled { opacity: .4; }
    /* Streamlit gives markdown containers margin-bottom:-1rem, which would shift the value off-centre */
    .stApp .st-key-top_k_stepper [data-testid="stMarkdownContainer"] { margin-bottom: 0 !important; }
    .stepper-value {
        display: flex; align-items: center; justify-content: center;
        height: 3rem; direction: ltr;
        font-size: 1.9rem; font-weight: 800; line-height: 1; color: var(--green-900);
    }
    :root {
        --green-950: #0b3a20;
        --green-900: #14532d;
        --green-800: #166534;
        --green-700: #15803d;
        --green-600: #16a34a;
        --green-100: #dcfce7;
        --green-50: #f0fdf4;
        --ink: #14231a;
        --muted: #607166;
        --line: #e0ece3;
    }

    /* ---------- App shell ---------- */
    .stApp { background: #f3f8f4; color: var(--ink); }
    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }
    .main .block-container { max-width: 1400px; padding: 1rem 1.8rem 2rem; }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f4a27 0%, #0b3a20 55%, #082c18 100%);
        border-right: 0;
    }

    
    /* ---------- Arabic bullet lists ---------- */
    .stApp ul,
    .stApp ol {
        direction: rtl !important;
        text-align: right !important;
        padding-right: 1.2rem !important;
        padding-left: 0 !important;
        list-style-position: outside !important;
    }
    
    .stApp ul li,
    .stApp ol li {
        text-align: right !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    .stApp ul li p,
    .stApp ol li p {
        margin: 0 !important;
        padding: 0 !important;
    }

    [data-testid="stSidebar"] * { color: #e8f5ec; }
    [data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }

    .brand { display: flex; align-items: center; gap: .65rem; margin-bottom: .2rem; }
    .brand .logo { font-size: 2.2rem; line-height: 1; }
    .brand .name { font-size: 1.9rem; font-weight: 800; letter-spacing: -.02em; color: #fff; }
    .brand-tag { font-size: .8rem; color: #b6d9c1; margin-block: 0 1.4rem; margin-inline: .2rem 0; }

    [data-testid="stSidebar"] div[role="radiogroup"] { gap: .35rem; }
    [data-testid="stSidebar"] div[role="radiogroup"] label {
        width: 100%;
        padding: .7rem .9rem;
        border-radius: 12px;
        cursor: pointer;
        transition: background .15s ease;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child { display: none; }
    /* Newer Streamlit: the radio circle is the first child of the label's inner div */
    [data-testid="stSidebar"] label[data-testid="stRadioOption"] > div > div:first-child { display: none; }
    /* The sidebar-wide light text colour made the selected language invisible (light on white) */
    [data-testid="stSidebar"] [data-testid="stSelectbox"] input {
        color: var(--ink) !important; -webkit-text-fill-color: var(--ink) !important;
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] svg { fill: var(--green-800) !important; }
    [data-testid="stSidebar"] div[role="radiogroup"] label:hover { background: rgba(255,255,255,.08); }
    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(100deg, #1b7a3f, #15803d);
        box-shadow: inset 0 0 0 1px rgba(255,255,255,.10);
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label p { font-size: 1rem; font-weight: 600; }

    .side-footer { margin-top: 3rem; color: #b6d9c1; font-size: .92rem; line-height: 1.5; }

    /* ---------- Top bar ---------- */
    .topbar {
        display: flex; align-items: center; justify-content: space-between;
        background: #fff; border: 1px solid var(--line); border-radius: 18px;
        padding: .9rem 1.3rem; margin-bottom: 1rem;
        box-shadow: 0 4px 16px rgba(20, 83, 45, .05);
    }
    .tb-left { display: flex; align-items: center; gap: .9rem; }
    .tb-icon {
        width: 46px; height: 46px; border-radius: 14px; background: var(--green-100);
        display: grid; place-items: center; font-size: 1.5rem;
    }
    .tb-title { font-size: 1.35rem; font-weight: 800; color: var(--ink); line-height: 1.2; }
    .tb-sub { font-size: .9rem; color: var(--muted); }
    .tb-user { display: flex; align-items: center; gap: .55rem; font-weight: 600; color: var(--ink); }
    .tb-user .av {
        width: 38px; height: 38px; border-radius: 50%; background: var(--green-800);
        display: grid; place-items: center; color: #fff;
    }

    /* ---------- Cards (st.container(border=True)) ---------- */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #fff;
        border: 1px solid var(--line) !important;
        border-radius: 20px !important;
        box-shadow: 0 6px 22px rgba(20, 83, 45, .06);
    }
    .card-head { display: flex; align-items: center; gap: .85rem; margin-bottom: .8rem; }
    .card-head .ic {
        width: 48px; height: 48px; border-radius: 14px; background: var(--green-100);
        display: grid; place-items: center; font-size: 1.4rem; flex-shrink: 0;
    }
    .card-head .t { font-size: 1.45rem; font-weight: 800; color: var(--ink); line-height: 1.15; }
    .card-head .s { font-size: .9rem; color: var(--muted); }

    /* ---------- Uploader ---------- */
    [data-testid="stFileUploader"] section {
        background: #fafdfb;
        border: 1.5px dashed #9ac8a6;
        border-radius: 16px;
    }
    [data-testid="stFileUploader"] button {
        background: var(--green-800); color: #fff; border: 0; border-radius: 10px;
    }
    [data-testid="stImage"] img { border-radius: 16px; }

    /* ---------- Buttons ---------- */
    .stButton > button {
    border-radius: 12px;
    min-height: 2.9rem;
    font-weight: 700;
    color: #166534 !important;
    background: #ffffff;
    border: 1px solid #cfe5d4;
    transition: transform .15s ease, box-shadow .15s ease;
    }

    .stButton > button[kind="primary"],
    .stButton > button[kind="primary"] p {
        color: #fff !important;
        font-size: 1.3rem !important;
        font-weight: 900 !important;
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(100deg, #166534, #1a8b46);
        border: 0;
        padding: 0.4rem 1rem;
        box-shadow: 0 6px 16px rgba(22, 101, 52, .22);
    }

    /* ---------- Prediction result ---------- */
    .pred-box {
        display: flex; align-items: center; justify-content: space-between; gap: 1rem;
        background: #fbfefc; border: 1px solid var(--line); border-radius: 16px;
        padding: .9rem 1.1rem; margin-top: .6rem;
    }
    .pred-left { display: flex; align-items: center; gap: .9rem; min-width: 0; }
    .pred-left img { width: 74px; height: 74px; object-fit: cover; border-radius: 12px; }
    .pred-name { font-size: 1.3rem; font-weight: 800; color: var(--ink); overflow-wrap: anywhere; }
    .pred-plant { color: var(--muted); font-size: .95rem; margin-bottom: .25rem; }
    .badge {
        display: inline-block; padding: .18rem .7rem; border-radius: 999px;
        font-size: .8rem; font-weight: 700;
    }
    .badge.warn { background: #fff1e0; color: #c2570c; }
    .badge.ok { background: var(--green-100); color: var(--green-800); }

    .ring-wrap { text-align: center; flex-shrink: 0; }
    .ring { width: 96px; height: 96px; border-radius: 50%; display: grid; place-items: center; }
    .ring-in {
        width: 76px; height: 76px; border-radius: 50%; background: #fff;
        display: grid; place-items: center; font-size: 1.25rem; font-weight: 800; color: var(--green-900);
    }
    .ring-cap { font-size: .8rem; color: var(--muted); margin-top: .25rem; }

    .sec-title { font-weight: 800; margin: 1rem 0 .5rem; color: var(--ink); }
    .other-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: .6rem; }
    .other-item {
        border: 1px solid var(--line); border-radius: 12px; padding: .65rem .8rem; background: #fff;
    }
    .other-item .n { font-weight: 650; font-size: .92rem; overflow-wrap: anywhere; }
    .other-item .p { color: var(--muted); font-size: .88rem; }

    /* ---------- Care card ---------- */
    .info-row {
        display: grid; grid-template-columns: repeat(3, 1fr); gap: .5rem;
        background: #f3f8f4; border-radius: 14px; padding: .9rem 1rem; margin-bottom: 1rem;
    }
    .info-item .l { color: var(--muted); font-size: .8rem; }
    .info-item .v { font-weight: 800; font-size: 1rem; color: var(--ink); overflow-wrap: anywhere; }
    .care-title { font-size: 1.15rem; font-weight: 800; margin-bottom: .6rem; }
    .care-item { display: flex; gap: .7rem; align-items: flex-start; margin-bottom: .75rem; line-height: 1.45; }
    .care-item .ck {
        width: 22px; height: 22px; border-radius: 50%; background: var(--green-700); color: #fff;
        display: grid; place-items: center; font-size: .75rem; flex-shrink: 0; margin-top: .1rem;
    }
    .note-box {
        background: #f0f8f2; border: 1px solid var(--line); border-radius: 14px;
        padding: .85rem 1rem; color: #3f5546; font-size: .92rem; margin-top: 1rem;
    }
    .empty-box {
        background: #f7fbf8; border: 1px dashed #bcd9c4; border-radius: 14px;
        padding: 1.4rem 1rem; color: var(--muted); text-align: center;
    }

    /* ---------- Similar plants ---------- */
    .sim-name { font-weight: 700; font-size: .88rem; margin-top: .35rem; overflow-wrap: anywhere; }
    .sim-score {
        display: inline-block; margin-top: .25rem; background: var(--green-100); color: var(--green-800);
        border-radius: 999px; padding: .1rem .6rem; font-size: .78rem; font-weight: 600;
    }

    @media (max-width: 900px) {
        .main .block-container { padding: .8rem 1rem 2rem; }
        .info-row, .other-grid { grid-template-columns: 1fr; }
        .pred-box { flex-direction: column; align-items: flex-start; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Helpers (UI only - لا علاقة لها بمنطق المشروع)
# --------------------------------------------------
def esc(value) -> str:
    return html.escape(str(value))


def pretty(name: str) -> str:
    return str(name).replace("_", " ")


def ltr(text) -> str:
    """Keep numbers/percentages readable inside Arabic (RTL) sentences."""
    return f'<bdi dir="ltr">{esc(text)}</bdi>'


def image_to_data_uri(img: Image.Image, size: int = 120) -> str:
    thumb = img.copy()
    thumb.thumbnail((size, size))
    buffer = io.BytesIO()
    thumb.save(buffer, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def card_header(icon: str, title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="card-head"><div class="ic">{icon}</div>'
        f'<div><div class="t">{esc(title)}</div><div class="s">{esc(subtitle)}</div></div></div>',
        unsafe_allow_html=True,
    )


def open_image(uploaded):
    try:
        return Image.open(uploaded).convert("RGB")
    except Exception:
        st.error(tr("Could not open this image. Please upload a valid image file.", "تعذّر فتح الصورة. ارفع ملف صورة صالحًا."))
        return None


def toggle_view_all():
    st.session_state.show_all_similar = not st.session_state.show_all_similar


def render_similar_grid(results, per_row: int = 6) -> None:
    for start in range(0, len(results), per_row):
        row = results[start:start + per_row]
        cols = st.columns(per_row, gap="small")
        for col, item in zip(cols, row):
            with col:
                st.image(item["path"], use_container_width=True)
                score = max(0.0, min(1.0, float(item["similarity"])))
                st.markdown(
                    f'<div class="sim-name">🍃 {esc(display_class_name(item["class_name"]))}</div>'
                    f'<div class="sim-score">{ltr(format(score, ".0%"))} {tr("similarity", "تشابه")}</div>',
                    unsafe_allow_html=True,
                )


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
if "language" not in st.session_state:
    st.session_state.language = "English"

with st.sidebar:
    st.markdown(
        '<div class="brand"><span class="logo">🌿</span><span class="name">GreenLens</span></div>'
        f'<div class="brand-tag">{tr("Healthy Plants &nbsp;•&nbsp; Better Tomorrow", "نباتات سليمة &nbsp;•&nbsp; غدٌ أفضل")}</div>',
        unsafe_allow_html=True,
    )

    


    st.markdown("""
    <style>
    /* Language button: match the sidebar navigation style */
    div.st-key-language_toggle button {
        width: 100% !important;
        min-height: 44px !important;
        padding: 0.65rem 1rem !important;
        border-radius: 10px !important;
        background-color: #245C43 !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        box-shadow: none !important;
    }

    div.st-key-language_toggle button p,
    div.st-key-language_toggle button span,
    div.st-key-language_toggle button div {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }

    div.st-key-language_toggle button:hover {
        background-color: #347653 !important;
        border-color: rgba(255, 255, 255, 0.3) !important;
    }

    /* Consistent navigation width */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] {
        width: 100% !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stRadio"] label {
        width: 100% !important;
        box-sizing: border-box !important;
        border-radius: 10px !important;
        padding: 0.65rem 1rem !important;
        min-height: 44px !important;
    }
    </style>
    """, unsafe_allow_html=True)



    button_label = (
      "🌐 English"
      if st.session_state.language == "العربية"
      else "🌐 العربية"
    )

    if st.button(button_label, key="language_toggle"):
        st.session_state.language = (
            "English"
            if st.session_state.language == "العربية"
            else "العربية"
        )
        st.rerun()

    nav_labels = {
        NAV_HOME: tr("🏠  Home", "🏠  الرئيسية"),
        NAV_ABOUT: tr("ℹ️  About GreenLens", "ℹ️  عن GreenLens"),
        NAV_SETTINGS: tr("⚙️  Settings", "⚙️  الإعدادات"),
    }
    current_index = NAV_ITEMS.index(st.session_state.page) if st.session_state.page in NAV_ITEMS else 0
    # Option VALUES stay constant; only the displayed text is translated, so
    # switching language can never lose the selected page.
    st.session_state.page = st.radio(
        tr("Navigation", "التنقل"), NAV_ITEMS, index=current_index,
        format_func=lambda item: nav_labels[item],
        label_visibility="collapsed", key="navigation_display",
    )

    st.markdown(
        f'<div class="side-footer">{tr("Healthy plants<br>support a healthier<br>planet 🌱", "النباتات السليمة<br>تدعم كوكبًا<br>أكثر صحة 🌱")}</div>',
        unsafe_allow_html=True,
    )

selected_page = st.session_state.page
is_arabic = st.session_state.language == "العربية"

# Apply RTL to the app layout when Arabic is selected; preserve LTR for English.
st.markdown(
    f"""<style>
    .stApp, .stApp [data-testid="stAppViewContainer"] {{ direction: {'rtl' if is_arabic else 'ltr'}; }}
    .stApp [data-testid="stAppViewContainer"] .main,
    .stApp [data-testid="stMain"] {{ text-align: {'right' if is_arabic else 'left'}; }}
    .stApp [data-testid="stHorizontalBlock"] {{ direction: {'rtl' if is_arabic else 'ltr'}; }}
    [data-testid="stSidebar"] {{ direction: {'rtl' if is_arabic else 'ltr'}; text-align: {'right' if is_arabic else 'left'}; }}
    /* Streamlit hard-codes text-align:left on these containers, which defeats RTL inheritance.
       "start" = left in LTR and right in RTL, so English is unchanged. */
    [data-testid="stMain"] [data-testid="stElementContainer"],
    [data-testid="stMain"] [data-testid="stMarkdown"],
    [data-testid="stSidebar"] [data-testid="stElementContainer"],
    [data-testid="stSidebar"] [data-testid="stMarkdown"] {{ text-align: start; }}
    </style>""",
    unsafe_allow_html=True,
)

LANG = st.session_state.language

def display_class_name(name):
    """Display-only label. The original class identifier is never modified."""
    return class_label(name, arabic=(LANG == "العربية"))

# --------------------------------------------------
# Top bar
# --------------------------------------------------
st.markdown(
    f"""
    <div class="topbar {'arabic-ui' if is_arabic else ''}" dir="{'rtl' if is_arabic else 'ltr'}">
        <div class="tb-left">
            <div class="tb-icon">🌿</div>
            <div>
                <div class="tb-title">{tr("Plant Disease Detection &amp; Care", "تشخيص أمراض النباتات والعناية بها")}</div>
                <div class="tb-sub">{tr("Identify plant diseases, find similar plants, and get care recommendations.", "تعرّف على أمراض النباتات، واعثر على صور مشابهة، واحصل على إرشادات العناية.")}</div>
            </div>
        </div>
        <div class="tb-user"><span class="av">🍃</span> GreenLens</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# PAGE: Home
# ==================================================
if selected_page == NAV_HOME:
    left_col, right_col = st.columns([1.15, 1], gap="large")

    # ---------------- Disease Detection card ----------------
    with left_col:
        with st.container(border=True):

            card_header(
                "🔬",
                tr("Disease Detection", "تشخيص أمراض النباتات"),
                tr(
                    "Upload a plant image to detect possible diseases and get instant results.",
                    "ارفع صورة للنبات للتعرّف على الأمراض المحتملة والحصول على النتائج فورًا."
                ),
            )

            preview_col, upload_col = st.columns(2, gap="medium")
            with upload_col:
                uploaded_file = st.file_uploader(
                    tr("Choose a clear image of a plant leaf", "اختر صورة واضحة لورقة النبات"),
                    type=["jpg", "jpeg", "png", "webp"],
                    help=tr(
                         "Upload a leaf image to analyze it and receive care guidance.",
                         "ارفع صورة لورقة النبات لتحليلها والحصول على إرشادات العناية."
                     ),
                    key="home_image_upload",
                    label_visibility="collapsed",
                )
                st.caption(tr(
                    "Supports: JPG, JPEG, PNG, WEBP",
                    "الصيغ المدعومة: JPG، JPEG، PNG، WEBP"
                ))

            image = None
            if uploaded_file is not None:
                sig = (uploaded_file.name, uploaded_file.size)
                if st.session_state.home_sig != sig:  # صورة جديدة -> امسح النتايج القديمة
                    st.session_state.home_sig = sig
                    st.session_state.prediction_result = None
                    st.session_state.home_similar = None
                    st.session_state.show_all_similar = False
                image = open_image(uploaded_file)
            else:
                st.session_state.home_sig = None
                st.session_state.prediction_result = None
                st.session_state.home_similar = None

            with preview_col:
                if image is not None:
                    st.image(
                        image,
                        caption=tr("Uploaded image", "الصورة المرفوعة"),
                        use_container_width=True,
                    )
                else:
                    
                    st.markdown(
                        f'<div class="empty-box">🌿<br>'
                        f'{tr("No image yet", "لم يتم اختيار صورة بعد")}<br>'
                        f'{tr("Upload a leaf image to start", "ارفع صورة ورقة نبات للبدء")}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )


            analyze = st.button(
                tr("🔍  Analyze Plant  ›", "🔍  حلّل النبات  ‹"),
                key="detect_disease_button",
                type="primary",
                use_container_width=True,
                disabled=image is None,
            )

            if analyze and image is not None:
                try:
                    with st.spinner(tr("Analyzing the plant image...", "جارٍ تحليل صورة النبات...")):
                        st.session_state.prediction_result = predict_plant(image)
                except Exception as error:
                    st.session_state.prediction_result = None
                    st.error(tr("Disease detection failed", "فشل تشخيص النبات") + f": {error}")
                # البحث عن صور مشابهة (نفس الدالة الأصلية)
                try:
                    with st.spinner(tr("Searching for similar plants...", "جارٍ البحث عن نباتات مشابهة...")):
                        st.session_state.home_similar = search_similar_images(
                            image, top_k=int(st.session_state.top_k)
                        )
                except Exception as error:
                    st.session_state.home_similar = []
                    st.error(tr("Similarity search failed", "فشل البحث عن الصور المشابهة") + f": {error}")

            prediction = st.session_state.prediction_result
            st.markdown(f'<div class="sec-title">{tr("📋 Prediction Result", "📋 نتيجة التشخيص")}</div>', unsafe_allow_html=True)
            if prediction is None:
                st.markdown(
                    f'<div class="empty-box">{tr("Click Analyze Plant to see the predicted class and confidence.", "اضغط على «حلّل النبات» لعرض التشخيص ونسبة الثقة.")}</div>',
                    unsafe_allow_html=True,
                )
            else:
                predicted_class = prediction["predicted_class"]
                confidence = max(0.0, min(1.0, float(prediction["confidence"])))
                is_healthy = "healthy" in str(predicted_class).lower()

                plant_name = ""
                try:
                    rec_for_name = generate_smart_recommendations(
                        class_name=predicted_class, confidence=confidence
                    )
                    plant_name = localize_recommendation(rec_for_name, is_arabic).get("plant", "")
                except Exception:
                    pass

                thumb = image_to_data_uri(image) if image is not None else ""
                thumb_html = f'<img src="{thumb}">' if thumb else ""
                badge_html = (
                    f'<span class="badge ok">{tr("Healthy", "سليم")}</span>'
                    if is_healthy
                    else f'<span class="badge warn">{tr("⚠ Possible disease", "⚠ مرض محتمل")}</span>'
                )
                degrees = confidence * 360
                st.markdown(
                    f'<div class="pred-box">'
                    f'<div class="pred-left">{thumb_html}<div>'
                    f'<div class="pred-name">{esc(display_class_name(predicted_class))}</div>'
                    f'<div class="pred-plant">{esc(plant_name)}</div>{badge_html}</div></div>'
                    f'<div class="ring-wrap">'
                    f'<div class="ring" style="background: conic-gradient(#16a34a {degrees:.0f}deg, #e3efe6 0deg);">'
                    f'<div class="ring-in">{confidence:.0%}</div></div>'
                    f'<div class="ring-cap">{tr("Confidence", "نسبة الثقة")}</div></div></div>',
                    unsafe_allow_html=True,
                )

                others = [
                    item for item in prediction.get("top_predictions", [])
                    if item["class_name"] != predicted_class
                ][:3]
                if others:
                    st.markdown(f'<div class="sec-title">{tr("Other Possible Classes", "احتمالات أخرى")}</div>', unsafe_allow_html=True)
                    tiles = "".join(
                        f'<div class="other-item"><div class="n">{esc(display_class_name(item["class_name"]))}</div>'
                        f'<div class="p">{ltr(format(max(0.0, min(1.0, float(item["confidence"]))), ".0%"))}</div></div>'
                        for item in others
                    )
                    st.markdown(f'<div class="other-grid">{tiles}</div>', unsafe_allow_html=True)

    # ---------------- Care Recommendations card ----------------
    with right_col:
        with st.container(border=True):
            card_header("🌱", tr("Care Recommendations", "توصيات العناية"), tr("Based on the predicted class and plant condition.", "بناءً على التشخيص وحالة النبات."))
            prediction = st.session_state.prediction_result
            if prediction is None:
                st.markdown(
                    f'<div class="empty-box">{tr("Care guidance will appear here after you analyze a plant.", "ستظهر إرشادات العناية هنا بعد تحليل النبات.")}</div>',
                    unsafe_allow_html=True,
                )
            else:
                try:
                    # Backend call is unchanged; only the returned text is translated for display.
                    recommendation = localize_recommendation(
                        generate_smart_recommendations(
                            class_name=prediction["predicted_class"],
                            confidence=float(prediction["confidence"]),
                        ),
                        is_arabic,
                    )
                    is_healthy = "healthy" in str(prediction["predicted_class"]).lower()
                    status_text = tr("Healthy", "سليم") if is_healthy else tr("Possible disease", "مرض محتمل")
                    st.markdown(
                        f'<div class="info-row">'
                        f'<div class="info-item"><div class="l">{tr("Plant", "النبات")}</div><div class="v">🌿 {esc(recommendation.get("plant", tr("Unknown", "غير معروف")))}</div></div>'
                        f'<div class="info-item"><div class="l">{tr("Condition", "الحالة")}</div><div class="v">{esc(recommendation.get("condition", tr("Unknown", "غير معروف")))}</div></div>'
                        f'<div class="info-item"><div class="l">{tr("Status", "الوضع")}</div><div class="v">{esc(status_text)}</div></div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    if recommendation.get("status"):
                        st.info(recommendation["status"])

                    st.markdown(f'<div class="care-title">{tr("What you can do", "ما الذي يمكنك فعله")}</div>', unsafe_allow_html=True)
                    advice = recommendation.get("recommendations", [])
                    if advice:
                        items_html = "".join(
                            f'<div class="care-item"><div class="ck">✓</div><div>{esc(item)}</div></div>'
                            for item in advice
                        )
                        st.markdown(items_html, unsafe_allow_html=True)
                    else:
                        st.caption(tr("No recommendations are available for this result.", "لا توجد توصيات متاحة لهذه النتيجة."))

                    disclaimer = recommendation.get(
                        "disclaimer",
                        tr(
                            "AI suggestions are not a confirmed diagnosis. Model scores are not necessarily calibrated probabilities.",
                            "اقتراحات الذكاء الاصطناعي ليست تشخيصًا مؤكدًا، ودرجات النموذج ليست بالضرورة احتمالات دقيقة.",
                        ),
                    )
                    st.markdown(f'<div class="note-box">💡 {esc(disclaimer)}</div>', unsafe_allow_html=True)
                except Exception as error:
                    st.error(tr("Could not generate care recommendations", "تعذّر إنشاء توصيات العناية") + f": {error}")

    # ---------------- Similar Plants card ----------------
    with st.container(border=True):
        home_similar = st.session_state.home_similar
        PREVIEW_COUNT = 6
        head_col, btn_col = st.columns([6, 1])
        with head_col:
            card_header("🖼️", tr("Similar Plants", "نباتات مشابهة"), tr("Find visually similar plants from our dataset.", "اعثر على صور نباتات مشابهة من مجموعة البيانات."))
        with btn_col:
            if home_similar and len(home_similar) > PREVIEW_COUNT:
                label = tr("Show Less", "عرض أقل") if st.session_state.show_all_similar else tr("View All  ›", "عرض الكل  ‹")
                st.button(label, key="view_all_button", on_click=toggle_view_all, use_container_width=True)

        if home_similar is None:
            st.markdown(
                f'<div class="empty-box">{tr("Similar images will appear here after you analyze a plant.", "ستظهر الصور المشابهة هنا بعد تحليل النبات.")}</div>',
                unsafe_allow_html=True,
            )
        elif home_similar:
            visible = home_similar if st.session_state.show_all_similar else home_similar[:PREVIEW_COUNT]
            render_similar_grid(visible, per_row=PREVIEW_COUNT)
        else:
            st.info(tr("No similar images were found. Check that the dataset is available and indexed.", "لم يتم العثور على صور مشابهة. تأكد من توفر مجموعة البيانات وفهرستها."))


# ==================================================
# PAGE: About
# ==================================================
elif selected_page == NAV_ABOUT:
    with st.container(border=True):
        card_header("ℹ️", tr("About GreenLens", "عن GreenLens"), tr("AI-assisted plant health workspace.", "منصة للمساعدة بالذكاء الاصطناعي في تقييم صحة النباتات."))
        st.write(tr(
            "GreenLens detects the most likely plant disease from a leaf image, gives practical care guidance, and finds visually similar images from the local dataset.",
            "يساعد GreenLens في تحديد المرض المحتمل للنبات من صورة الورقة، ويقدم إرشادات عملية للعناية بالنبات، ويعثر على صور مشابهة من مجموعة الصور المحلية."
        ))
        st.markdown(tr(
            "- 🔬 **Disease detection** – estimates the most likely condition from the supported classes.\n"
            "- 🌱 **Care recommendations** – guidance based on the model prediction.\n"
            "- 🖼️ **Similar plants** – visual search over the local dataset.",
            "- 🔬 **تشخيص الأمراض** – يقدّر الحالة الأكثر احتمالًا من بين الفئات التي يدعمها النموذج.\n"
            "- 🌱 **توصيات العناية** – إرشادات مبنية على نتيجة النموذج.\n"
            "- 🖼️ **النباتات المشابهة** – البحث بصريًا في مجموعة الصور المحلية."
        ))
        
        st.markdown(
            f"""
            <div dir="{'rtl' if is_arabic else 'ltr'}" style="
                text-align: {'right' if is_arabic else 'left'};
                background-color: #f0fdf4;
                color: #14532d;
                border: 1px solid #86c99a;
                border-radius: 10px;
                padding: 12px 16px;
                font-weight: 700;
                font-size: 1rem;
            ">
                {tr("⚠️ AI-assisted results should be verified before making treatment decisions.", "⚠️ يجب التحقق من النتائج المدعومة بالذكاء الاصطناعي قبل اتخاذ قرارات العلاج.")}
            </div>
            """,
            unsafe_allow_html=True,
        )



# ==================================================
# PAGE: Settings
# ==================================================
else:
    with st.container(border=True):
        card_header("⚙️", tr("Settings", "الإعدادات"), tr("Adjust how GreenLens behaves.", "عدّل طريقة عمل GreenLens."))
        current_k = clamp_top_k(st.session_state.top_k)

        st.markdown(
            f'<div class="stepper-label">{tr("Number of similar images to retrieve", "عدد الصور المتشابهة المطلوب عرضها")}</div>',
            unsafe_allow_html=True,
        )
        with st.container(key="top_k_stepper"):
            minus_col, value_col, plus_col = st.columns([1, 1.3, 1], gap="small")
            with minus_col:
                st.button(
                    "−", key="top_k_minus", on_click=change_top_k, args=(-1,),
                    disabled=current_k <= TOP_K_MIN, use_container_width=True,
                )
            with value_col:
                st.markdown(f'<div class="stepper-value">{current_k}</div>', unsafe_allow_html=True)
            with plus_col:
                st.button(
                    "+", key="top_k_plus", on_click=change_top_k, args=(1,),
                    disabled=current_k >= TOP_K_MAX, use_container_width=True,
                )

        st.caption(tr(
            f"Choose between {TOP_K_MIN} and {TOP_K_MAX} images. Applies to the next similarity search.",
            f"اختر من {TOP_K_MIN} إلى {TOP_K_MAX} صورة. سيُطبّق هذا الإعداد عند البحث التالي عن الصور المتشابهة.",
        ))


# --------------------------------------------------
# Footer
# --------------------------------------------------
st.caption(tr(
    "GreenLens | AI-assisted plant analysis. Predictions should be verified before making treatment decisions.",
    "GreenLens | تحليل النباتات بمساعدة الذكاء الاصطناعي. يجب التحقق من النتائج قبل اتخاذ قرارات العلاج."
))