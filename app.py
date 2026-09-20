
from __future__ import annotations

import json
import time
import hashlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# Optional GIS packages
try:
    import rasterio
    from rasterio.io import MemoryFile
except Exception:
    rasterio = None
    MemoryFile = None

try:
    import folium
    from streamlit_folium import st_folium
except Exception:
    folium = None
    st_folium = None

try:
    import pycountry
except Exception:
    pycountry = None


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_NAME = "HuggingFaceTB/SmolVLM-256M-Instruct"

DATASET_DIR = BASE_DIR / "datasets" / "rsvqa"
DATASET_FILE = DATASET_DIR / "training_data.json"
DATASET_IMAGE_DIR = DATASET_DIR / "images" / "Images_LR"

REGION_COLORS = {
    "Water candidate": (35, 125, 245),
    "Vegetation candidate": (45, 180, 70),
    "Construction candidate": (240, 75, 45),
    "Road/line candidate": (245, 205, 35),
}

st.set_page_config(
    page_title="SatQuery AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "page": "Dashboard",
    "history": [],
    "analysis_result": None,
    "overlay_result": None,
    "deep_scan_result": None,
    "analysis_file_signature": None,
    "map_result": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# UI
# ============================================================

st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(circle at 15% 10%, rgba(0,110,255,.18), transparent 28%),
        radial-gradient(circle at 85% 20%, rgba(0,210,255,.10), transparent 25%),
        linear-gradient(135deg,#020611 0%,#06101e 50%,#02050d 100%);
    color:#eaf6ff;
}
[data-testid="stSidebar"] {
    background:rgba(3,10,20,.96);
    border-right:1px solid rgba(100,190,255,.16);
}
.block-container {
    max-width:1400px;
    padding-top:2rem;
}
.hero {
    padding:28px;
    border:1px solid rgba(100,200,255,.20);
    border-radius:24px;
    background:linear-gradient(135deg,rgba(7,25,48,.92),rgba(4,12,24,.78));
    box-shadow:0 20px 70px rgba(0,0,0,.35);
    margin-bottom:22px;
}
.hero h1 {
    font-size:42px;
    margin:0;
    color:#eaf8ff;
}
.hero p {
    color:#9fc4dc;
    font-size:17px;
}
.card {
    padding:20px;
    border:1px solid rgba(100,190,255,.16);
    border-radius:18px;
    background:rgba(8,20,35,.72);
    min-height:120px;
}
.metric {
    font-size:30px;
    font-weight:700;
    color:#7fdcff;
}
.muted {
    color:#8da9bd;
}
.answer {
    padding:20px;
    border-radius:16px;
    border:1px solid rgba(75,205,255,.25);
    background:rgba(4,24,40,.75);
    font-size:20px;
}
.small-note {
    color:#8da9bd;
    font-size:13px;
}
</style>
""", unsafe_allow_html=True)


def hero(title, subtitle):
    st.markdown(
        f"""
        <div class="hero">
            <h1>🛰️ {title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def card(title, value, note):
    st.markdown(
        f"""
        <div class="card">
            <div class="muted">{title}</div>
            <div class="metric">{value}</div>
            <div class="small-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# SMOLVLM
# ============================================================

@st.cache_resource
def load_smolvlm():
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return processor, model, device


def ask_smolvlm(image: Image.Image, question: str):
    processor, model, device = load_smolvlm()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=prompt,
        images=[image],
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }

    start = time.perf_counter()

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=False,
        )

    elapsed = time.perf_counter() - start

    answer = processor.batch_decode(
        output,
        skip_special_tokens=True,
    )[0].strip()

    if "Assistant:" in answer:
        answer = answer.split("Assistant:", 1)[-1].strip()

    return answer, elapsed, device


# ============================================================
# IMAGE HELPERS
# ============================================================

def image_signature(uploaded_file):
    data = uploaded_file.getvalue()
    return hashlib.sha1(data).hexdigest()


def candidate_mask(rgb: np.ndarray, region: str):
    rgb_f = rgb.astype(np.float32)

    r = rgb_f[:, :, 0]
    g = rgb_f[:, :, 1]
    b = rgb_f[:, :, 2]

    total = r + g + b + 1.0
    green_index = (2.0 * g - r - b) / total
    blue_index = (b - r) / total
    red_index = (r - b) / total
    brightness = total / (255.0 * 3.0)

    if region == "Water candidate":
        mask = (
            (blue_index > 0.025)
            & (blue_index > green_index * 0.70)
            & (brightness > 0.12)
        )
    elif region == "Vegetation candidate":
        mask = (
            (green_index > 0.035)
            & (g >= r * 0.98)
            & (g >= b * 0.96)
            & (brightness > 0.10)
        )
    elif region == "Construction candidate":
        mask = (
            (brightness > 0.35)
            & (np.abs(red_index) < 0.16)
            & (np.abs(blue_index) < 0.16)
            & (green_index < 0.08)
        )
    else:
        mask = np.zeros(rgb.shape[:2], dtype=bool)

    return (mask.astype(np.uint8) * 255)


def add_region_candidates(image: Image.Image, max_per_class=5):
    rgb = np.array(image.convert("RGB"))
    output = rgb.copy()

    h, w = rgb.shape[:2]
    image_area = h * w
    rows = []

    for region in [
        "Water candidate",
        "Vegetation candidate",
        "Construction candidate",
    ]:
        mask = candidate_mask(rgb, region)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        contours = sorted(
            contours,
            key=cv2.contourArea,
            reverse=True,
        )

        count = 0

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < image_area * 0.0006:
                continue

            x, y, bw, bh = cv2.boundingRect(contour)

            if bw < 8 or bh < 8:
                continue

            color = REGION_COLORS[region]

            overlay = output.copy()
            cv2.rectangle(
                overlay,
                (x, y),
                (x + bw, y + bh),
                color,
                -1,
            )
            output = cv2.addWeighted(
                overlay,
                0.16,
                output,
                0.84,
                0,
            )

            cv2.rectangle(
                output,
                (x, y),
                (x + bw, y + bh),
                color,
                3,
            )

            label = region.replace(" candidate", "")

            cv2.putText(
                output,
                label,
                (x, max(18, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

            rows.append({
                "Type": "Region candidate",
                "Class": region,
                "Confidence": "heuristic",
                "X": x,
                "Y": y,
                "Width": bw,
                "Height": bh,
                "Area %": round(area / image_area * 100, 3),
            })

            count += 1

            if count >= max_per_class:
                break

    return Image.fromarray(output), rows


def add_line_candidates(image: Image.Image):
    rgb = np.array(image.convert("RGB"))
    output = rgb.copy()

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 160)

    minimum = max(30, min(rgb.shape[:2]) // 12)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=55,
        minLineLength=minimum,
        maxLineGap=12,
    )

    rows = []

    if lines is None:
        return Image.fromarray(output), rows

    color = REGION_COLORS["Road/line candidate"]

    for index, line in enumerate(lines[:20]):
        x1, y1, x2, y2 = map(int, line.reshape(-1)[:4])

        length = float(
            ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        )

        if length < 30:
            continue

        cv2.line(
            output,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        rows.append({
            "Type": "Line candidate",
            "Class": "Road/line candidate",
            "Confidence": "heuristic",
            "X1": x1,
            "Y1": y1,
            "X2": x2,
            "Y2": y2,
            "Length": round(length, 1),
        })

    return Image.fromarray(output), rows


# ============================================================
# DEEP SCAN
# ============================================================

def deep_scan_image(image, question, rows=2, cols=2):
    width, height = image.size
    results = []

    tile_w = width // cols
    tile_h = height // rows

    for r in range(rows):
        for c in range(cols):
            left = c * tile_w
            top = r * tile_h
            right = width if c == cols - 1 else (c + 1) * tile_w
            bottom = height if r == rows - 1 else (r + 1) * tile_h

            tile = image.crop((left, top, right, bottom))

            answer, elapsed, device = ask_smolvlm(
                tile,
                question,
            )

            results.append({
                "Region": f"R{r + 1}C{c + 1}",
                "Answer": answer,
                "Time (s)": round(elapsed, 2),
                "Device": device,
            })

    return results


# ============================================================
# GEOTIFF METADATA
# ============================================================

def get_geotiff_metadata(uploaded_file):
    if rasterio is None or MemoryFile is None:
        return {"Available": False}

    try:
        data = uploaded_file.getvalue()

        with MemoryFile(data) as memfile:
            with memfile.open() as src:
                return {
                    "Available": True,
                    "Width": src.width,
                    "Height": src.height,
                    "Bands": src.count,
                    "CRS": str(src.crs),
                    "Resolution": str(src.res),
                    "Bounds": str(src.bounds),
                    "Driver": src.driver,
                }

    except Exception as exc:
        return {
            "Available": False,
            "Error": str(exc),
        }


# ============================================================
# MAP
# ============================================================

COUNTRY_COORDS = {
    "India": (20.5937, 78.9629),
    "United States": (37.0902, -95.7129),
    "China": (35.8617, 104.1954),
    "Japan": (36.2048, 138.2529),
    "Brazil": (-14.2350, -51.9253),
    "Australia": (-25.2744, 133.7751),
    "United Kingdom": (55.3781, -3.4360),
    "France": (46.2276, 2.2137),
    "Germany": (51.1657, 10.4515),
}


def render_map(country):
    if folium is None or st_folium is None:
        st.warning(
            "Map packages are not installed. "
            "Run: pip install folium streamlit-folium"
        )
        return

    lat, lon = COUNTRY_COORDS.get(
        country,
        (20.5937, 78.9629),
    )

    fmap = folium.Map(
        location=[lat, lon],
        zoom_start=4,
        control_scale=True,
    )

    folium.Marker(
        [lat, lon],
        tooltip=country,
        popup=country,
    ).add_to(fmap)

    st_folium(
        fmap,
        height=430,
        use_container_width=True,
    )


# ============================================================
# MAP & GIS HELPERS
# ============================================================

def safe_text(value):
    return str(value).replace("<", "&lt;").replace(">", "&gt;")


def ask_map_context(image):
    questions = [
        "What country or geographic region does this map or satellite image appear to show? Answer only with the most likely country or region.",
        "Is this image more likely a satellite image, a map, an aerial image, or a street-level image? Answer with one type.",
        "What visible geographic features are present, such as rivers, roads, cities, farmland, coastlines, or mountains? Give a short list.",
    ]
    results = []
    total = 0.0
    device = "cpu"
    for q in questions:
        answer, elapsed, device = ask_smolvlm(image, q)
        results.append(answer)
        total += elapsed
    return {
        "region": results[0],
        "type": results[1],
        "features": results[2],
        "elapsed": total,
        "device": device,
    }


def country_coordinates(name):
    if not name:
        return None
    name_l = name.lower().strip()
    fallback = {
        "india": (20.59, 78.96), "united states": (39.83, -98.58),
        "usa": (39.83, -98.58), "china": (35.86, 104.20),
        "japan": (36.20, 138.25), "brazil": (-14.24, -51.93),
        "australia": (-25.27, 133.78), "united kingdom": (55.38, -3.44),
        "france": (46.23, 2.21), "germany": (51.17, 10.45),
        "canada": (56.13, -106.35), "nepal": (28.39, 84.12),
        "bangladesh": (23.68, 90.36), "pakistan": (30.38, 69.35),
        "sri lanka": (7.87, 80.77), "myanmar": (21.92, 95.96),
    }
    for key, coords in fallback.items():
        if key in name_l:
            return coords
    return None


def render_country_map(label):
    if folium is None or st_folium is None:
        st.info("Folium is not installed, so the interactive map is unavailable.")
        return
    coords = country_coordinates(label)
    if coords is None:
        st.info("A reliable country coordinate could not be extracted from the model response.")
        return
    fmap = folium.Map(location=list(coords), zoom_start=4, control_scale=True)
    folium.Marker(list(coords), tooltip=label, popup=label).add_to(fmap)
    st_folium(fmap, height=430, use_container_width=True)


def format_bytes(size):
    units = ["B", "KB", "MB", "GB"]
    size = float(size)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("# 🛰️ SatQuery AI")
st.sidebar.caption("Interactive Vision-Language Assistant")
st.sidebar.markdown("---")

NAV_ITEMS = {
    "🏠 Overview": "Dashboard",
    "🧠 AI Image Analysis": "Image Analysis",
    "🎯 Region Detection": "Region Detection",
    "🔎 Deep Scan": "Deep Scan",
    "🗺️ Map & GIS": "Map & GIS",
    "📐 GeoTIFF Metadata": "GeoTIFF Metadata",
    "📚 RSVQA Explorer": "RSVQA Explorer",
    "🕘 Query History": "Query History",
    "⚙️ System & Model": "System & Model",
}

current_label = next((k for k, v in NAV_ITEMS.items() if v == st.session_state.page), "🏠 Overview")
selected_label = st.sidebar.radio(
    "Navigation",
    list(NAV_ITEMS.keys()),
    index=list(NAV_ITEMS.keys()).index(current_label),
)
page = NAV_ITEMS[selected_label]
st.session_state.page = page

st.sidebar.markdown("---")
st.sidebar.markdown("### Project Stack")
st.sidebar.caption("🐍 Python + Streamlit")
st.sidebar.caption("🧠 SmolVLM-256M-Instruct")
st.sidebar.caption("🛰️ RSVQA-LR dataset")
st.sidebar.caption("🗺️ Rasterio + Folium")
st.sidebar.caption("☁️ Supabase-ready architecture")

st.sidebar.markdown("### Runtime")
device_label = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
st.sidebar.success(f"Compute: {device_label}")
st.sidebar.caption("CPU inference can be slower for vision-language analysis.")


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    hero(
        "SatQuery AI",
        "An interactive vision-language assistant for multimodal remote-sensing image analysis through natural-language queries."
    )

    st.markdown("### 🛰️ Mission Control")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card("Vision-Language Model", "SmolVLM", "Image + text understanding")
    with c2:
        sample_count = 0
        if DATASET_FILE.exists():
            try:
                with open(DATASET_FILE, "r", encoding="utf-8") as f:
                    sample_count = len(json.load(f))
            except Exception:
                sample_count = 0
        card("RSVQA Samples", f"{sample_count:,}" if sample_count else "Ready", "Remote-sensing VQA examples")
    with c3:
        card("Analysis Modules", "7", "VQA, regions, scan, GIS and metadata")
    with c4:
        card("Runtime", "GPU" if torch.cuda.is_available() else "CPU", "Current inference device")

    st.markdown("### 🔬 What this prototype can do")
    features = [
        ("01", "Natural-language VQA", "Upload a satellite or aerial image and ask questions such as water presence, urban/rural context, buildings and vegetation."),
        ("02", "Candidate region analysis", "Generate visual candidate boxes for water, vegetation and construction using image-processing heuristics."),
        ("03", "Line / road candidates", "Use edge and line analysis to highlight possible road or linear structures for inspection."),
        ("04", "Deep regional scan", "Split an image into tiles and ask the same question on each region to compare local evidence."),
        ("05", "GeoTIFF inspection", "Read raster dimensions, bands, CRS, resolution, bounds and other available spatial metadata."),
        ("06", "Map & GIS context", "Use vision-language interpretation to estimate visible geographic context and display an interactive map when a country is identifiable."),
        ("07", "RSVQA evaluation", "Compare SmolVLM responses with the prepared RSVQA question-answer dataset."),
    ]
    for i in range(0, len(features), 2):
        cols = st.columns(2)
        for col, item in zip(cols, features[i:i+2]):
            with col:
                st.markdown(
                    f"""<div class='card'><div class='muted'>{item[0]}</div><h3>{item[1]}</h3><p class='muted'>{item[2]}</p></div>""",
                    unsafe_allow_html=True,
                )

    st.markdown("### 🔄 End-to-end workflow")
    st.info("Upload image → Ask a question → SmolVLM inference → Store result in session history → Optional region/deep-scan/GIS inspection → Compare with RSVQA when required.")

    st.markdown("### ⚠️ Important technical note")
    st.warning("Region Detection currently uses visual image-processing heuristics. It should be presented as candidate-region analysis, not as a trained semantic-segmentation model. SmolVLM inference is also dependent on available CPU/GPU resources.")


# ============================================================
# IMAGE ANALYSIS
# ============================================================

elif page == "Image Analysis":

    hero(
        "Image Analysis",
        "Upload a satellite image and ask SatQuery AI a question."
    )

    uploaded = st.file_uploader(
        "Upload satellite / remote-sensing image",
        type=["jpg", "jpeg", "png", "webp", "tif", "tiff"],
        key="analysis_upload",
    )

    if uploaded is None:
        st.info("Upload an image to start analysis.")

    else:
        signature = image_signature(uploaded)

        if signature != st.session_state.analysis_file_signature:
            st.session_state.analysis_file_signature = signature
            st.session_state.analysis_result = None
            st.session_state.overlay_result = None
            st.session_state.deep_scan_result = None

        image = Image.open(uploaded).convert("RGB")

        st.image(
            image,
            caption=uploaded.name,
            use_container_width=True,
        )

        st.markdown("### 💬 Ask your question")

        question = st.text_input(
            "Question",
            value="Is there water in the image?",
            placeholder="Example: Is there water in the image?",
        )

        quick = st.columns(4)

        questions = [
            "Is there water in the image?",
            "Are there buildings?",
            "Is this an urban area?",
            "Is there vegetation?",
        ]

        for col, q in zip(quick, questions):
            with col:
                if st.button(q, use_container_width=True):
                    question = q

        if st.button(
            "🚀 Analyze Image",
            type="primary",
            use_container_width=True,
        ):

            if len(question.strip()) < 5:
                st.warning("Please enter a meaningful question.")

            else:
                with st.spinner(
                    "SatQuery AI is analysing the image..."
                ):
                    try:
                        answer, elapsed, device = ask_smolvlm(
                            image,
                            question,
                        )

                        st.session_state.analysis_result = {
                            "answer": answer,
                            "elapsed": elapsed,
                            "device": device,
                            "question": question,
                            "image": uploaded.name,
                        }

                        st.session_state.history.append({
                            "Time": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "Image": uploaded.name,
                            "Question": question,
                            "Answer": answer,
                            "Time (s)": round(elapsed, 2),
                            "Device": device,
                        })

                    except Exception as exc:
                        st.error("VQA analysis failed.")
                        st.exception(exc)

        result = st.session_state.analysis_result

        if result:
            st.markdown("### 🤖 AI Answer")

            st.markdown(
                f"""
                <div class="answer">
                    {result["answer"]}
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.caption(
                f'SmolVLM • {result["device"].upper()} • '
                f'{result["elapsed"]:.2f} seconds'
            )


# ============================================================
# REGION DETECTION
# ============================================================

elif page == "Region Detection":

    hero(
        "Visual Region Detection",
        "Generate candidate visual regions and line overlays for inspection."
    )

    uploaded = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png", "webp", "tif", "tiff"],
        key="region_upload",
    )

    if uploaded:

        image = Image.open(uploaded).convert("RGB")

        c1, c2 = st.columns(2)

        with c1:
            detect_regions = st.checkbox(
                "Detect water / vegetation / construction",
                value=True,
            )

        with c2:
            detect_lines = st.checkbox(
                "Detect road / line candidates",
                value=True,
            )

        if st.button(
            "🎯 Generate Colored Boxes & Region Outlines",
            type="primary",
            use_container_width=True,
        ):

            annotated = image.copy()
            rows = []

            with st.spinner("Generating visual overlays..."):

                if detect_regions:
                    annotated, region_rows = add_region_candidates(
                        annotated
                    )
                    rows.extend(region_rows)

                if detect_lines:
                    annotated, line_rows = add_line_candidates(
                        annotated
                    )
                    rows.extend(line_rows)

            st.session_state.overlay_result = {
                "image": annotated,
                "rows": rows,
            }

        result = st.session_state.overlay_result

        if result:
            st.image(
                result["image"],
                caption="Annotated result",
                use_container_width=True,
            )

            st.markdown("""
            **Legend**

            🔵 Water candidate  
            🟢 Vegetation candidate  
            🔴 Construction candidate  
            🟡 Road / line candidate
            """)

            if result["rows"]:
                st.dataframe(
                    pd.DataFrame(result["rows"]),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No candidate regions were found.")

            st.warning(
                "These are visual heuristic candidate regions, "
                "not a trained satellite segmentation model."
            )


# ============================================================
# DEEP SCAN
# ============================================================

elif page == "Deep Scan":

    hero(
        "Deep Scan",
        "Divide the image into multiple regions and ask SmolVLM the same question."
    )

    uploaded = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png", "webp", "tif", "tiff"],
        key="deep_upload",
    )

    if uploaded:

        image = Image.open(uploaded).convert("RGB")

        st.image(
            image,
            caption=uploaded.name,
            use_container_width=True,
        )

        question = st.text_input(
            "Deep scan question",
            value="Is there water in this region?",
        )

        rows = st.selectbox(
            "Rows",
            [2, 3],
            index=0,
        )

        cols = st.selectbox(
            "Columns",
            [2, 3],
            index=0,
        )

        if st.button(
            "🔎 Start Deep Scan",
            type="primary",
            use_container_width=True,
        ):

            with st.spinner(
                "Analysing image regions... This can take time on CPU."
            ):
                try:
                    result = deep_scan_image(
                        image,
                        question,
                        rows,
                        cols,
                    )

                    st.session_state.deep_scan_result = result

                except Exception as exc:
                    st.error("Deep scan failed.")
                    st.exception(exc)

        if st.session_state.deep_scan_result:
            st.dataframe(
                pd.DataFrame(
                    st.session_state.deep_scan_result
                ),
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# MAP & GIS
# ============================================================

elif page == "Map & GIS":

    hero(
        "Map & GIS Context",
        "Use vision-language analysis to inspect geographic context and optionally display a country-level interactive map."
    )

    uploaded = st.file_uploader(
        "Upload satellite / map image",
        type=["jpg", "jpeg", "png", "webp", "tif", "tiff"],
        key="map_upload",
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption=uploaded.name, use_container_width=True)

        st.markdown("### 🧭 Geographic interpretation")
        st.caption("The model's geographic identification is an estimate. Do not treat visual country recognition as precise geolocation.")

        if st.button("🗺️ Analyse Map / Geographic Context", type="primary", use_container_width=True):
            with st.spinner("Analysing geographic context... This may take time on CPU."):
                try:
                    context = ask_map_context(image)
                    st.session_state.map_result = context
                except Exception as exc:
                    st.error("Map analysis failed.")
                    st.exception(exc)

        result = st.session_state.map_result
        if result:
            c1, c2, c3 = st.columns(3)
            with c1:
                card("Likely region", safe_text(result["region"]), "Vision-language estimate")
            with c2:
                card("Image type", safe_text(result["type"]), "Model classification")
            with c3:
                card("Inference", f"{result["elapsed"]:.1f}s", result["device"].upper())

            st.markdown("### 🌍 Visible geographic features")
            st.write(result["features"])

            st.markdown("### 📍 Interactive country-level context")
            render_country_map(result["region"])

            st.info("Map marker placement is country-level context only. The app does not claim exact coordinates from the image.")


# ============================================================
# GEOTIFF METADATA
# ============================================================

elif page == "GeoTIFF Metadata":

    hero(
        "GeoTIFF Metadata",
        "Inspect dimensions, bands, CRS, resolution and spatial bounds."
    )

    uploaded = st.file_uploader(
        "Upload GeoTIFF",
        type=["tif", "tiff"],
        key="geotiff_upload",
    )

    if uploaded:

        metadata = get_geotiff_metadata(uploaded)

        if not metadata.get("Available"):
            st.error(
                metadata.get(
                    "Error",
                    "Rasterio is unavailable."
                )
            )

        else:

            c1, c2, c3 = st.columns(3)

            with c1:
                card(
                    "Dimensions",
                    f'{metadata["Width"]} × {metadata["Height"]}',
                    "Raster size",
                )

            with c2:
                card(
                    "Bands",
                    metadata["Bands"],
                    "Number of raster bands",
                )

            with c3:
                card(
                    "Driver",
                    metadata["Driver"],
                    "Raster format",
                )

            st.markdown("### Spatial information")

            st.json(metadata)


# ============================================================
# RSVQA EXPLORER
# ============================================================

elif page == "RSVQA Explorer":

    hero(
        "RSVQA Explorer",
        "Explore the downloaded RSVQA-LR questions and answers."
    )

    if not DATASET_FILE.exists():
        st.error(
            f"Dataset file not found:\n{DATASET_FILE}"
        )

    else:

        with open(
            DATASET_FILE,
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        st.success(
            f"Loaded {len(data):,} RSVQA samples."
        )

        index = st.number_input(
            "Sample index",
            min_value=0,
            max_value=max(0, len(data) - 1),
            value=0,
            step=1,
        )

        sample = data[index]

        image_path = (
            BASE_DIR / "datasets" / "rsvqa" /
            sample["image"]
        )

        if image_path.exists():

            image = Image.open(image_path).convert("RGB")

            st.image(
                image,
                caption=str(image_path.name),
                width=500,
            )

            st.markdown("### Question")
            st.write(sample["question"])

            st.markdown("### Ground-truth answer")
            st.success(sample["answer"])

            if st.button(
                "🧠 Ask SmolVLM on this sample",
                type="primary",
            ):

                with st.spinner(
                    "Running SmolVLM..."
                ):

                    try:

                        answer, elapsed, device = ask_smolvlm(
                            image,
                            sample["question"],
                        )

                        st.markdown("### SmolVLM answer")

                        st.markdown(
                            f"""
                            <div class="answer">
                                {answer}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        st.caption(
                            f"{device.upper()} • "
                            f"{elapsed:.2f} seconds"
                        )

                        exact = (
                            answer.strip().lower()
                            == sample["answer"].strip().lower()
                        )

                        if exact:
                            st.success(
                                "Exact match with ground truth."
                            )
                        else:
                            st.info(
                                "Answer differs from the ground-truth text."
                            )

                    except Exception as exc:
                        st.error("SmolVLM failed.")
                        st.exception(exc)

        else:
            st.error(
                f"Image not found: {image_path}"
            )


# ============================================================
# QUERY HISTORY
# ============================================================

elif page == "Query History":

    hero(
        "Query History",
        "Review questions analysed during this session."
    )

    if not st.session_state.history:
        st.info("No queries yet.")

    else:

        df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )

        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()


# ============================================================
# SYSTEM & MODEL
# ============================================================

elif page == "System & Model":

    hero(
        "System & Model",
        "Technical information about the local SatQuery AI prototype and its runtime environment."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        card("Model", "SmolVLM-256M-Instruct", "Hugging Face vision-language model")
    with c2:
        card("Framework", "Streamlit", "Python-based interactive web UI")
    with c3:
        card("Compute", "CUDA" if torch.cuda.is_available() else "CPU", "Detected by PyTorch")

    st.markdown("### 🧩 Architecture")
    st.code("""User
  ↓
Streamlit UI (app.py)
  ↓
Image + Natural-language question
  ↓
SmolVLM-256M-Instruct
  ↓
Answer / visual analysis
  ├── Region candidate heuristics
  ├── Deep regional scan
  ├── GeoTIFF metadata
  ├── Map & GIS context
  └── RSVQA comparison
""", language="text")

    st.markdown("### 📦 Local project paths")
    st.write({
        "Application": str(BASE_DIR / "app.py"),
        "RSVQA training data": str(DATASET_FILE),
        "RSVQA image directory": str(DATASET_IMAGE_DIR),
    })

    st.markdown("### 💾 Session status")
    st.write({
        "Queries in current session": len(st.session_state.history),
        "Image analysis result available": bool(st.session_state.analysis_result),
        "Region overlay available": bool(st.session_state.overlay_result),
        "Deep scan available": bool(st.session_state.deep_scan_result),
        "Map context available": bool(st.session_state.map_result),
    })

    st.warning("This is a local prototype. Query history currently lives in Streamlit session state; it is not yet persisted to Supabase.")
