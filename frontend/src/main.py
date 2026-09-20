from __future__ import annotations

import json
import time
import html
import hashlib
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import torch
from PIL import Image
from transformers import BlipForQuestionAnswering, BlipProcessor

# Optional packages. Install them for full map/country functionality.
try:
    import pycountry
except ImportError:
    pycountry = None

try:
    from countryinfo import CountryInfo
except ImportError:
    CountryInfo = None

try:
    import folium
    from streamlit_folium import st_folium
except ImportError:
    folium = None
    st_folium = None

try:
    import rasterio
    from rasterio.io import MemoryFile
except ImportError:
    rasterio = None
    MemoryFile = None

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
VQA_MODEL_NAME = "Salesforce/blip-vqa-base"
BASE_DIR = Path(__file__).resolve().parent
RSVQA_DIR = BASE_DIR / "data" / "RSVQA"
RSVQA_IMAGE_DIR = BASE_DIR / "RSVQA_Images" / "Images_LR"

st.set_page_config(
    page_title="SatQuery AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "map_info" not in st.session_state:
    st.session_state.map_info = None
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "overlay_result" not in st.session_state:
    st.session_state.overlay_result = None
if "zoom_result" not in st.session_state:
    st.session_state.zoom_result = None
if "deep_scan_result" not in st.session_state:
    st.session_state.deep_scan_result = None
if "analysis_file_name" not in st.session_state:
    st.session_state.analysis_file_name = None
if "analysis_file_signature" not in st.session_state:
    st.session_state.analysis_file_signature = None
if "analysis_question" not in st.session_state:
    st.session_state.analysis_question = "Is there water in the image?"
if "map_file_name" not in st.session_state:
    st.session_state.map_file_name = None
if "ui_model_version" not in st.session_state:
    st.session_state.ui_model_version = "optical_no_generic_yolo_v1"
    st.session_state.overlay_result = None
    st.session_state.analysis_result = None

# ------------------------------------------------------------
# COLORS
# RGB values for annotation output.
# ------------------------------------------------------------
REGION_COLORS = {
    "Water candidate": (35, 125, 245),       # blue
    "Vegetation candidate": (45, 180, 70),   # green
    "Construction candidate": (240, 75, 45), # red/orange
    "Road/line candidate": (245, 205, 35),  # yellow
}

# Fallback data for common countries when countryinfo is absent.
NEIGHBOR_FALLBACK = {
    "India": ["Pakistan", "China", "Nepal", "Bhutan", "Bangladesh", "Myanmar"],
    "China": ["India", "Pakistan", "Afghanistan", "Tajikistan", "Kyrgyzstan", "Kazakhstan", "Russia", "Mongolia", "North Korea", "Nepal", "Bhutan", "Myanmar", "Laos", "Vietnam"],
    "Pakistan": ["India", "China", "Afghanistan", "Iran"],
    "Nepal": ["China", "India"],
    "Bhutan": ["China", "India"],
    "Bangladesh": ["India", "Myanmar"],
    "Myanmar": ["India", "Bangladesh", "China", "Laos", "Thailand"],
    "Japan": ["Russia", "North Korea", "South Korea"],
    "France": ["Belgium", "Luxembourg", "Germany", "Switzerland", "Italy", "Spain", "Andorra", "Monaco"],
    "Germany": ["Denmark", "Poland", "Czechia", "Austria", "Switzerland", "France", "Luxembourg", "Belgium", "Netherlands"],
    "Brazil": ["Argentina", "Bolivia", "Colombia", "Guyana", "Paraguay", "Peru", "Suriname", "Uruguay", "Venezuela", "French Guiana"],
    "United States": ["Canada", "Mexico"],
    "Canada": ["United States"],
    "Mexico": ["United States", "Guatemala", "Belize"],
    "United Kingdom": ["Ireland"],
    "Spain": ["Portugal", "France", "Andorra"],
    "Italy": ["France", "Switzerland", "Austria", "Slovenia", "Vatican City", "San Marino"],
    "Russia": ["Norway", "Finland", "Estonia", "Latvia", "Lithuania", "Poland", "Belarus", "Ukraine", "Georgia", "Azerbaijan", "Kazakhstan", "China", "Mongolia", "North Korea"],
}

# ------------------------------------------------------------
# HIGH-RESOLUTION IMAGE VIEWER
# ------------------------------------------------------------
def render_zoomable_image(image: Image.Image, title: str = "Image", height: int = 560, key: str = "viewer"):
    """Render a centered full-resolution image with constrained zoom/pan."""
    from io import BytesIO
    import base64

    buf = BytesIO()
    image.convert("RGB").save(buf, format="PNG", optimize=False)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    safe_title = html.escape(title)
    viewer_id = "zoomviewer_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]

    markup = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
* {{ box-sizing:border-box; }}
html, body {{ margin:0; padding:0; background:transparent; overflow:hidden; font-family:Inter,system-ui,sans-serif; }}
.viewer {{
  width:100%; height:{height}px; position:relative; overflow:hidden;
  border:1px solid rgba(95,180,255,.25); border-radius:16px;
  background:radial-gradient(circle at 50% 25%, rgba(12,43,82,.34), transparent 48%), #050a12;
  cursor:grab; user-select:none; touch-action:none;
}}
.viewer.dragging {{ cursor:grabbing; }}
.toolbar {{
  position:absolute; z-index:20; left:12px; top:12px; display:flex; gap:7px;
  padding:7px; border:1px solid rgba(255,255,255,.12); border-radius:12px;
  background:rgba(5,10,18,.92);
}}
button {{
  width:34px; height:32px; border:1px solid rgba(120,200,255,.22); border-radius:9px;
  color:#dff5ff; background:#0b1624; font-size:16px; cursor:pointer;
}}
button:hover {{ background:#102238; border-color:rgba(120,220,255,.55); }}
.zoom-label {{ min-width:58px; display:flex; align-items:center; justify-content:center; color:#9fc7e8; font-size:12px; }}
.title {{
  position:absolute; z-index:20; right:12px; top:12px; max-width:48%;
  padding:7px 10px; border-radius:9px; color:#b9d8ee; font-size:12px;
  background:rgba(5,10,18,.88); border:1px solid rgba(255,255,255,.10);
  overflow:hidden; white-space:nowrap; text-overflow:ellipsis;
}}
.stage {{ position:absolute; inset:0; overflow:hidden; display:flex; align-items:center; justify-content:center; }}
.image {{
  position:absolute; left:50%; top:50%; width:auto; height:auto; max-width:none; max-height:none;
  transform-origin:50% 50%; will-change:transform; user-select:none; -webkit-user-drag:none;
  image-rendering:auto; pointer-events:none;
}}
.hint {{
  position:absolute; z-index:20; bottom:10px; left:50%; transform:translateX(-50%);
  padding:6px 10px; border-radius:999px; color:#8eabc2; font-size:11px;
  background:rgba(5,10,18,.82); border:1px solid rgba(255,255,255,.08);
}}
</style>
</head>
<body>
<div id="{viewer_id}" class="viewer">
  <div class="toolbar">
    <button id="minus" title="Zoom out">−</button>
    <button id="reset" title="Fit / reset">↺</button>
    <button id="plus" title="Zoom in">+</button>
    <div id="zoom" class="zoom-label">100%</div>
  </div>
  <div class="title">{safe_title}</div>
  <div class="stage"><img id="img" class="image" src="data:image/png;base64,{encoded}" draggable="false" alt="{safe_title}"></div>
  <div class="hint">Scroll to zoom • Drag to pan • Double-click to zoom</div>
</div>
<script>
(() => {{
  const viewer = document.getElementById('{viewer_id}');
  const img = document.getElementById('img');
  const zoomText = document.getElementById('zoom');
  const minus = document.getElementById('minus');
  const plus = document.getElementById('plus');
  const reset = document.getElementById('reset');

  let scale = 1;
  let panX = 0;
  let panY = 0;
  let dragging = false;
  let lastX = 0;
  let lastY = 0;

  function fit() {{
    const vw = viewer.clientWidth;
    const vh = viewer.clientHeight;
    const iw = img.naturalWidth;
    const ih = img.naturalHeight;
    if (!iw || !ih || !vw || !vh) return;
    const pad = 28;
    scale = Math.min((vw - pad) / iw, (vh - pad) / ih);
    panX = 0;
    panY = 0;
    render();
  }}

  function clampPan() {{
    const vw = viewer.clientWidth;
    const vh = viewer.clientHeight;
    const iw = img.naturalWidth * scale;
    const ih = img.naturalHeight * scale;
    const maxX = Math.max(0, (iw - vw) / 2);
    const maxY = Math.max(0, (ih - vh) / 2);
    panX = Math.max(-maxX, Math.min(maxX, panX));
    panY = Math.max(-maxY, Math.min(maxY, panY));
  }}

  function render() {{
    clampPan();
    img.style.transform = `translate(calc(-50% + ${{panX}}px), calc(-50% + ${{panY}}px)) scale(${{scale}})`;
    zoomText.textContent = `${{Math.round(scale * 100)}}%`;
  }}

  function setScale(next, cursorX = viewer.clientWidth / 2, cursorY = viewer.clientHeight / 2) {{
    const old = scale;
    const bounded = Math.max(0.2, Math.min(10, next));
    if (Math.abs(bounded - old) < 0.0001) return;
    const ratio = bounded / old;
    const cx = cursorX - viewer.clientWidth / 2;
    const cy = cursorY - viewer.clientHeight / 2;
    panX = cx - (cx - panX) * ratio;
    panY = cy - (cy - panY) * ratio;
    scale = bounded;
    render();
  }}

  img.addEventListener('load', () => requestAnimationFrame(fit));
  window.addEventListener('resize', () => requestAnimationFrame(fit));

  viewer.addEventListener('wheel', (e) => {{
    e.preventDefault();
    const rect = viewer.getBoundingClientRect();
    setScale(scale * (e.deltaY < 0 ? 1.18 : 0.85), e.clientX - rect.left, e.clientY - rect.top);
  }}, {{passive:false}});

  viewer.addEventListener('mousedown', (e) => {{
    if (e.button !== 0) return;
    dragging = true;
    viewer.classList.add('dragging');
    lastX = e.clientX;
    lastY = e.clientY;
  }});

  window.addEventListener('mousemove', (e) => {{
    if (!dragging) return;
    panX += e.clientX - lastX;
    panY += e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;
    render();
  }});

  window.addEventListener('mouseup', () => {{
    dragging = false;
    viewer.classList.remove('dragging');
  }});

  viewer.addEventListener('dblclick', (e) => {{
    const rect = viewer.getBoundingClientRect();
    const factor = scale < 2 ? 2 : 1;
    setScale(factor, e.clientX - rect.left, e.clientY - rect.top);
  }});

  minus.addEventListener('click', () => setScale(scale * 0.8));
  plus.addEventListener('click', () => setScale(scale * 1.25));
  reset.addEventListener('click', fit);

  requestAnimationFrame(fit);
}})();
</script>
</body>
</html>
"""
    components.html(markup, height=height, scrolling=False)

# ------------------------------------------------------------
# MODEL LOADERS
# ------------------------------------------------------------
@st.cache_resource

def load_vqa_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained(VQA_MODEL_NAME)
    model = BlipForQuestionAnswering.from_pretrained(VQA_MODEL_NAME)
    model.to(device)
    model.eval()
    return processor, model, device



# ------------------------------------------------------------
# VQA
# ------------------------------------------------------------
def ask_vqa(image: Image.Image, question: str):
    processor, model, device = load_vqa_model()
    inputs = processor(image, question, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=30,
            num_beams=5,
        )
    elapsed = time.perf_counter() - start
    answer = processor.decode(output[0], skip_special_tokens=True).strip()
    return answer, elapsed, device


def crop_percent_region(image: Image.Image, x_pct: float, y_pct: float, w_pct: float, h_pct: float):
    """Crop a percentage-based region while keeping coordinates valid."""
    width, height = image.size
    x_pct = max(0.0, min(100.0, float(x_pct)))
    y_pct = max(0.0, min(100.0, float(y_pct)))
    w_pct = max(1.0, min(100.0 - x_pct, float(w_pct)))
    h_pct = max(1.0, min(100.0 - y_pct, float(h_pct)))

    left = int(round(width * x_pct / 100.0))
    top = int(round(height * y_pct / 100.0))
    right = int(round(width * (x_pct + w_pct) / 100.0))
    bottom = int(round(height * (y_pct + h_pct) / 100.0))

    right = max(left + 1, min(width, right))
    bottom = max(top + 1, min(height, bottom))
    return image.crop((left, top, right, bottom))


def deep_scan_image(image: Image.Image, question: str, rows: int = 3, cols: int = 3):
    """Ask the same question on multiple overlapping tiles to reveal small features."""
    width, height = image.size
    answers = []
    tile_w = max(1, int(width / cols))
    tile_h = max(1, int(height / rows))
    overlap_x = max(2, int(tile_w * 0.12))
    overlap_y = max(2, int(tile_h * 0.12))

    for r in range(rows):
        for c in range(cols):
            left = max(0, c * tile_w - overlap_x)
            top = max(0, r * tile_h - overlap_y)
            right = min(width, (c + 1) * tile_w + overlap_x)
            bottom = min(height, (r + 1) * tile_h + overlap_y)
            tile = image.crop((left, top, right, bottom))
            answer, elapsed, device = ask_vqa(tile, question)
            answers.append({
                "tile": f"R{r+1}C{c+1}",
                "answer": answer,
                "elapsed": elapsed,
                "device": device,
                "image": tile,
            })

    text_answers = [item["answer"].strip().lower() for item in answers]
    positive_words = ("yes", "water", "river", "lake", "sea", "ocean", "stream", "pond")
    negative_words = ("no", "none", "not", "nothing")
    positives = sum(any(word in text for word in positive_words) and not text.startswith("no") for text in text_answers)
    negatives = sum(any(word in text for word in negative_words) for text in text_answers)

    if positives > 0:
        summary = f"Possible positive evidence found in {positives} of {len(answers)} scanned regions."
    elif negatives == len(answers):
        summary = "No positive evidence was found across the scanned regions."
    else:
        summary = "The scanned regions produced mixed or uncertain answers."

    return answers, summary


def identify_map(image: Image.Image):
    questions = {
        "country": "What country is shown in this map or image?",
        "map_type": "What type of map is this, such as political, physical, satellite, road, or terrain map?",
        "map_name": "What is the name or title of this map?",
        "admin_area": "What state, province, region, or administrative area is shown in this map or image?",
    }
    info = {}
    for key, question in questions.items():
        try:
            info[key], _, _ = ask_vqa(image, question)
        except Exception as exc:
            info[key] = f"Unavailable: {exc}"
    return info

# ------------------------------------------------------------
# COUNTRY / MAP HELPERS
# ------------------------------------------------------------
def get_country_list():
    if pycountry is not None:
        names = sorted({c.name for c in pycountry.countries})
        names += ["United States", "United Kingdom", "Russia", "South Korea", "North Korea"]
        return sorted(set(names))
    return sorted(NEIGHBOR_FALLBACK.keys())


def normalize_country(text: str):
    value = (text or "").strip()
    aliases = {
        "usa": "United States",
        "u.s.a.": "United States",
        "us": "United States",
        "uk": "United Kingdom",
        "russia": "Russian Federation",
        "south korea": "Korea, Republic of",
        "north korea": "Korea, Democratic People's Republic of",
        "vietnam": "Viet Nam",
        "czech republic": "Czechia",
        "iran": "Iran, Islamic Republic of",
    }
    low = value.lower()
    if low in aliases:
        return aliases[low]
    countries = get_country_list()
    for country in countries:
        if country.lower() == low:
            return country
    for country in countries:
        if country.lower() in low and len(country) >= 4:
            return country
    return value


def get_country_alpha2(country: str):
    """Return a two-letter ISO country code for subdivision lookup."""
    if pycountry is None:
        return None

    aliases = {
        "United States": "US",
        "United Kingdom": "GB",
        "Russian Federation": "RU",
        "Korea, Republic of": "KR",
        "Korea, Democratic People's Republic of": "KP",
        "Viet Nam": "VN",
        "Iran, Islamic Republic of": "IR",
        "Czechia": "CZ",
    }
    if country in aliases:
        return aliases[country]

    try:
        return pycountry.countries.lookup(country).alpha_2
    except Exception:
        pass

    low = (country or "").lower().strip()
    for item in pycountry.countries:
        if item.name.lower() == low:
            return item.alpha_2
    return None


@st.cache_data(show_spinner=False)
def get_admin1_list(country: str):
    """Return state/province/region names for the selected country."""
    if pycountry is None:
        return []
    code = get_country_alpha2(country)
    if not code:
        return []
    try:
        values = [item.name for item in pycountry.subdivisions.get(country_code=code)]
        return sorted(set(values))
    except Exception:
        return []


def normalize_admin_area(text: str, country: str):
    """Match a VQA administrative-area answer against the country's ISO subdivisions."""
    import difflib
    import re

    raw = (text or "").strip()
    if not raw or raw.lower().startswith("unavailable:"):
        return None

    subdivisions = get_admin1_list(country)
    if not subdivisions:
        return raw

    def clean(value):
        value = value.lower().strip()
        value = re.sub(r"[^a-z0-9\s]", " ", value)
        value = re.sub(r"\s+", " ", value)
        return value

    target = clean(raw)

    # Strong exact match first.
    for name in subdivisions:
        if target == clean(name):
            return name

    # Handle answers such as "Andhra Pradesh, India" or "California state".
    for name in subdivisions:
        cname = clean(name)
        if cname and (cname in target or target in cname):
            return name

    # Conservative fuzzy matching to avoid turning an unrelated answer into a state.
    scored = [(difflib.SequenceMatcher(None, target, clean(name)).ratio(), name) for name in subdivisions]
    scored.sort(reverse=True)
    if scored and scored[0][0] >= 0.72:
        return scored[0][1]
    return raw


def get_admin_area_label(country: str):
    """Human-friendly label for the administrative subdivision level."""
    labels = {
        "US": "State", "CA": "Province", "IN": "State", "AU": "State / Territory",
        "BR": "State", "MX": "State", "DE": "State", "AT": "State", "CH": "Canton",
        "GB": "Country / Region", "ES": "Autonomous community", "IT": "Region",
    }
    return labels.get(get_country_alpha2(country), "State / Province / Region")


def get_neighbors(country: str):
    # Prefer CountryInfo for broad country coverage.
    if CountryInfo is not None:
        candidates = [country]
        if country == "United States":
            candidates.append("United States of America")
        if country == "Russian Federation":
            candidates.append("Russia")
        if country == "Viet Nam":
            candidates.append("Vietnam")
        for candidate in candidates:
            try:
                values = CountryInfo(candidate).neighbors()
                if values is not None:
                    return sorted(set(values))
            except Exception:
                pass
    return sorted(set(NEIGHBOR_FALLBACK.get(country, [])))


def country_center(country: str):
    if CountryInfo is not None:
        try:
            ll = CountryInfo(country).latlng()
            if ll and len(ll) == 2:
                return float(ll[0]), float(ll[1])
        except Exception:
            pass
    fallback = {
        "India": (20.5937, 78.9629),
        "China": (35.8617, 104.1954),
        "Japan": (36.2048, 138.2529),
        "France": (46.2276, 2.2137),
        "Germany": (51.1657, 10.4515),
        "Brazil": (-14.235, -51.9253),
        "United States": (39.8283, -98.5795),
        "Canada": (56.1304, -106.3468),
        "Australia": (-25.2744, 133.7751),
        "United Kingdom": (55.3781, -3.4360),
    }
    return fallback.get(country, (20.0, 0.0))

# ------------------------------------------------------------
# GEOTIFF METADATA
# ------------------------------------------------------------
def read_geotiff_metadata(file_bytes: bytes):
    if rasterio is None or MemoryFile is None:
        return None
    try:
        with MemoryFile(file_bytes) as mem:
            with mem.open() as src:
                b = src.bounds
                return {
                    "CRS": str(src.crs),
                    "Width": src.width,
                    "Height": src.height,
                    "Bands": src.count,
                    "Resolution X": src.res[0],
                    "Resolution Y": src.res[1],
                    "Left": b.left,
                    "Right": b.right,
                    "Top": b.top,
                    "Bottom": b.bottom,
                    "Center X": (b.left + b.right) / 2,
                    "Center Y": (b.bottom + b.top) / 2,
                }
    except Exception:
        return None

# ------------------------------------------------------------
# REGION CANDIDATES
# These are visual heuristics, not trained satellite segmentation.
# ------------------------------------------------------------
def candidate_mask(rgb: np.ndarray, region: str):
    """Create a tolerant candidate mask for optical remote-sensing images.

    This is intentionally a visual heuristic, not a trained segmentation model.
    The thresholds are adaptive so muted/low-saturation satellite imagery still
    produces useful candidate regions.
    """
    rgb_f = rgb.astype(np.float32)
    r = rgb_f[:, :, 0]
    g = rgb_f[:, :, 1]
    b = rgb_f[:, :, 2]

    total = r + g + b + 1.0
    green_index = (2.0 * g - r - b) / total
    blue_index = (b - r) / total
    red_index = (r - b) / total
    brightness = total / (255.0 * 3.0)

    # Adaptive channels help when the image is darker/lower saturation than
    # ordinary photographs.
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray_norm = gray / 255.0

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
        # Built-up candidate: brighter, relatively neutral pixels with weak
        # vegetation/water response. This is deliberately labelled a candidate.
        mask = (
            (brightness > 0.35)
            & (np.abs(red_index) < 0.16)
            & (np.abs(blue_index) < 0.16)
            & (green_index < 0.08)
        )
    else:
        mask = np.zeros(gray.shape, dtype=bool)

    mask = mask.astype(np.uint8) * 255
    return mask


def _append_box(output, rows, region, x, y, bw, bh, area, image_area):
    color = REGION_COLORS[region]
    # Draw a translucent fill plus a strong border so boxes remain visible.
    overlay = output.copy()
    cv2.rectangle(overlay, (x, y), (x + bw, y + bh), color, -1)
    output = cv2.addWeighted(overlay, 0.12, output, 0.88, 0)
    cv2.rectangle(output, (x, y), (x + bw, y + bh), color, 3)
    label = f"{region} • {area / max(image_area, 1) * 100:.1f}%"
    cv2.rectangle(output, (x, max(0, y - 24)), (min(output.shape[1] - 1, x + 235), y), color, -1)
    cv2.putText(
        output,
        label,
        (x + 6, max(17, y - 7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    rows.append({
        "Type": "Region candidate",
        "Class": region,
        "Confidence": "heuristic",
        "X1": x,
        "Y1": y,
        "X2": x + bw,
        "Y2": y + bh,
        "Area": int(area),
    })
    return output


def add_region_candidates(image: Image.Image, min_area_ratio: float = 0.0006, max_per_class: int = 5):
    """Draw visible candidate boxes for major visual land-cover regions.

    Uses connected components first. If the image is too uniform for clean
    contours, a small tile-based fallback selects the strongest candidate
    patches so the overlay remains useful on low-resolution RSVQA imagery.
    """
    rgb = np.array(image.convert("RGB"))
    output = rgb.copy()
    height, width = output.shape[:2]
    image_area = height * width
    min_area = max(20.0, image_area * min_area_ratio)
    rows = []

    kernel = np.ones((3, 3), np.uint8)

    for region in ["Water candidate", "Vegetation candidate", "Construction candidate"]:
        mask = candidate_mask(rgb, region)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            if bw < 8 or bh < 8:
                continue
            candidates.append((area, x, y, bw, bh))

        # Keep the largest non-overlapping regions for readable boxes.
        candidates.sort(reverse=True)
        kept = []
        for item in candidates:
            area, x, y, bw, bh = item
            box_a = (x, y, x + bw, y + bh)
            overlaps = False
            for _, kx, ky, kbw, kbh in kept:
                box_b = (kx, ky, kx + kbw, ky + kbh)
                ix1, iy1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
                ix2, iy2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                union = bw * bh + kbw * kbh - inter
                if union and inter / union > 0.45:
                    overlaps = True
                    break
            if not overlaps:
                kept.append(item)
            if len(kept) >= max_per_class:
                break

        # Fallback for imagery where color thresholds produce no clean contour.
        if not kept:
            tile = max(24, min(height, width) // 5)
            scored = []
            h_tiles = max(1, height // tile)
            w_tiles = max(1, width // tile)
            raw_mask = candidate_mask(rgb, region).astype(np.float32) / 255.0
            for ty in range(h_tiles):
                for tx in range(w_tiles):
                    y1 = ty * tile
                    x1 = tx * tile
                    y2 = min(height, y1 + tile)
                    x2 = min(width, x1 + tile)
                    score = float(raw_mask[y1:y2, x1:x2].mean())
                    if score > 0.06:
                        scored.append((score, x1, y1, x2 - x1, y2 - y1))
            scored.sort(reverse=True)
            kept = [(score * tile * tile, x, y, bw, bh) for score, x, y, bw, bh in scored[:2]]

        for area, x, y, bw, bh in kept:
            output = _append_box(output, rows, region, int(x), int(y), int(bw), int(bh), float(area), image_area)

    return Image.fromarray(output), rows


# ------------------------------------------------------------
# LINE / ROAD CANDIDATES
# ------------------------------------------------------------
def add_line_candidates(image: Image.Image):
    rgb = np.array(image.convert("RGB"))
    output = rgb.copy()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    minimum = max(30, min(rgb.shape[:2]) // 12)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=55, minLineLength=minimum, maxLineGap=12)
    rows = []

    if lines is None:
        return Image.fromarray(output), rows

    color = REGION_COLORS["Road/line candidate"]
    count = 0
    for line in lines:
        coords = np.asarray(line).reshape(-1)
        if coords.size < 4:
            continue
        x1, y1, x2, y2 = [int(value) for value in coords[:4]]
        length = float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
        if length < 30:
            continue
        cv2.line(output, (x1, y1), (x2, y2), color, 2)
        rows.append({
            "Type": "Line candidate",
            "Class": "Road/line candidate",
            "Confidence": "heuristic",
            "X1": x1,
            "Y1": y1,
            "X2": x2,
            "Y2": y2,
            "Area": round(length, 1),
        })
        count += 1
        if count >= 20:
            break
    return Image.fromarray(output), rows

# ------------------------------------------------------------
# COUNTRY MAP
# ------------------------------------------------------------
def render_country_map(country: str):
    if folium is None or st_folium is None:
        st.warning("Folium support is unavailable. Install folium and streamlit-folium.")
        return
    lat, lon = country_center(country)
    fmap = folium.Map(location=[lat, lon], zoom_start=4, control_scale=True)
    folium.Marker([lat, lon], tooltip=country, popup=country).add_to(fmap)
    st_folium(fmap, height=420, use_container_width=True)

# ------------------------------------------------------------
# HISTORY
# ------------------------------------------------------------
def save_query(image_name, question, answer, elapsed, device):
    st.session_state.history.append({
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Image": image_name,
        "Question": question,
        "Answer": answer,
        "Response Time (s)": round(elapsed, 3),
        "Device": device,
    })

# ============================================================
# POLISHED STREAMLIT UI
# ============================================================


PAGE_ORDER = [
    "Dashboard",
    "Image Analysis",
    "Map Identification",
    "Analytics",
    "Dataset",
    "Model",
]


def set_page(page_name: str):
    current = st.session_state.get("page_nav", "Dashboard")
    if current != page_name:
        st.session_state.previous_page = current
        st.session_state.page_nav = page_name
        st.session_state._page_changed_by_callback = True


def go_back():
    current = st.session_state.get("page_nav", "Dashboard")
    previous = st.session_state.get("previous_page")

    if previous and previous != current:
        st.session_state.page_nav = previous
        st.session_state.previous_page = None
        return

    index = PAGE_ORDER.index(current) if current in PAGE_ORDER else 0
    st.session_state.page_nav = PAGE_ORDER[max(0, index - 1)]


def go_next():
    current = st.session_state.get("page_nav", "Dashboard")
    index = PAGE_ORDER.index(current) if current in PAGE_ORDER else 0
    next_index = min(len(PAGE_ORDER) - 1, index + 1)
    st.session_state.previous_page = current
    st.session_state.page_nav = PAGE_ORDER[next_index]


def set_analysis_question(prompt: str):
    st.session_state.analysis_question = prompt


def is_valid_question(text: str) -> bool:
    """Validate that the input looks like a meaningful image-analysis question."""
    q = " ".join((text or "").strip().split())
    if len(q) < 5:
        return False

    # Reject obvious non-text / repeated-character input.
    if not any(ch.isalpha() for ch in q):
        return False
    if len(set(q.lower().replace(" ", ""))) <= 2:
        return False

    # A SatQuery question should either look like a question or reference
    # something that can reasonably be analysed in a remote-sensing image.
    question_starters = (
        "is ", "are ", "am ", "was ", "were ", "do ", "does ", "did ",
        "can ", "could ", "would ", "will ", "what ", "where ", "which ",
        "who ", "when ", "why ", "how ", "describe ", "identify ",
        "find ", "show ", "count ", "tell me ",
    )
    domain_words = {
        "image", "satellite", "map", "remote", "sensing", "water", "river",
        "lake", "sea", "ocean", "road", "street", "building", "buildings",
        "house", "houses", "urban", "city", "town", "village", "forest",
        "tree", "trees", "vegetation", "field", "farmland", "land", "crop",
        "bridge", "airport", "runway", "vehicle", "vehicles", "ship", "boat",
        "harbor", "port", "construction", "structure", "region", "country",
        "state", "province", "district", "area", "object", "objects",
    }

    low = q.lower()
    looks_like_question = low.startswith(question_starters) or "?" in q
    has_domain_term = any(word in low for word in domain_words)
    return looks_like_question and has_domain_term


def inject_ui():
    st.markdown(
        """
        <style>
        :root {
            --bg: #020611;
            --panel: #07111f;
            --panel-2: #0a1626;
            --line: #17304d;
            --line-strong: #23527d;
            --text: #f4f8ff;
            --muted: #8ea1bb;
            --cyan: #38d9ff;
            --blue: #5d7cff;
            --teal: #35d6bd;
        }

        html, body, [data-testid="stAppViewContainer"], .stApp {
            background: var(--bg) !important;
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            color: var(--text);
        }
        [data-testid="stAppViewContainer"] { position: relative; overflow: hidden; }
        [data-testid="stAppViewContainer"]::before {
            content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 0;
            background:
                radial-gradient(circle at 82% 8%, rgba(64,110,255,.16), transparent 23%),
                radial-gradient(circle at 12% 78%, rgba(0,196,255,.08), transparent 20%),
                linear-gradient(180deg, #040916 0%, #020611 62%, #02040b 100%);
        }
        [data-testid="stAppViewContainer"]::after {
            content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 0; opacity: .18;
            background-image: linear-gradient(rgba(72,117,168,.08) 1px, transparent 1px), linear-gradient(90deg, rgba(72,117,168,.08) 1px, transparent 1px);
            background-size: 72px 72px; mask-image: linear-gradient(to bottom, transparent, black 12%, black 88%, transparent);
        }

        .space-bg { position: fixed; inset: 0; pointer-events: none; z-index: 0; overflow: hidden; }
        /* Distant observatory-style solar system: deliberately subtle and far behind the app. */
        .solar-system {
            position:absolute; right:-2.5%; top:3%; width:520px; height:520px;
            border-radius:50%; opacity:.42; filter:saturate(.82) blur(.15px);
            transform:scale(.72);
            transform-origin:70% 25%;
        }
        .solar-system .sun {
            position:absolute; left:50%; top:50%; width:42px; height:42px;
            margin:-21px; border-radius:50%;
            background:radial-gradient(circle at 35% 35%, #fffbd5 0 13%, #ffd66b 30%, #ff9d37 58%, rgba(255,140,40,.22) 74%, transparent 79%);
            box-shadow:0 0 28px rgba(255,190,85,.62),0 0 95px rgba(255,157,55,.22);
        }
        .orbit {
            position:absolute; left:50%; top:50%; border:1px solid rgba(125,196,255,.11);
            border-radius:50%; transform-origin:center;
        }
        .o1{width:92px;height:92px;margin:-46px;animation:orbitSpin 18s linear infinite}
        .o2{width:136px;height:136px;margin:-68px;animation:orbitSpin 25s linear infinite reverse}
        .o3{width:188px;height:188px;margin:-94px;animation:orbitSpin 33s linear infinite}
        .o4{width:250px;height:250px;margin:-125px;animation:orbitSpin 42s linear infinite reverse}
        .o5{width:318px;height:318px;margin:-159px;animation:orbitSpin 54s linear infinite}
        .o6{width:390px;height:390px;margin:-195px;animation:orbitSpin 68s linear infinite reverse}
        .asteroid-belt{position:absolute;left:50%;top:50%;width:286px;height:286px;margin:-143px;border:1px dashed rgba(227,201,146,.07);border-radius:50%;animation:orbitSpin 90s linear infinite;}
        .planet { position:absolute; left:50%; top:50%; border-radius:50%; transform-origin:-46px center; }
        .p1{width:5px;height:5px;margin-left:46px;margin-top:-2.5px;background:#d6ecff;box-shadow:0 0 7px #8acbff;animation:planet1 18s linear infinite}
        .p2{width:7px;height:7px;margin-left:68px;margin-top:-3.5px;background:#f6c37d;box-shadow:0 0 8px rgba(246,195,125,.5);animation:planet2 25s linear infinite reverse}
        .p3{width:8px;height:8px;margin-left:94px;margin-top:-4px;background:#5baaff;box-shadow:0 0 9px rgba(91,170,255,.58);animation:planet3 33s linear infinite}
        .p4{width:7px;height:7px;margin-left:125px;margin-top:-3.5px;background:#e08773;box-shadow:0 0 8px rgba(224,135,115,.42);animation:planet4 42s linear infinite reverse}
        .p5{width:14px;height:14px;margin-left:159px;margin-top:-7px;background:radial-gradient(circle at 35% 35%,#f3d59b,#d38f45 60%,#a2632b);box-shadow:0 0 12px rgba(230,170,95,.28);animation:planet5 54s linear infinite}
        .p5::after{content:"";position:absolute;left:50%;top:50%;width:23px;height:6px;margin:-3px -11.5px;border:1px solid rgba(238,213,170,.32);border-radius:50%;transform:rotate(-17deg)}
        .p6{width:11px;height:11px;margin-left:195px;margin-top:-5.5px;background:radial-gradient(circle at 35% 35%,#d2f1ff,#4f9ec4 58%,#2b5775);box-shadow:0 0 9px rgba(120,200,255,.2);animation:planet6 68s linear infinite reverse}
        .p7{width:12px;height:12px;margin-left:195px;margin-top:-6px;background:radial-gradient(circle at 35% 35%,#d7b07a,#8a6541 62%,#5e472f);box-shadow:0 0 8px rgba(211,176,122,.16);animation:planet7 80s linear infinite}
        .p8{width:10px;height:10px;margin-left:195px;margin-top:-5px;background:#9db7d6;box-shadow:0 0 8px rgba(157,183,214,.18);animation:planet8 96s linear infinite reverse}
        .orbit-glow{position:absolute;inset:-20px;border-radius:50%;background:radial-gradient(circle,rgba(77,177,255,.035),transparent 62%);filter:blur(18px)}
        .milky-way { position:absolute; width:1100px;height:260px;left:-280px;top:38%; transform:rotate(-18deg); opacity:.12; background:radial-gradient(ellipse at center,rgba(176,205,255,.7) 0%,rgba(114,152,255,.34) 30%,rgba(61,119,255,.08) 48%,transparent 74%); filter:blur(22px); }
        @keyframes orbitSpin { from{transform:rotate(0)} to{transform:rotate(360deg)} }
        @keyframes planet1 { from{transform:rotate(0)} to{transform:rotate(360deg)} }
        @keyframes planet2 { from{transform:rotate(0)} to{transform:rotate(360deg)} }
        @keyframes planet3 { from{transform:rotate(0)} to{transform:rotate(360deg)} }
        @keyframes planet4 { from{transform:rotate(0)} to{transform:rotate(360deg)} }
        @keyframes planet5 { from{transform:rotate(0)} to{transform:rotate(360deg)} }
        @keyframes planet6 { from{transform:rotate(0)} to{transform:rotate(360deg)} }
        @keyframes planet7 { from{transform:rotate(0)} to{transform:rotate(360deg)} }
        @keyframes planet8 { from{transform:rotate(0)} to{transform:rotate(360deg)} }
        /* Keep the animated space layer behind every Streamlit content surface. */
        [data-testid="stImage"], [data-testid="stImage"] img,
        [data-testid="stFileUploader"], [data-testid="stButton"],
        [data-testid="stDownloadButton"], [data-testid="stDataFrame"],
        [data-baseweb="select"], [data-baseweb="input"], textarea, input {
            position: relative;
            z-index: 2 !important;
        }
        .space-stars { position:absolute; inset:-20%; width:140%; height:140%; background-repeat:repeat; z-index:0; }
        .stars-a {
            background-size: 250px 250px; opacity:.55;
            background-image:
              radial-gradient(circle at 12px 24px, rgba(255,255,255,.95) 0 1px, transparent 1.5px),
              radial-gradient(circle at 78px 180px, rgba(133,211,255,.8) 0 1px, transparent 1.6px),
              radial-gradient(circle at 156px 70px, rgba(255,255,255,.82) 0 1px, transparent 1.5px),
              radial-gradient(circle at 210px 134px, rgba(177,198,255,.75) 0 1px, transparent 1.7px);
            animation: starDriftA 34s linear infinite;
        }
        .stars-b {
            background-size: 340px 340px; opacity:.30;
            background-image:
              radial-gradient(circle at 42px 90px, rgba(255,255,255,.8) 0 1.2px, transparent 1.8px),
              radial-gradient(circle at 190px 220px, rgba(103,189,255,.72) 0 1.1px, transparent 1.7px),
              radial-gradient(circle at 285px 55px, rgba(255,255,255,.66) 0 1px, transparent 1.6px);
            animation: starDriftB 52s linear infinite reverse;
        }
        .stars-c {
            background-size: 500px 500px; opacity:.16;
            background-image:
              radial-gradient(circle at 90px 290px, rgba(194,166,255,.65) 0 1.5px, transparent 2px),
              radial-gradient(circle at 365px 160px, rgba(255,255,255,.72) 0 1.3px, transparent 1.9px);
            animation: starDriftC 70s linear infinite;
        }
        .galaxy { position:absolute; border-radius:50%; filter:blur(14px); opacity:.22; mix-blend-mode:screen; }
        .galaxy::after { content:""; position:absolute; inset:22% 7%; border-radius:50%; border:7px solid rgba(119,203,255,.10); filter:blur(4px); transform:rotate(-18deg); }
        .galaxy-1 { width:500px; height:220px; left:-120px; top:14%; background:radial-gradient(ellipse at center, rgba(71,154,255,.82) 0%, rgba(92,73,255,.36) 20%, rgba(25,144,255,.10) 42%, transparent 70%); transform:rotate(-18deg); animation:galaxyFloat1 24s ease-in-out infinite alternate; }
        .galaxy-2 { width:420px; height:180px; right:-100px; top:48%; background:radial-gradient(ellipse at center, rgba(83,227,255,.68) 0%, rgba(72,99,255,.26) 22%, rgba(182,72,255,.09) 45%, transparent 72%); transform:rotate(24deg); animation:galaxyFloat2 30s ease-in-out infinite alternate; }
        .nebula { position:absolute; width:620px; height:320px; border-radius:50%; filter:blur(90px); opacity:.10; background:radial-gradient(circle, rgba(0,220,255,.5), rgba(83,89,255,.20) 35%, transparent 72%); animation:nebulaPulse 14s ease-in-out infinite alternate; }
        .nebula-1 { left:20%; bottom:-110px; } .nebula-2 { right:6%; top:-120px; transform:rotate(25deg); animation-delay:-5s; }
        .shooting-star { position:absolute; width:110px; height:2px; border-radius:999px; background:linear-gradient(90deg, rgba(255,255,255,0), rgba(133,220,255,.95)); filter:drop-shadow(0 0 6px rgba(111,213,255,.75)); opacity:0; transform:rotate(-28deg); }
        .shooting-star::after { content:""; position:absolute; right:0; top:50%; width:5px; height:5px; transform:translateY(-50%); border-radius:50%; background:#e9fbff; box-shadow:0 0 14px #8fe3ff; }
        .shoot-1 { left:12%; top:20%; animation:shoot 9s 2s linear infinite; }
        .shoot-2 { left:60%; top:12%; animation:shoot 12s 7s linear infinite; transform:rotate(-32deg) scale(.75); }

        @keyframes starDriftA { 0%{transform:translate3d(0,0,0)} 50%{transform:translate3d(-70px,55px,0)} 100%{transform:translate3d(-140px,110px,0)} }
        @keyframes starDriftB { 0%{transform:translate3d(0,0,0)} 50%{transform:translate3d(95px,-45px,0)} 100%{transform:translate3d(190px,-90px,0)} }
        @keyframes starDriftC { 0%{transform:translate3d(0,0,0) rotate(.0deg)} 50%{transform:translate3d(-40px,-90px,0) rotate(.3deg)} 100%{transform:translate3d(-80px,-180px,0) rotate(.6deg)} }
        @keyframes galaxyFloat1 { 0%{transform:translate3d(0,0,0) rotate(-18deg) scale(1)} 100%{transform:translate3d(110px,45px,0) rotate(-11deg) scale(1.08)} }
        @keyframes galaxyFloat2 { 0%{transform:translate3d(0,0,0) rotate(24deg) scale(1)} 100%{transform:translate3d(-100px,-35px,0) rotate(17deg) scale(1.1)} }
        @keyframes nebulaPulse { 0%{transform:scale(.94);opacity:.07} 100%{transform:scale(1.08);opacity:.13} }
        @keyframes shoot { 0%{opacity:0;transform:translate3d(0,0,0) rotate(-28deg)} 8%{opacity:.9} 35%{opacity:.75} 55%{opacity:0;transform:translate3d(420px,220px,0) rotate(-28deg)} 100%{opacity:0;transform:translate3d(560px,290px,0) rotate(-28deg)} }
        @media (prefers-reduced-motion: reduce) { .space-stars, .galaxy, .nebula, .shooting-star, .solar-system .orbit, .solar-system .planet { animation:none !important; } }

        [data-testid="stAppViewContainer"] { isolation: isolate; }
        [data-testid="stAppViewContainer"] > .main { position:relative; z-index:1; }
        .stApp { background:transparent !important; }
        .block-container { max-width:1500px; padding-top:1.2rem; padding-bottom:2.2rem; position:relative; z-index:1; }
        [data-testid="stHeader"] { background:transparent; }
        [data-testid="stToolbar"] { visibility:hidden; height:0; }

        [data-testid="stSidebar"] { background:#050b15 !important; border-right:1px solid #14304a; z-index:10; }
        [data-testid="stSidebarContent"] { padding-top:1rem; }
        [data-testid="stSidebar"] .stRadio > label { display:none; }
        [data-testid="stSidebar"] [role="radiogroup"] { gap:.38rem; }
        [data-testid="stSidebar"] [role="radiogroup"] label { padding:.72rem .85rem; border-radius:10px; border:1px solid transparent; color:#b4c2d6; transition:all .18s ease; }
        [data-testid="stSidebar"] [role="radiogroup"] label:hover { background:#0b1829; border-color:#173b5b; color:#eef7ff; }
        [data-testid="stSidebar"] [role="radiogroup"] label[data-checked="true"] { background:#0d2136; border-color:#1e638d; color:#65ddff; box-shadow:inset 3px 0 0 #38d9ff; }

        .brand { display:flex; align-items:center; gap:.75rem; margin:.2rem 0 1.4rem; }
        .brand-logo { width:44px; height:44px; display:flex; align-items:center; justify-content:center; border-radius:11px; background:linear-gradient(145deg,#143e68,#0b7d9a); border:1px solid #2a77a1; box-shadow:0 0 22px rgba(45,190,255,.16); font-size:1.35rem; }
        .brand-title { font-size:1.05rem; font-weight:800; letter-spacing:-.03em; }
        .brand-sub { color:#7589a3; font-size:.72rem; margin-top:.12rem; }

        .hero { padding:1.55rem 1.7rem; border:1px solid #1a3856; border-radius:20px; background:#07111f; box-shadow:0 16px 40px rgba(0,0,0,.28), inset 0 1px 0 rgba(255,255,255,.025); margin-bottom:1.1rem; position:relative; overflow:hidden; }
        .hero::after { content:""; position:absolute; top:0; right:0; width:260px; height:100%; background:linear-gradient(90deg, transparent, rgba(61,145,255,.10)); pointer-events:none; }
        .eyebrow { display:inline-flex; align-items:center; gap:.45rem; font-size:.72rem; font-weight:700; text-transform:uppercase; letter-spacing:.12em; color:#7ecaf4; margin-bottom:.62rem; }
        .hero h1 { margin:0; font-size:clamp(2rem,4vw,3.15rem); line-height:1.03; letter-spacing:-.055em; }
        .hero p { color:#93a7bf; max-width:820px; margin:.7rem 0 0; font-size:.94rem; line-height:1.65; }
        .status { display:inline-flex; align-items:center; gap:.42rem; margin-top:1rem; padding:.4rem .7rem; border-radius:8px; background:#09221f; border:1px solid #165849; color:#67e2cf; font-size:.74rem; font-weight:700; }
        .dot { width:7px; height:7px; border-radius:50%; background:#35d6bd; box-shadow:0 0 9px rgba(53,214,189,.75); }

        .section-title { font-size:1.04rem; font-weight:780; margin:1.15rem 0 .68rem; letter-spacing:-.02em; }
        .muted { color:#7f93ad; font-size:.83rem; }
        .metric-card { min-height:112px; padding:1rem 1.05rem; border-radius:14px; border:1px solid #173653; background:#07111f; box-shadow:0 10px 28px rgba(0,0,0,.16); }
        .metric-label { color:#8095ae; font-size:.72rem; font-weight:650; text-transform:uppercase; letter-spacing:.08em; }
        .metric-value { margin-top:.33rem; font-size:1.7rem; font-weight:800; letter-spacing:-.04em; }
        .metric-sub { margin-top:.2rem; color:#6f849e; font-size:.71rem; }

        .panel { border:1px solid #183957; background:#07111f; border-radius:15px; padding:1rem; box-shadow:0 10px 26px rgba(0,0,0,.16); }
        .panel-header { display:flex; align-items:center; justify-content:space-between; gap:1rem; margin-bottom:.75rem; }
        .panel-title { font-size:.95rem; font-weight:750; }
        .pill { display:inline-flex; align-items:center; padding:.27rem .55rem; border-radius:7px; background:#0c2135; color:#91aac5; font-size:.67rem; border:1px solid #1b4668; }
        .answer-card { padding:1.1rem 1.2rem; border-left:3px solid #42d9ff; border-radius:11px; background:#081827; border-top:1px solid #163957; border-right:1px solid #163957; border-bottom:1px solid #163957; box-shadow:0 12px 28px rgba(0,0,0,.15); }
        .answer-label { color:#8ea3bc; font-size:.71rem; text-transform:uppercase; letter-spacing:.09em; font-weight:700; }
        .answer-text { font-size:1.4rem; font-weight:720; line-height:1.35; margin-top:.38rem; }
        .feature-card { padding:1rem; border-radius:14px; border:1px solid #173653; background:#07111f; height:100%; box-shadow:0 10px 26px rgba(0,0,0,.14); transition:transform .16s ease, border-color .16s ease, box-shadow .16s ease; }
        .feature-card:hover { transform:translateY(-2px); border-color:#276188; box-shadow:0 14px 32px rgba(0,0,0,.22); }
        .feature-icon { font-size:1.2rem; } .feature-name { font-weight:700; margin:.45rem 0 .25rem; } .feature-desc { color:#7f93ad; font-size:.77rem; line-height:1.55; }
        .upload-note { padding:.72rem .9rem; border-radius:10px; background:#081827; border:1px dashed #28506f; color:#899db6; font-size:.76rem; }

        div[data-testid="stFileUploader"] section { border-radius:12px; border:1px dashed #28506f; background:#07111f; }
        /* Dark satellite-theme buttons: prevent Streamlit's default white buttons. */
        div[data-testid="stButton"] > button,
        div[data-testid="stDownloadButton"] > button {
            border-radius:10px !important;
            font-weight:750 !important;
            min-height:2.65rem !important;
            transition:transform .15s ease,box-shadow .15s ease,border-color .15s ease,background .15s ease !important;
            cursor:pointer !important;
            background:#0a1727 !important;
            color:#dff5ff !important;
            border:1px solid #214663 !important;
            box-shadow:inset 0 1px 0 rgba(255,255,255,.025) !important;
        }
        div[data-testid="stButton"] > button:hover,
        div[data-testid="stDownloadButton"] > button:hover {
            transform:translateY(-1px);
            box-shadow:0 8px 22px rgba(0,120,255,.16) !important;
            border-color:#2b7aad !important;
            background:#0d2136 !important;
            color:#ffffff !important;
        }
        div[data-testid="stButton"] > button[kind="primary"],
        div[data-testid="stButton"] > button[data-testid="baseButton-primary"] {
            background:linear-gradient(90deg,#126b94,#1b8da2) !important;
            border:1px solid #2aa4bf !important;
            color:white !important;
        }
        div[data-testid="stButton"] > button[kind="secondary"] {
            background:#0a1727 !important;
            color:#dff5ff !important;
        }
        div[data-testid="stExpander"] { border-radius:12px; border:1px solid #1a4161; background:#07111f; }
        div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, textarea, input { border-radius:10px !important; }
        div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, textarea, input { background:#081522 !important; border-color:#1d4665 !important; }
        div[data-testid="stCheckbox"] label, div[data-testid="stRadio"] label { cursor:pointer !important; }
        .space-tag { display:inline-flex; align-items:center; gap:.45rem; padding:.3rem .58rem; border-radius:7px; font-size:.67rem; font-weight:750; color:#91dcff; background:#092039; border:1px solid #1f5a82; }
        div[data-testid="stDataFrame"] { border-radius:12px; overflow:hidden; border:1px solid #173653; }
        .legend-row { display:flex; flex-wrap:wrap; gap:.42rem; margin:.55rem 0 0; }
        .legend-chip { display:inline-flex; align-items:center; gap:.35rem; padding:.28rem .52rem; border-radius:7px; background:#081827; font-size:.67rem; color:#8da0b8; border:1px solid #1a3b58; }
        .legend-dot { width:8px; height:8px; border-radius:50%; display:inline-block; }
        .footer { text-align:center; color:#5f728a; font-size:.72rem; padding:1.6rem 0 .2rem; }

     /* HERO EARTH / GLOBE */
     .hero { position:relative; overflow:hidden; min-height:360px; display:flex; align-items:center; justify-content:space-between; gap:2rem; }
     .hero-copy { position:relative; z-index:4; width:58%; }
     .hero-globe-wrap { position:absolute; right:2.5%; top:50%; width:360px; height:360px; transform:translateY(-50%); z-index:2; pointer-events:none; }
     .hero-globe { position:absolute; left:50%; top:50%; width:235px; height:235px; transform:translate(-50%,-50%); border-radius:50%; overflow:hidden; background:radial-gradient(circle at 34% 28%, #73f5ff 0%, #18a9d0 17%, #075f94 43%, #062a55 70%, #020d1d 100%); box-shadow:0 0 25px rgba(0,225,255,.55), 0 0 70px rgba(0,155,255,.28), inset -28px -22px 55px rgba(0,0,0,.65); animation:globeFloat 5s ease-in-out infinite; }
     .hero-globe::before { content:""; position:absolute; inset:-8%; border-radius:50%; background:repeating-linear-gradient(12deg, transparent 0 20px, rgba(116,245,255,.15) 21px 23px, transparent 24px 42px); transform:rotate(-12deg); animation:globeTexture 8s linear infinite; }
     .hero-globe::after { content:""; position:absolute; inset:0; border-radius:50%; background:linear-gradient(120deg, rgba(255,255,255,.34), transparent 28%, transparent 67%, rgba(0,0,0,.48)); mix-blend-mode:screen; }
     .globe-grid { position:absolute; inset:-15%; border:1px solid rgba(93,239,255,.34); border-radius:50%; z-index:2; }
     .globe-grid-h { transform:rotateX(65deg) scaleY(.72); box-shadow:0 0 0 13px rgba(70,218,255,.06), 0 0 0 28px rgba(70,218,255,.04); }
     .globe-grid-v { transform:rotateY(65deg) scaleX(.72); }
     .globe-shine { position:absolute; width:90px; height:90px; left:30px; top:22px; border-radius:50%; background:radial-gradient(circle, rgba(255,255,255,.55), rgba(255,255,255,.08) 38%, transparent 70%); filter:blur(4px); z-index:4; }
     .globe-cloud { position:absolute; z-index:3; border-radius:50%; filter:blur(7px); background:rgba(124,250,255,.24); }
     .cloud-a { width:120px; height:34px; left:75px; top:92px; transform:rotate(-18deg); animation:cloudDrift 7s ease-in-out infinite; }
     .cloud-b { width:90px; height:25px; left:35px; top:150px; transform:rotate(13deg); animation:cloudDrift 9s ease-in-out infinite reverse; }
     .hero-orbit { position:absolute; left:50%; top:50%; width:325px; height:125px; border:1px solid rgba(74,231,255,.42); border-radius:50%; transform:translate(-50%,-50%) rotate(-20deg); box-shadow:0 0 15px rgba(0,207,255,.14); }
     .orbit-b { width:300px; height:160px; transform:translate(-50%,-50%) rotate(54deg); border-color:rgba(72,191,255,.22); }
     .hero-orbit::after { content:""; position:absolute; width:7px; height:7px; right:13%; top:8%; border-radius:50%; background:#62f6ff; box-shadow:0 0 12px #2bdfff, 0 0 24px #1bb9ff; }
     .globe-scanline { position:absolute; left:50%; top:50%; width:260px; height:2px; transform:translate(-50%,-50%); background:linear-gradient(90deg, transparent, rgba(93,245,255,.7), transparent); filter:blur(.4px); animation:scanGlobe 3.4s ease-in-out infinite; z-index:6; }
     @keyframes globeFloat { 0%,100%{transform:translate(-50%,-50%) translateY(0) rotate(-1deg)} 50%{transform:translate(-50%,-50%) translateY(-7px) rotate(1deg)} }
     @keyframes globeTexture { from{transform:translateX(-25px) rotate(-12deg)} to{transform:translateX(25px) rotate(-12deg)} }
     @keyframes cloudDrift { 0%,100%{translate:0 0} 50%{translate:12px -5px} }
     @keyframes scanGlobe { 0%,100%{opacity:.1; transform:translate(-50%,-110px)} 50%{opacity:.8; transform:translate(-50%,110px)} }
     @media (max-width:900px) { .hero { min-height:520px; } .hero-copy { width:100%; } .hero-globe-wrap { opacity:.55; right:50%; transform:translate(50%,-15%); top:68%; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title, subtitle, eyebrow="SATQUERY AI", status_text=None, show_globe=False):
    status = f'<div class="status"><span class="dot"></span>{html.escape(status_text)}</div>' if status_text else ''
    globe = ""
    if show_globe:
        globe = '''
        <div class="hero-globe-wrap" aria-hidden="true">
            <div class="hero-orbit orbit-a"></div>
            <div class="hero-orbit orbit-b"></div>
            <div class="hero-globe">
                <div class="globe-grid globe-grid-h"></div>
                <div class="globe-grid globe-grid-v"></div>
                <div class="globe-shine"></div>
                <div class="globe-cloud cloud-a"></div>
                <div class="globe-cloud cloud-b"></div>
            </div>
            <div class="globe-scanline"></div>
        </div>
        '''
    st.markdown(f'''<div class="hero"><div class="hero-copy"><div class="space-tag">✦ ORBITAL VISION SYSTEM</div><div class="eyebrow" style="margin-top:.8rem">🛰️ {html.escape(eyebrow)}</div><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p>{status}</div>{globe}</div>''', unsafe_allow_html=True)


def metric_card(label, value, sub=""):
    st.markdown(f'''<div class="metric-card"><div class="metric-label">{html.escape(str(label))}</div><div class="metric-value">{html.escape(str(value))}</div><div class="metric-sub">{html.escape(str(sub))}</div></div>''', unsafe_allow_html=True)


def section(title, subtitle=None):
    st.markdown(f'<div class="section-title">{html.escape(title)}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="muted">{html.escape(subtitle)}</div>', unsafe_allow_html=True)


inject_ui()

# Animated space background layers
st.markdown(
    '''
    <div class="space-bg" aria-hidden="true">
        <div class="space-stars stars-a"></div>
        <div class="space-stars stars-b"></div>
        <div class="space-stars stars-c"></div>
        <div class="galaxy galaxy-1"></div>
        <div class="galaxy galaxy-2"></div>
        <div class="milky-way"></div>
        <div class="solar-system">
            <div class="orbit-glow"></div>
            <div class="sun"></div>
            <div class="orbit o1"></div>
            <div class="orbit o2"></div>
            <div class="orbit o3"></div>
            <div class="orbit o4"></div>
            <div class="orbit o5"></div>
            <div class="orbit o6"></div>
            <div class="asteroid-belt"></div>
            <div class="planet p1"></div>
            <div class="planet p2"></div>
            <div class="planet p3"></div>
            <div class="planet p4"></div>
            <div class="planet p5"></div>
            <div class="planet p6"></div>
            <div class="planet p7"></div>
            <div class="planet p8"></div>
        </div>
        <div class="nebula nebula-1"></div>
        <div class="nebula nebula-2"></div>
        <div class="shooting-star shoot-1"></div>
        <div class="shooting-star shoot-2"></div>
    </div>
    ''',
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
# Always start the application on the Dashboard for a fresh session.
if "page_nav" not in st.session_state:
    st.session_state.page_nav = "Dashboard"

with st.sidebar:
    st.markdown('''<div class="brand"><div class="brand-logo">🛰️</div><div><div class="brand-title">SatQuery AI</div><div class="brand-sub">Remote Sensing Intelligence</div></div></div>''', unsafe_allow_html=True)
    page = st.radio(
        "Navigation",
        ["Dashboard", "Image Analysis", "Map Identification", "Analytics", "Dataset", "Model"],
        key="page_nav",
        label_visibility="collapsed",
    )
    st.markdown('<div style="height:.45rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">SYSTEM</div>', unsafe_allow_html=True)
    compute = "CUDA GPU" if torch.cuda.is_available() else "CPU"
    st.markdown(f'''<div class="panel" style="margin-top:.45rem"><div style="font-size:.76rem;color:#8b95a8">Vision-Language Model</div><div style="font-size:.78rem;font-weight:700;margin-top:.25rem;word-break:break-word">BLIP VQA</div><div style="font-size:.71rem;color:#68738a;margin-top:.55rem">Compute: {html.escape(compute)}</div></div>''', unsafe_allow_html=True)
    st.markdown('<div style="height:.55rem"></div>', unsafe_allow_html=True)
    if st.button("🗑️  Clear Query History", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# ------------------------------------------------------------
# BACK / NEXT NAVIGATION
# ------------------------------------------------------------
# Keep the existing sidebar and page design unchanged. These two controls
# provide a simple workflow navigation without removing the current options.
if "previous_page" not in st.session_state:
    st.session_state.previous_page = None
if "_last_rendered_page" not in st.session_state:
    st.session_state._last_rendered_page = page
if "_page_changed_by_callback" not in st.session_state:
    st.session_state._page_changed_by_callback = False

# If the user changed the page directly from the sidebar, remember where they
# came from so the Back button returns there.
if (
    page != st.session_state._last_rendered_page
    and not st.session_state._page_changed_by_callback
):
    st.session_state.previous_page = st.session_state._last_rendered_page

st.session_state._page_changed_by_callback = False
st.session_state._last_rendered_page = page

nav_left, nav_center, nav_right = st.columns([1.2, 5.6, 1.2])
with nav_left:
    st.button(
        "← Back",
        use_container_width=True,
        key="global_back_btn",
        on_click=go_back,
        disabled=(page == "Dashboard"),
    )
with nav_center:
    st.markdown(
        f'<div style="text-align:center;color:#8ea1bb;font-size:.78rem;padding-top:.55rem;letter-spacing:.04em;">'
        f'STEP {PAGE_ORDER.index(page) + 1} / {len(PAGE_ORDER)} &nbsp;•&nbsp; {html.escape(page)}'
        f'</div>',
        unsafe_allow_html=True,
    )
with nav_right:
    st.button(
        "Next →",
        use_container_width=True,
        key="global_next_btn",
        on_click=go_next,
        disabled=(page == PAGE_ORDER[-1]),
    )

st.markdown('<div style="height:.35rem"></div>', unsafe_allow_html=True)

# ============================================================
# DASHBOARD
# ============================================================
if page == "Dashboard":
    hero("Observe Earth in high fidelity.", "SatQuery AI combines a vision-language assistant with remote-sensing tools in a single interactive workspace.", status_text="System ready", show_globe=True)
    history = st.session_state.history
    total = len(history)
    avg = sum(x["Response Time (s)"] for x in history) / total if total else 0
    device = "GPU" if torch.cuda.is_available() else "CPU"
    latest = history[-1]["Answer"] if history else "No queries yet"
    cols = st.columns(4, gap="medium")
    with cols[0]: metric_card("Queries", total, "Questions processed")
    with cols[1]: metric_card("Avg. response", f"{avg:.2f}s", "Across this session")
    with cols[2]: metric_card("Compute", device, "Inference device")
    with cols[3]: metric_card("Latest answer", latest if len(latest) < 32 else latest[:29] + "…", "Most recent VQA result")
    section("What you can do", "A focused workspace for image understanding, map context, and experiment tracking.")
    fcols = st.columns(3, gap="medium")
    features = [
        ("💬", "Ask visual questions", "Upload a satellite or map image and ask natural-language questions with BLIP VQA.", "Image Analysis", "Open Image Analysis"),
        ("🎯", "Create visual overlays", "Generate colored region candidates, candidate regions and line features directly on the image.", "Image Analysis", "Open Overlay Tools"),
        ("🌍", "Identify map context", "Estimate country and map details, inspect neighbouring countries, and view an interactive map.", "Map Identification", "Open Map Identification"),
    ]
    for idx, (col, (icon, name, desc, target_page, action_label)) in enumerate(zip(fcols, features)):
        with col:
            st.markdown(
                f'<div class="feature-card"><div class="feature-icon">{icon}</div><div class="feature-name">{html.escape(name)}</div><div class="feature-desc">{html.escape(desc)}</div></div>',
                unsafe_allow_html=True,
            )
            st.button(
                action_label + "  →",
                use_container_width=True,
                key=f"dashboard_action_{idx}",
                on_click=set_page,
                args=(target_page,),
            )
    st.markdown('<div style="height:.35rem"></div>', unsafe_allow_html=True)
    section("Recent activity")
    if history:
        df = pd.DataFrame(history).tail(6).copy()
        df.insert(0, "#", range(len(history) - len(df) + 1, len(history) + 1))
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.markdown('<div class="upload-note">No activity yet. Use a workspace button above or the sidebar to start your first query.</div>', unsafe_allow_html=True)
        st.button(
            "🚀 Start Your First Query",
            type="primary",
            use_container_width=False,
            key="dashboard_start_query",
            on_click=set_page,
            args=("Image Analysis",),
        )

# ============================================================
# IMAGE ANALYSIS
# ============================================================
elif page == "Image Analysis":
    hero("Image Analysis", "Upload your own remote-sensing image, ask a text question, and generate optional visual annotations.", eyebrow="IMAGE WORKSPACE")
    uploaded = st.file_uploader("Upload your own map or satellite image", type=["jpg", "jpeg", "png", "webp", "tif", "tiff"], key="analysis_upload", help="Supported: JPG, JPEG, PNG, WEBP, TIF and TIFF")
    if uploaded is None:
        st.markdown('<div class="upload-note">💡 Tip: use a clear satellite or map image. Larger images may take longer for inference and overlay generation.</div>', unsafe_allow_html=True)
    else:
        try:
            image = Image.open(uploaded).convert("RGB")
        except Exception as exc:
            st.error("Could not open this image.")
            st.exception(exc)
            st.stop()
        file_bytes = uploaded.getvalue()
        file_signature = hashlib.sha256(file_bytes).hexdigest()
        if st.session_state.analysis_file_signature != file_signature:
            st.session_state.analysis_file_signature = file_signature
            st.session_state.analysis_file_name = uploaded.name
            st.session_state.analysis_result = None
            st.session_state.overlay_result = None
            st.session_state.zoom_result = None
            st.session_state.deep_scan_result = None
            st.session_state.analysis_question = "Is there water in the image?"

        left, right = st.columns([1.25, .95], gap="large")
        with left:
            section("Uploaded image", uploaded.name)
            render_zoomable_image(image, uploaded.name, height=560, key=f"analysis_{file_signature}")
            stats = st.columns(3)
            with stats[0]: metric_card("Width", f"{image.width}px", "Pixels")
            with stats[1]: metric_card("Height", f"{image.height}px", "Pixels")
            with stats[2]: metric_card("Format", uploaded.name.split(".")[-1].upper(), "Input type")

            st.markdown('<div class="muted" style="margin-top:1rem;margin-bottom:.35rem">ZOOM-AWARE ANALYSIS</div>', unsafe_allow_html=True)
            st.caption("Viewer zoom is for inspection. Use the region controls below to make the AI analyze the zoomed area itself.")
            z1, z2 = st.columns(2)
            with z1:
                roi_x = st.slider("Region X (%)", 0, 100, 0, key="roi_x")
                roi_y = st.slider("Region Y (%)", 0, 100, 0, key="roi_y")
            with z2:
                roi_w = st.slider("Region width (%)", 1, 100, 100, key="roi_w")
                roi_h = st.slider("Region height (%)", 1, 100, 100, key="roi_h")

            roi_image = crop_percent_region(image, roi_x, roi_y, roi_w, roi_h)
            render_zoomable_image(roi_image, "Selected analysis region", height=360, key=f"roi_{file_signature}")

            rz1, rz2 = st.columns(2)
            with rz1:
                if st.button("🔎 Analyze Zoomed Region", use_container_width=True, key="analyze_zoom_btn"):
                    active_question = str(st.session_state.analysis_question).strip()
                    if not is_valid_question(active_question):
                        st.error("❌ Enter a valid image-related question before analyzing the zoomed region.")
                    else:
                        with st.spinner("Analyzing the selected zoomed region…"):
                            try:
                                zoom_answer, zoom_elapsed, zoom_device = ask_vqa(roi_image, active_question)
                                st.session_state.zoom_result = {
                                    "answer": zoom_answer,
                                    "elapsed": zoom_elapsed,
                                    "device": zoom_device,
                                    "question": active_question,
                                    "region": (roi_x, roi_y, roi_w, roi_h),
                                }
                            except Exception as exc:
                                st.session_state.zoom_result = None
                                st.error("Zoomed-region analysis failed.")
                                st.exception(exc)
            with rz2:
                if st.button("🔬 Deep Scan Image", use_container_width=True, key="deep_scan_btn"):
                    active_question = str(st.session_state.analysis_question).strip()
                    if not is_valid_question(active_question):
                        st.error("❌ Enter a valid image-related question before starting the deep scan.")
                    else:
                        with st.spinner("Scanning multiple image regions…"):
                            try:
                                scan_items, scan_summary = deep_scan_image(image, active_question, rows=3, cols=3)
                                st.session_state.deep_scan_result = {
                                    "items": scan_items,
                                    "summary": scan_summary,
                                    "question": active_question,
                                }
                            except Exception as exc:
                                st.session_state.deep_scan_result = None
                                st.error("Deep scan failed.")
                                st.exception(exc)

            if st.session_state.zoom_result:
                zr = st.session_state.zoom_result
                st.markdown('<div class="answer-card"><div class="answer-label">ZOOMED REGION ANSWER</div><div class="answer-text">' + html.escape(zr["answer"]) + '</div></div>', unsafe_allow_html=True)
                st.caption(f'Zoom region x={zr["region"][0]}%, y={zr["region"][1]}%, width={zr["region"][2]}%, height={zr["region"][3]}% • {zr["device"].upper()} • {zr["elapsed"]:.2f}s')

            if st.session_state.deep_scan_result:
                ds = st.session_state.deep_scan_result
                st.markdown(f'<div class="answer-card"><div class="answer-label">DEEP SCAN SUMMARY</div><div class="answer-text">{html.escape(ds["summary"])}</div></div>', unsafe_allow_html=True)
                scan_cols = st.columns(3)
                for i, item in enumerate(ds["items"]):
                    with scan_cols[i % 3]:
                        st.image(item["image"], caption=f'{item["tile"]}: {item["answer"]}', use_container_width=True)

        with right:
            section("Ask SatQuery AI", "Choose a quick query or type your own question.")

            st.markdown('<div class="muted" style="margin-bottom:.35rem">QUICK QUERIES</div>', unsafe_allow_html=True)
            quick_queries = [
                "Is there water in the image?",
                "Are there buildings in the image?",
                "Is there vegetation in the image?",
                "Is this an urban area?",
                "What is the main object in the image?",
                "Describe the image."
            ]
            qcols = st.columns(2)
            for idx, prompt in enumerate(quick_queries):
                with qcols[idx % 2]:
                    st.button(
                        prompt,
                        use_container_width=True,
                        key=f"quick_query_{idx}",
                        on_click=set_analysis_question,
                        args=(prompt,),
                    )

            question = st.text_area(
                "Active query",
                placeholder="Type your own question here…",
                height=105,
                key="analysis_question",
                help="Enter a meaningful question about the uploaded image. Invalid or unrelated text will be rejected."
            )

            question_clean = " ".join(str(question).strip().split())
            if question_clean and not is_valid_question(question_clean):
                st.warning(
                    "⚠️ Invalid question. Please enter a meaningful image-related question, "
                    "for example: 'Is there water in the image?'"
                )
            else:
                st.caption("✅ Valid query. Press Analyze Image to send it to BLIP VQA.")

            with st.expander("⚙️ Detection options", expanded=True):
                opt1, opt2 = st.columns(2)
                with opt1:
                    do_regions = st.checkbox(
                        "Water / vegetation / construction candidates",
                        value=True,
                        key="opt_regions"
                    )
                with opt2:
                    do_lines = st.checkbox(
                        "Draw line / road candidates",
                        value=False,
                        key="opt_lines"
                    )
                st.info(
                    "Object bounding-box detection is temporarily disabled for this optical satellite image. "
                    "The previous generic detector produced false labels such as 'boat'. "
                    "A satellite-trained detector will be added only after it is validated on remote-sensing data."
                )

            analyze_clicked = st.button("🚀 Analyze Image", type="primary", use_container_width=True, key="analyze_image_btn")
            if analyze_clicked:
                active_question = str(st.session_state.analysis_question).strip()
                if not active_question:
                    st.warning("⚠️ Please enter a question before starting the analysis.")
                elif not is_valid_question(active_question):
                    st.error(
                        "❌ Invalid question. Please enter a meaningful question about the uploaded image "
                        "(for example: 'Are there buildings in the image?')."
                    )
                else:
                    with st.spinner("Running Vision-Language analysis…"):
                        try:
                            answer, elapsed, device = ask_vqa(image, active_question)
                            save_query(uploaded.name, active_question, answer, elapsed, device)
                            st.session_state.analysis_result = {
                                "answer": answer, "elapsed": elapsed, "device": device,
                                "question": active_question, "image": uploaded.name
                            }
                            st.success("Analysis completed")
                        except Exception as exc:
                            st.session_state.analysis_result = None
                            st.error("VQA analysis failed.")
                            st.exception(exc)

            if st.session_state.analysis_result:
                result = st.session_state.analysis_result
                st.markdown(f'<div class="answer-card"><div class="answer-label">AI Answer</div><div class="answer-text">{html.escape(result["answer"])}</div></div>', unsafe_allow_html=True)
                st.caption(f'BLIP VQA • {result["device"].upper()} • {result["elapsed"]:.2f} seconds')
        st.divider()
        section("Visual Region Detection", "Optional overlays help inspect candidate areas and linear features in the uploaded image.")
        if st.button("🎯 Generate Colored Boxes & Region Outlines", use_container_width=True, key="generate_overlay_btn"):
            annotated = image.copy()
            rows = []
            errors = []
            with st.spinner("Generating visual overlays…"):
                if do_regions:
                    try:
                        annotated, region_rows = add_region_candidates(annotated)
                        rows.extend(region_rows)
                    except Exception as exc:
                        errors.append(f"Region detection: {exc}")
                if do_lines:
                    try:
                        annotated, line_rows = add_line_candidates(annotated)
                        rows.extend(line_rows)
                    except Exception as exc:
                        errors.append(f"Line detection: {exc}")
            st.session_state.overlay_result = {"image": annotated, "rows": rows, "errors": errors}

        if st.session_state.overlay_result:
            overlay = st.session_state.overlay_result
            render_zoomable_image(overlay["image"], "Annotated result", height=560, key=f"overlay_{st.session_state.analysis_file_signature}")
            st.markdown('<div class="legend-row"><span class="legend-chip"><span class="legend-dot" style="background:#237df5"></span>Water candidate</span><span class="legend-chip"><span class="legend-dot" style="background:#2db446"></span>Vegetation candidate</span><span class="legend-chip"><span class="legend-dot" style="background:#f04b2d"></span>Construction candidate</span><span class="legend-chip"><span class="legend-dot" style="background:#f5cd23"></span>Road / line candidate</span></div>', unsafe_allow_html=True)
            for message in overlay["errors"]:
                st.warning(message)
            if overlay["rows"]:
                st.dataframe(pd.DataFrame(overlay["rows"]), use_container_width=True, hide_index=True)
            else:
                st.info("No candidate regions were found with the selected settings. Try enabling the region options.")

# ============================================================
# MAP IDENTIFICATION
# ============================================================
elif page == "Map Identification":
    hero("Map Identification", "Estimate the country and map details from an uploaded map or satellite image, then inspect geographic context.", eyebrow="GEOGRAPHIC CONTEXT")
    uploaded = st.file_uploader("Upload any country map or satellite image", type=["jpg", "jpeg", "png", "webp", "tif", "tiff"], key="map_upload")
    if uploaded is None:
        st.markdown('<div class="upload-note">🌍 Upload a map to identify its country, map type, map name, and neighbouring countries.</div>', unsafe_allow_html=True)
    else:
        image = Image.open(uploaded).convert("RGB")
        if st.session_state.map_file_name != uploaded.name:
            st.session_state.map_file_name = uploaded.name
            st.session_state.map_info = None
        left, right = st.columns([1.1, .9], gap="large")
        with left:
            render_zoomable_image(image, uploaded.name, height=520, key=f"map_{st.session_state.map_file_name}")
        with right:
            st.markdown('<div class="panel"><div class="panel-header"><div class="panel-title">Map analysis</div><span class="pill">BLIP VQA</span></div><div class="muted">Run the identification model, then choose a country for geographic lookup.</div></div>', unsafe_allow_html=True)
            if st.button("🌍 Identify Map", type="primary", use_container_width=True, key="identify_map_btn"):
                with st.spinner("Identifying map…"):
                    st.session_state.map_info = identify_map(image)
        if st.session_state.map_info:
            info = st.session_state.map_info
            ai_country = normalize_country(info.get("country", ""))
            countries = get_country_list()
            idx = countries.index(ai_country) if ai_country in countries else 0
            selected = st.selectbox("Country for geographic lookup", countries, index=idx)
            neighbors = get_neighbors(selected)

            admin_label = get_admin_area_label(selected)
            detected_admin = normalize_admin_area(info.get("admin_area", ""), selected)
            admin_list = get_admin1_list(selected)
            admin_idx = admin_list.index(detected_admin) if detected_admin in admin_list else 0

            st.divider()
            c1, c2, c3, c4 = st.columns(4)
            with c1: metric_card("AI country", ai_country or "Unknown", "Model estimate")
            with c2: metric_card("Detected area", detected_admin or "Unknown", admin_label)
            with c3: metric_card("Neighbours", len(neighbors), "Land neighbours found")
            with c4: metric_card("Image type", uploaded.name.split(".")[-1].upper(), "Uploaded format")

            if admin_list:
                selected_admin = st.selectbox(
                    f"{admin_label} for geographic lookup",
                    admin_list,
                    index=admin_idx,
                    help="BLIP provides the initial estimate. Use this selector to correct or inspect the country's administrative subdivisions.",
                )
            else:
                selected_admin = detected_admin
                st.info(f"Administrative boundaries are not available from the local ISO subdivision catalogue for {selected}.")

            details = st.columns(2, gap="large")
            with details[0]:
                section("Map name")
                st.markdown(f'<div class="panel">{html.escape(str(info.get("map_name", "Unknown")))}</div>', unsafe_allow_html=True)
                section("Map type")
                st.markdown(f'<div class="panel">{html.escape(str(info.get("map_type", "Unknown")))}</div>', unsafe_allow_html=True)
                section(f"Detected {admin_label}")
                st.markdown(f'<div class="panel"><strong>{html.escape(str(selected_admin or "Unknown"))}</strong><br><span class="muted">VQA estimate: {html.escape(str(info.get("admin_area", "Unknown")))}</span></div>', unsafe_allow_html=True)
            with details[1]:
                section("Countries around it")
                if neighbors:
                    ncols = st.columns(3)
                    for i, country in enumerate(neighbors):
                        ncols[i % 3].success(country)
                else:
                    st.info("No land neighbours found or this is an island country.")

                if admin_list:
                    section(f"Administrative areas in {selected}")
                    st.caption(f"{len(admin_list)} subdivisions available for lookup")
                    preview = admin_list[:24]
                    area_cols = st.columns(3)
                    for i, area in enumerate(preview):
                        area_cols[i % 3].markdown(f"`{area}`")

            section("Interactive country map")
            render_country_map(selected)
            geo = read_geotiff_metadata(uploaded.getvalue())
            if geo:
                section("GeoTIFF metadata")
                st.dataframe(pd.DataFrame(geo.items(), columns=["Property", "Value"]), use_container_width=True, hide_index=True)
            st.caption("For PNG/JPG maps, country identification is an AI estimate and can be corrected with the country selector. For geospatial TIFFs, metadata provides spatial reference information when present.")

# ============================================================
# ANALYTICS
# ============================================================
elif page == "Analytics":
    hero("Dynamic Analytics", "Track query volume, response time, and session history while you experiment with the assistant.", eyebrow="SESSION INSIGHTS")
    if not st.session_state.history:
        st.info("Run some questions from Image Analysis first to populate the analytics dashboard.")
    else:
        df = pd.DataFrame(st.session_state.history)
        cols = st.columns(3)
        with cols[0]: metric_card("Total queries", len(df), "This session")
        with cols[1]: metric_card("Average response", f"{df['Response Time (s)'].mean():.2f}s", "Mean inference time")
        with cols[2]: metric_card("Fastest response", f"{df['Response Time (s)'].min():.2f}s", "Best recorded query")
        l, r = st.columns(2, gap="large")
        with l:
            section("Response time")
            chart = df[["Response Time (s)"]].copy()
            chart.index = range(1, len(chart) + 1)
            st.line_chart(chart)
        with r:
            section("Question frequency")
            st.bar_chart(df["Question"].value_counts())
        section("Query history")
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download CSV", df.to_csv(index=False).encode("utf-8"), "satquery_history.csv", "text/csv", use_container_width=True)

# ============================================================
# DATASET
# ============================================================
elif page == "Dataset":
    hero("RSVQA Dataset", "Inspect the local RSVQA assets used for remote-sensing VQA experiments and previews.", eyebrow="DATA & EVALUATION")
    if not RSVQA_DIR.exists():
        st.warning(f"Dataset folder not found: {RSVQA_DIR}")
    else:
        json_files = sorted(RSVQA_DIR.glob("*.json"))
        tif_count = len(list(RSVQA_IMAGE_DIR.glob("*.tif"))) if RSVQA_IMAGE_DIR.exists() else 0
        c1, c2 = st.columns(2)
        with c1: metric_card("RSVQA TIFF images", tif_count, "Local image files")
        with c2: metric_card("Annotation JSON files", len(json_files), "Local metadata / labels")
        section("Available annotation files")
        st.dataframe(pd.DataFrame({"File": [f.name for f in json_files]}), use_container_width=True, hide_index=True)
        if json_files:
            selected = st.selectbox("Preview JSON", [f.name for f in json_files])
            path = RSVQA_DIR / selected
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                meta_cols = st.columns(2)
                with meta_cols[0]: metric_card("Type", type(data).__name__, "Parsed successfully")
                with meta_cols[1]:
                    count = len(data) if isinstance(data, (list, dict)) else 1
                    metric_card("Records / keys", count, "Top-level size")
                with st.expander("Preview content", expanded=True):
                    if isinstance(data, list):
                        st.json(data[:3])
                    elif isinstance(data, dict):
                        st.json(dict(list(data.items())[:3]))
                    else:
                        st.write(data)
            except Exception as exc:
                st.error("Could not read this JSON file.")
                st.exception(exc)

# ============================================================
# MODEL
# ============================================================
elif page == "Model":
    hero("Model & System", "A transparent view of the models and compute used by the current SatQuery AI implementation.", eyebrow="SYSTEM STATUS")
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown('<div class="feature-card"><div class="feature-icon">🧠</div><div class="feature-name">Vision-Language Model</div><div class="feature-desc">Salesforce BLIP VQA for image-grounded natural-language question answering.</div></div>', unsafe_allow_html=True)
        st.code(VQA_MODEL_NAME, language="text")
    with c2:
        detector_desc = (
            "Object detection is disabled for the current optical satellite workflow. "
            "A validated remote-sensing checkpoint is required before bounding boxes are shown."
        )
        st.markdown(f'<div class="feature-card"><div class="feature-icon">🎯</div><div class="feature-name">Object Detector</div><div class="feature-desc">{html.escape(detector_desc)}</div></div>', unsafe_allow_html=True)
        st.code("Disabled until a validated remote-sensing model is added", language="text")
    with c3:
        gpu_text = "CUDA GPU available" if torch.cuda.is_available() else "CPU mode"
        st.markdown('<div class="feature-card"><div class="feature-icon">⚡</div><div class="feature-name">Compute</div><div class="feature-desc">Runtime automatically selects CUDA when available and falls back to CPU otherwise.</div></div>', unsafe_allow_html=True)
        st.success(gpu_text)
    section("Current implementation")
    st.markdown('<div class="panel"><div class="panel-title">Interactive inference workspace</div><div class="muted" style="margin-top:.4rem;line-height:1.65">Upload an image → ask a natural-language question → receive a VQA answer → optionally generate visual overlays → review results and session analytics.</div></div>', unsafe_allow_html=True)
    st.warning("The colored water/vegetation/construction overlays are visual candidate regions, not a trained satellite segmentation model. For the final research version, replace these heuristics with a remote-sensing segmentation model trained and evaluated on suitable data.")

st.markdown('<div class="footer">SatQuery AI • Interactive Vision-Language Assistant for Remote Sensing</div>', unsafe_allow_html=True)
