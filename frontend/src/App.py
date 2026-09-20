import streamlit as st
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"

st.set_page_config(
    page_title="SatQuery AI",
    page_icon="🛰️",
    layout="wide"
)


@st.cache_resource
def load_model():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )

    return processor, model


def prepare_image(image):

    image = image.convert("RGB")

    max_size = (1024, 1024)

    image.thumbnail(max_size)

    return image


def analyze_image(image, question):

    processor, model = load_model()

    image = prepare_image(image)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image"
                },
                {
                    "type": "text",
                    "text": question
                }
            ]
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True
    )

    inputs = processor(
        text=prompt,
        images=[image],
        return_tensors="pt"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False
    )

    input_length = inputs["input_ids"].shape[1]

    response_ids = generated_ids[:, input_length:]

    response = processor.batch_decode(
        response_ids,
        skip_special_tokens=True
    )[0]

    return response


# =========================================================
# HEADER
# =========================================================

st.title("🛰️ SatQuery AI")

st.write(
    "Interactive Vision-Language Assistant for "
    "Multimodal Remote Sensing Image Analysis"
)


# =========================================================
# PHASE 2
# SATELLITE IMAGE UPLOAD
# =========================================================

st.header("📡 Phase 2 — Satellite Image Upload")

uploaded_file = st.file_uploader(
    "Upload Satellite / Remote Sensing Image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "tif",
        "tiff"
    ]
)


if uploaded_file:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🛰️ Uploaded Image")

        st.image(
            image,
            use_container_width=True
        )

    with col2:

        st.subheader("📊 Image Information")

        st.write(
            "Filename:",
            uploaded_file.name
        )

        st.write(
            "Width:",
            image.width
        )

        st.write(
            "Height:",
            image.height
        )

        st.write(
            "Format:",
            image.format
        )


    # =====================================================
    # PHASE 3
    # TEXT QUERY
    # =====================================================

    st.header("💬 Phase 3 — Text Query")

    question_type = st.selectbox(
        "Select Question Type",

        [
            "Describe Image",
            "Identify Features",
            "Land Cover",
            "Vegetation",
            "Water Bodies",
            "Urban / Buildings",
            "Custom Question"
        ]
    )


    if question_type == "Describe Image":

        default_question = (
            "Describe this remote sensing image in detail."
        )

    elif question_type == "Identify Features":

        default_question = (
            "What geographical and human-made "
            "features are visible in this image?"
        )

    elif question_type == "Land Cover":

        default_question = (
            "What types of land cover are visible "
            "in this satellite image?"
        )

    elif question_type == "Vegetation":

        default_question = (
            "What vegetation features can you "
            "identify in this image?"
        )

    elif question_type == "Water Bodies":

        default_question = (
            "Are there any water bodies visible "
            "in this image? Describe them."
        )

    elif question_type == "Urban / Buildings":

        default_question = (
            "What buildings, roads, urban areas "
            "or infrastructure are visible?"
        )

    else:

        default_question = ""


    question = st.text_area(
        "📝 Your Question",

        value=default_question,

        height=100,

        placeholder=(
            "Ask anything about the satellite image..."
        )
    )


    # =====================================================
    # PHASE 4
    # VISION LANGUAGE MODEL
    # =====================================================

    st.header("🤖 Phase 4 — Vision-Language Analysis")


    if st.button(
        "🔍 Analyze Image",
        type="primary"
    ):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "AI is analyzing the satellite image..."
            ):

                try:

                    answer = analyze_image(
                        image,
                        question
                    )

                    st.success("Analysis Complete")

                    st.subheader(
                        "🤖 AI Answer"
                    )

                    st.write(answer)


                except Exception as e:

                    st.error(
                        f"AI analysis failed: {e}"
                    )


    # =====================================================
    # PHASE 5
    # DYNAMIC DASHBOARD
    # =====================================================

    st.header("📊 Phase 5 — Dynamic Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Image Width",
            f"{image.width}px"
        )

    with col2:

        st.metric(
            "Image Height",
            f"{image.height}px"
        )

    with col3:

        st.metric(
            "Image Format",
            str(image.format)
        )

    with col4:

        st.metric(
            "Question Length",
            len(question)
        )


# =========================================================
# FUTURE PHASES
# =========================================================

st.divider()

st.header("🚀 Advanced Modules")

st.info(
    """
    Phase 6  → RSVQA Dataset

    Phase 7  → Model Fine-Tuning

    Phase 8  → Object Detection

    Phase 9  → Change Detection

    Phase 10 → Interactive GIS Map
    """
)