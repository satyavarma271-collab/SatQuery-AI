import json
import random
import os
import html

import streamlit as st
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText
)


# =========================================================
# CONFIG
# =========================================================

MODEL_NAME = "HuggingFaceTB/SmolVLM-256M-Instruct"

DATASET_FILE = "datasets/rsvqa/training_data.json"

DATASET_IMAGE_DIR = "datasets/rsvqa/images/Images_LR"


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="SatQuery AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.html("""
<style>

.stApp {
    background:
        radial-gradient(
            circle at 15% 10%,
            rgba(37, 99, 235, 0.22),
            transparent 30%
        ),
        radial-gradient(
            circle at 85% 15%,
            rgba(124, 58, 237, 0.20),
            transparent 30%
        ),
        linear-gradient(
            135deg,
            #020617 0%,
            #07112b 45%,
            #020617 100%
        );

    color: #e5e7eb;
}

.block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}


/* =====================================================
   HERO
===================================================== */

.hero {
    padding: 35px 20px 30px;
    text-align: center;
}

.logo {
    font-size: 42px;
    font-weight: 800;
    letter-spacing: -1px;
}

.logo span {
    color: #60a5fa;
}

.subtitle {
    color: #94a3b8;
    font-size: 17px;
    margin-top: 8px;
}

.status {
    display: inline-block;
    margin-top: 18px;
    padding: 7px 16px;
    border-radius: 999px;
    background: rgba(34,197,94,0.12);
    border: 1px solid rgba(34,197,94,0.30);
    color: #86efac;
    font-size: 13px;
}


/* =====================================================
   CARDS
===================================================== */

.card {
    background: rgba(15, 23, 42, 0.72);
    border: 1px solid rgba(148,163,184,0.16);
    border-radius: 18px;
    padding: 22px;
    min-height: 120px;
    box-shadow: 0 15px 45px rgba(0,0,0,0.20);
}

.card-title {
    color: #94a3b8;
    font-size: 13px;
    margin-bottom: 8px;
}

.card-value {
    color: #f8fafc;
    font-size: 28px;
    font-weight: 750;
}

.card-small {
    color: #64748b;
    font-size: 12px;
    margin-top: 6px;
}


/* =====================================================
   FEATURES
===================================================== */

.feature {
    background: linear-gradient(
        145deg,
        rgba(15,23,42,0.88),
        rgba(30,41,59,0.55)
    );

    border: 1px solid rgba(96,165,250,0.16);
    border-radius: 18px;
    padding: 24px;
    height: 100%;
}

.feature-icon {
    font-size: 30px;
}

.feature-title {
    font-size: 18px;
    font-weight: 700;
    margin-top: 10px;
}

.feature-text {
    color: #94a3b8;
    font-size: 14px;
    line-height: 1.6;
    margin-top: 8px;
}


/* =====================================================
   SECTION
===================================================== */

.section-title {
    font-size: 24px;
    font-weight: 750;
    margin-top: 30px;
    margin-bottom: 16px;
}


/* =====================================================
   RESULT
===================================================== */

.result {
    background: rgba(30,41,59,0.70);
    border: 1px solid rgba(96,165,250,0.25);
    border-radius: 18px;
    padding: 24px;
    margin-top: 18px;
}

.result-label {
    color: #60a5fa;
    font-size: 13px;
    font-weight: 700;
}

.result-text {
    color: #f8fafc;
    font-size: 18px;
    line-height: 1.6;
    margin-top: 10px;
}


/* =====================================================
   COMPARISON
===================================================== */

.compare-card {
    background: rgba(15,23,42,0.75);
    border: 1px solid rgba(148,163,184,0.16);
    border-radius: 18px;
    padding: 22px;
    min-height: 150px;
}

.compare-title {
    color: #94a3b8;
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
}

.compare-answer {
    color: #f8fafc;
    font-size: 20px;
    font-weight: 700;
    margin-top: 12px;
    line-height: 1.5;
}


/* =====================================================
   UPLOADER
===================================================== */

[data-testid="stFileUploader"] {
    background: rgba(15,23,42,0.65);
    border: 1px dashed rgba(96,165,250,0.35);
    border-radius: 16px;
    padding: 10px;
}


/* =====================================================
   INPUT
===================================================== */

.stTextInput input {
    background: #0f172a !important;
    color: #f8fafc !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
}


/* =====================================================
   BUTTON
===================================================== */

.stButton button {
    border-radius: 12px;
    border: 1px solid rgba(96,165,250,0.35);
    background: linear-gradient(
        135deg,
        #2563eb,
        #4f46e5
    );
    color: white;
    font-weight: 700;
    padding: 10px 20px;
}

.stButton button:hover {
    border-color: #60a5fa;
}


/* =====================================================
   FOOTER
===================================================== */

.footer {
    text-align: center;
    color: #64748b;
    font-size: 12px;
    margin-top: 50px;
}

</style>
""")


# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_dataset():

    if not os.path.exists(DATASET_FILE):

        raise FileNotFoundError(
            f"Dataset file not found: {DATASET_FILE}"
        )

    with open(
        DATASET_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


dataset = load_dataset()


# =========================================================
# LOAD SMOLVLM
# =========================================================

@st.cache_resource
def load_model():

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME
    )

    model.eval()

    return processor, model


# =========================================================
# AI ANALYSIS FUNCTION
# =========================================================

def ask_smolvlm(image, question):

    processor, model = load_model()

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

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=30
        )

    answer = processor.decode(
        output[0][
            inputs["input_ids"].shape[-1]:
        ],
        skip_special_tokens=True
    )

    return answer.strip()


# =========================================================
# HERO
# =========================================================

st.html("""
<div class="hero">

    <div class="logo">
        🛰️ <span>SatQuery</span> AI
    </div>

    <div class="subtitle">
        Interactive Vision-Language Assistant
        for Multimodal Remote Sensing Image Analysis
    </div>

    <div class="status">
        ● AI SYSTEM ONLINE
    </div>

</div>
""")


# =========================================================
# METRICS
# =========================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.html("""
    <div class="card">

        <div class="card-title">
            DATASET SAMPLES
        </div>

        <div class="card-value">
            77,232
        </div>

        <div class="card-small">
            RSVQA question-answer pairs
        </div>

    </div>
    """)


with col2:

    st.html("""
    <div class="card">

        <div class="card-title">
            VISION MODEL
        </div>

        <div class="card-value">
            SmolVLM
        </div>

        <div class="card-small">
            256M parameter model
        </div>

    </div>
    """)


with col3:

    st.html("""
    <div class="card">

        <div class="card-title">
            IMAGE SUPPORT
        </div>

        <div class="card-value">
            RGB + TIF
        </div>

        <div class="card-small">
            Remote sensing imagery
        </div>

    </div>
    """)


with col4:

    st.html("""
    <div class="card">

        <div class="card-title">
            ANALYSIS MODE
        </div>

        <div class="card-value">
            VQA
        </div>

        <div class="card-small">
            Question-based analysis
        </div>

    </div>
    """)


# =========================================================
# CORE CAPABILITIES
# =========================================================

st.html("""
<div class="section-title">
    ✨ Core Capabilities
</div>
""")


f1, f2, f3 = st.columns(3)


with f1:

    st.html("""
    <div class="feature">

        <div class="feature-icon">
            🔍
        </div>

        <div class="feature-title">
            Ask Visual Questions
        </div>

        <div class="feature-text">
            Upload a remote sensing image and ask
            natural-language questions about its content.
        </div>

    </div>
    """)


with f2:

    st.html("""
    <div class="feature">

        <div class="feature-icon">
            🧠
        </div>

        <div class="feature-title">
            Vision-Language AI
        </div>

        <div class="feature-text">
            SmolVLM combines visual information
            with natural-language queries.
        </div>

    </div>
    """)


with f3:

    st.html("""
    <div class="feature">

        <div class="feature-icon">
            🛰️
        </div>

        <div class="feature-title">
            Remote Sensing Analysis
        </div>

        <div class="feature-text">
            Designed for satellite and
            remote-sensing image understanding.
        </div>

    </div>
    """)


# =========================================================
# IMAGE ANALYSIS WORKSPACE
# =========================================================

st.html("""
<div class="section-title">
    🛰️ Image Analysis Workspace
</div>
""")


uploaded_file = st.file_uploader(
    "Upload Satellite / Remote Sensing Image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "tif",
        "tiff",
        "webp"
    ]
)


if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    image_col, query_col = st.columns(
        [1.25, 1]
    )


    # =====================================================
    # IMAGE
    # =====================================================

    with image_col:

        st.image(
            image,
            caption="Uploaded Remote Sensing Image",
            use_container_width=True
        )


    # =====================================================
    # QUESTION AREA
    # =====================================================

    with query_col:

        st.markdown(
            "### 💬 Ask SatQuery AI"
        )

        st.markdown(
            "**Quick Questions**"
        )

        q1, q2 = st.columns(2)


        with q1:

            if st.button(
                "🏙️ Rural or Urban?",
                use_container_width=True
            ):

                st.session_state["question"] = (
                    "Is it a rural or an urban area?"
                )


            if st.button(
                "🏢 Are there buildings?",
                use_container_width=True
            ):

                st.session_state["question"] = (
                    "Are there buildings in this image?"
                )


        with q2:

            if st.button(
                "🌊 Is there water?",
                use_container_width=True
            ):

                st.session_state["question"] = (
                    "Is there a water area in this image?"
                )


            if st.button(
                "🌳 Is there vegetation?",
                use_container_width=True
            ):

                st.session_state["question"] = (
                    "Is there vegetation in this image?"
                )


        question = st.text_input(
            "Your question",
            value=st.session_state.get(
                "question",
                ""
            ),
            placeholder=(
                "Example: Is this a rural or urban area?"
            ),
            key="question_input"
        )


        analyze = st.button(
            "🔍 Analyze Image",
            use_container_width=True
        )


        # =================================================
        # AI ANALYSIS
        # =================================================

        if analyze:

            if question.strip() == "":

                st.warning(
                    "Please enter a question."
                )

            else:

                try:

                    with st.spinner(
                        "SatQuery AI is analyzing the image..."
                    ):

                        answer = ask_smolvlm(
                            image,
                            question
                        )


                    safe_answer = html.escape(
                        answer
                    )

                    st.html(
                        f"""
                        <div class="result">

                            <div class="result-label">
                                🤖 AI ANALYSIS
                            </div>

                            <div class="result-text">
                                {safe_answer}
                            </div>

                        </div>
                        """
                    )


                except Exception as error:

                    st.error(
                        f"AI analysis failed: {error}"
                    )


# =========================================================
# RSVQA DATASET EXPLORER
# =========================================================

st.html("""
<div class="section-title">
    📚 RSVQA Dataset Explorer
</div>
""")


st.write(
    f"Available samples: **{len(dataset):,}**"
)


if st.button(
    "🎲 Load Random RSVQA Sample"
):

    sample = random.choice(
        dataset
    )

    st.session_state[
        "sample_question"
    ] = sample["question"]

    st.session_state[
        "sample_answer"
    ] = sample["answer"]

    st.session_state[
        "sample_image"
    ] = sample["image"]

    st.session_state.pop(
        "sample_ai_answer",
        None
    )


# =========================================================
# RSVQA SAMPLE
# =========================================================

if "sample_question" in st.session_state:

    sample_col1, sample_col2 = st.columns(
        [1, 1]
    )


    # =====================================================
    # DATASET IMAGE
    # =====================================================

    with sample_col1:

        image_reference = st.session_state[
            "sample_image"
        ]

        image_filename = os.path.basename(
            image_reference
        )

        image_path = os.path.join(
            DATASET_IMAGE_DIR,
            image_filename
        )


        try:

            if not os.path.exists(image_path):

                st.error(
                    "Dataset image not found:"
                )

                st.code(
                    image_path
                )

            else:

                sample_image = Image.open(
                    image_path
                ).convert("RGB")


                st.image(
                    sample_image,
                    caption="RSVQA Dataset Image",
                    use_container_width=True
                )


        except Exception as error:

            st.error(
                f"Could not load dataset image: {error}"
            )


    # =====================================================
    # QUESTION + GROUND TRUTH
    # =====================================================

    with sample_col2:

        st.markdown(
            "### 💬 Dataset Question"
        )

        st.info(
            st.session_state[
                "sample_question"
            ]
        )


        st.markdown(
            "### 📚 Ground Truth Answer"
        )

        st.success(
            st.session_state[
                "sample_answer"
            ]
        )


        st.markdown(
            "### 📊 Dataset Information"
        )

        st.write(
            "Source: RSVQA-LR"
        )

        st.write(
            "Purpose: Remote Sensing Visual Question Answering"
        )


    # =====================================================
    # AI ANALYSIS
    # =====================================================

    st.markdown("---")

    st.markdown(
        "### 🤖 Ask SmolVLM About This Dataset Sample"
    )


    analyze_sample = st.button(
        "🧠 Analyze RSVQA Sample with AI",
        use_container_width=True
    )


    if analyze_sample:

        image_reference = st.session_state[
            "sample_image"
        ]

        image_filename = os.path.basename(
            image_reference
        )

        image_path = os.path.join(
            DATASET_IMAGE_DIR,
            image_filename
        )


        if not os.path.exists(image_path):

            st.error(
                "Dataset image could not be found."
            )

        else:

            try:

                sample_image = Image.open(
                    image_path
                ).convert("RGB")


                sample_question = (
                    st.session_state[
                        "sample_question"
                    ]
                )


                with st.spinner(
                    "SmolVLM is analyzing the RSVQA sample..."
                ):

                    ai_answer = ask_smolvlm(
                        sample_image,
                        sample_question
                    )


                st.session_state[
                    "sample_ai_answer"
                ] = ai_answer


            except Exception as error:

                st.error(
                    f"AI analysis failed: {error}"
                )


    # =====================================================
    # AI VS GROUND TRUTH
    # =====================================================

    if "sample_ai_answer" in st.session_state:

        st.markdown(
            "### 📊 AI vs Ground Truth"
        )


        ground_truth = str(
            st.session_state[
                "sample_answer"
            ]
        ).strip().lower()


        ai_answer = str(
            st.session_state[
                "sample_ai_answer"
            ]
        ).strip().lower()


        compare1, compare2 = st.columns(2)


        # =================================================
        # GROUND TRUTH
        # =================================================

        with compare1:

            safe_ground_truth = html.escape(
                str(
                    st.session_state[
                        "sample_answer"
                    ]
                )
            )

            st.html(
                f"""
                <div class="compare-card">

                    <div class="compare-title">
                        📚 Ground Truth
                    </div>

                    <div class="compare-answer">
                        {safe_ground_truth}
                    </div>

                </div>
                """
            )


        # =================================================
        # AI ANSWER
        # =================================================

        with compare2:

            safe_ai_answer = html.escape(
                str(
                    st.session_state[
                        "sample_ai_answer"
                    ]
                )
            )

            st.html(
                f"""
                <div class="compare-card">

                    <div class="compare-title">
                        🤖 SmolVLM Answer
                    </div>

                    <div class="compare-answer">
                        {safe_ai_answer}
                    </div>

                </div>
                """
            )


        # =================================================
        # MATCH / DIFFERENT
        # =================================================

        st.markdown(
            "### 🔎 Evaluation Result"
        )


        if ground_truth == ai_answer:

            st.success(
                "✅ MATCH — AI answer matches the Ground Truth."
            )

            st.metric(
                "Answer Match",
                "MATCH"
            )

        else:

            st.warning(
                "⚠️ DIFFERENT — AI answer does not exactly "
                "match the Ground Truth."
            )

            st.metric(
                "Answer Match",
                "DIFFERENT"
            )


        st.info(
            "This is an exact text comparison between "
            "the generated answer and the RSVQA ground-truth answer."
        )


# =========================================================
# FOOTER
# =========================================================

st.html("""
<div class="footer">

    SatQuery AI
    •
    Remote Sensing Intelligence Platform

</div>
""")