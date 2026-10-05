import os
import sqlite3
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from google import genai


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NeuroPulse",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "cnn_attention_model.keras"
)

SCALER_FILE = os.path.join(
    BASE_DIR,
    "scaler.pkl"
)

ENCODER_FILE = os.path.join(
    BASE_DIR,
    "label_encoder.pkl"
)

DB_FILE = os.path.join(
    BASE_DIR,
    "neuropulse_history.db"
)

HISTORY_CSV_FILE = os.path.join(
    BASE_DIR,
    "neuropulse_history.csv"
)


# ============================================================
# EEG CONFIGURATION
# ============================================================

CHANNELS = [
    "TP9",
    "AF7",
    "AF8",
    "TP10",
]

BANDS = [
    "delta",
    "theta",
    "alpha",
    "beta",
    "gamma",
]

EMOTIONS = [
    ("😡", "ANGRY"),
    ("😌", "CALM"),
    ("😨", "FEAR"),
    ("😊", "HAPPY"),
    ("😢", "SAD"),
]

EEG_FEATURES = [
    "TP9_delta",
    "TP9_theta",
    "TP9_alpha",
    "TP9_beta",
    "TP9_gamma",
    "TP9_mean",
    "TP9_std",

    "AF7_delta",
    "AF7_theta",
    "AF7_alpha",
    "AF7_beta",
    "AF7_gamma",
    "AF7_mean",
    "AF7_std",

    "AF8_delta",
    "AF8_theta",
    "AF8_alpha",
    "AF8_beta",
    "AF8_gamma",
    "AF8_mean",
    "AF8_std",

    "TP10_delta",
    "TP10_theta",
    "TP10_alpha",
    "TP10_beta",
    "TP10_gamma",
    "TP10_mean",
    "TP10_std",
]


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {

    "page": "Home",

    "dataset": None,
    "dataset_name": "",

    "selected_row": 0,
    "eeg_row_number": 1,

    "last_prediction": None,
    "last_confidence": None,
    "last_probabilities": None,
    "last_analysis_type": None,

    "dataset_prediction": None,
    "dataset_prediction_share": None,
    "dataset_average_confidence": None,
    "dataset_probabilities": None,
    "dataset_valid_rows": None,
    "dataset_total_rows": None,
    "dataset_emotion_counts": None,

    "wellness_chat_history": [],
    "gemini_interaction_id": None,
    "wellness_recommendation_key": None,
}


for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def html(
    content,
    unsafe_allow_html=True
):

    cleaned = "\n".join(

        line.strip()

        for line in content.splitlines()

        if line.strip()

    )

    st.markdown(
        cleaned,
        unsafe_allow_html=unsafe_allow_html
    )


def page_header(
    title,
    subtitle
):

    st.title(title)

    st.caption(subtitle)

    st.divider()


def emotion_icon(
    emotion
):

    return {

        "ANGRY": "😡",

        "CALM": "😌",

        "FEAR": "😨",

        "HAPPY": "😊",

        "SAD": "😢",

    }.get(

        str(emotion).upper(),

        "🧠"

    )


def wellness_message(
    emotion
):

    messages = {

        "CALM":
        "The model detected a CALM pattern. Continue activities that help you maintain a comfortable state.",

        "HAPPY":
        "The model detected a HAPPY pattern. Continue positive and enjoyable activities.",

        "SAD":
        "The model detected a SAD pattern. Consider taking a short break or doing a calming activity.",

        "ANGRY":
        "The model detected an ANGRY pattern. Consider pausing, breathing slowly, and stepping away from a stressful activity.",

        "FEAR":
        "The model detected a FEAR pattern. Consider slowing down, taking controlled breaths, and moving to a comfortable environment.",

    }

    return messages.get(

        str(emotion).upper(),

        "Take a moment to check in with yourself."

    )


def reset_wellness_chat():

    st.session_state.wellness_chat_history = []

    st.session_state.gemini_interaction_id = None

    st.session_state.wellness_recommendation_key = None


def clear_prediction_results():

    st.session_state.last_prediction = None

    st.session_state.last_confidence = None

    st.session_state.last_probabilities = None

    st.session_state.last_analysis_type = None

    st.session_state.dataset_prediction = None

    st.session_state.dataset_prediction_share = None

    st.session_state.dataset_average_confidence = None

    st.session_state.dataset_probabilities = None

    st.session_state.dataset_valid_rows = None

    st.session_state.dataset_total_rows = None

    st.session_state.dataset_emotion_counts = None

    reset_wellness_chat()


def compact_ai_text(
    text,
    max_sentences=3,
    max_chars=520
):

    if not text:

        return (
            "I’m here to help with a quick wellness check-in."
        )

    text = str(text).strip()

    sentences = []

    current_sentence = ""

    for character in text:

        current_sentence += character

        if character in ".!?":

            sentence = current_sentence.strip()

            if sentence:

                sentences.append(
                    sentence
                )

            current_sentence = ""

            if len(sentences) >= max_sentences:

                break

    if (
        current_sentence.strip()
        and
        len(sentences) < max_sentences
    ):

        sentences.append(
            current_sentence.strip()
        )

    result = " ".join(
        sentences
    ).strip()

    if len(result) > max_chars:

        result = (
            result[:max_chars]
            .rsplit(" ", 1)[0]
            .rstrip()
        )

        if not result.endswith(
            (".", "!", "?")
        ):

            result += "."

    return result


# ============================================================
# GEMINI CLIENT
# ============================================================

@st.cache_resource
def get_gemini_client():

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:

        return None

    return genai.Client(
        api_key=api_key
    )


gemini_client = get_gemini_client()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_neuropulse_model():

    loaded_model = tf.keras.models.load_model(
        MODEL_FILE,
        compile=False
    )

    loaded_scaler = joblib.load(
        SCALER_FILE
    )

    loaded_encoder = joblib.load(
        ENCODER_FILE
    )

    return (
        loaded_model,
        loaded_scaler,
        loaded_encoder
    )


model = None

scaler = None

label_encoder = None

model_error = None


try:

    model, scaler, label_encoder = (
        load_neuropulse_model()
    )

except Exception as exc:

    model_error = str(exc)


# ============================================================
# DATABASE
# ============================================================

def initialize_database():

    conn = sqlite3.connect(
        DB_FILE
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            row_number INTEGER,
            emotion TEXT,
            confidence REAL
        )
        """
    )

    conn.commit()

    conn.close()


def save_prediction(
    row_number,
    emotion,
    confidence
):

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    conn = sqlite3.connect(
        DB_FILE
    )

    conn.execute(
        """
        INSERT INTO predictions
        (
            timestamp,
            row_number,
            emotion,
            confidence
        )
        VALUES (?, ?, ?, ?)
        """,
        (
            timestamp,
            int(row_number),
            str(emotion),
            float(confidence)
        )
    )

    conn.commit()

    conn.close()


    record = pd.DataFrame(
        [
            {
                "timestamp": timestamp,
                "row_number": int(row_number),
                "emotion": str(emotion),
                "confidence": float(confidence)
            }
        ]
    )


    record.to_csv(
        HISTORY_CSV_FILE,
        mode="a",
        header=not os.path.exists(
            HISTORY_CSV_FILE
        ),
        index=False
    )


def get_history():

    conn = sqlite3.connect(
        DB_FILE
    )

    history = pd.read_sql_query(
        """
        SELECT *
        FROM predictions
        ORDER BY id DESC
        """,
        conn
    )

    conn.close()

    return history


initialize_database()


# ============================================================
# EEG VALIDATION
# ============================================================

def validate_eeg_features(
    dataframe
):

    return [

        feature

        for feature in EEG_FEATURES

        if feature not in dataframe.columns

    ]


def prepare_eeg_data(
    dataframe
):

    missing_features = (
        validate_eeg_features(
            dataframe
        )
    )


    if missing_features:

        raise ValueError(

            "Missing EEG features:\n"
            +
            ", ".join(
                missing_features
            )

        )


    feature_data = (

        dataframe[
            EEG_FEATURES
        ]

        .copy()

        .apply(
            pd.to_numeric,
            errors="coerce"
        )

    )


    values = feature_data.to_numpy(
        dtype=np.float32
    )


    valid_mask = np.isfinite(
        values
    ).all(
        axis=1
    )


    return (
        values,
        valid_mask
    )


# ============================================================
# SINGLE EEG ROW PREDICTION
# ============================================================

def predict_emotion(
    dataframe,
    row_number
):

    if model is None:

        raise RuntimeError(
            "CNN + Attention model could not be loaded."
        )


    if scaler is None:

        raise RuntimeError(
            "Scaler could not be loaded."
        )


    if label_encoder is None:

        raise RuntimeError(
            "Label encoder could not be loaded."
        )


    if (

        row_number < 0

        or

        row_number >= len(dataframe)

    ):

        raise ValueError(
            "Invalid EEG row selected."
        )


    values, valid_mask = (
        prepare_eeg_data(
            dataframe
        )
    )


    if not valid_mask[row_number]:

        raise ValueError(

            "The selected EEG row contains "
            "missing or invalid values."

        )


    X = values[
        row_number
    ].reshape(
        1,
        28
    )


    X_scaled = scaler.transform(
        X
    )


    X_cnn = X_scaled.reshape(
        1,
        4,
        7
    ).astype(
        np.float32
    )


    probabilities = np.asarray(

        model.predict(
            X_cnn,
            verbose=0
        )[0],

        dtype=np.float32

    ).flatten()


    if len(probabilities) != len(
        label_encoder.classes_
    ):

        raise RuntimeError(

            "Model output classes do not match "
            "the label encoder."

        )


    prediction_index = int(

        np.argmax(
            probabilities
        )

    )


    emotion = str(

        label_encoder.inverse_transform(
            [
                prediction_index
            ]
        )[0]

    )


    confidence = float(

        probabilities[
            prediction_index
        ]
        *
        100

    )


    return (

        emotion,
        confidence,
        probabilities

    )


# ============================================================
# ENTIRE DATASET PREDICTION
# ============================================================

def predict_entire_dataset(
    dataframe
):

    if model is None:

        raise RuntimeError(
            "CNN + Attention model could not be loaded."
        )


    if scaler is None:

        raise RuntimeError(
            "Scaler could not be loaded."
        )


    if label_encoder is None:

        raise RuntimeError(
            "Label encoder could not be loaded."
        )


    if dataframe.empty:

        raise ValueError(
            "The uploaded dataset is empty."
        )


    values, valid_mask = (
        prepare_eeg_data(
            dataframe
        )
    )


    total_rows = len(
        values
    )


    valid_values = values[
        valid_mask
    ]


    invalid_rows = (

        total_rows
        -
        len(valid_values)

    )


    if len(valid_values) == 0:

        raise ValueError(

            "No valid EEG rows were found."

        )


    X_scaled = scaler.transform(
        valid_values
    )


    X_cnn = X_scaled.reshape(
        -1,
        4,
        7
    ).astype(
        np.float32
    )


    probabilities = np.asarray(

        model.predict(
            X_cnn,
            verbose=0
        ),

        dtype=np.float32

    )


    if probabilities.ndim != 2:

        raise RuntimeError(

            "Unexpected model output shape."

        )


    if probabilities.shape[1] != len(
        label_encoder.classes_
    ):

        raise RuntimeError(

            "Model output classes do not match "
            "the label encoder."

        )


    prediction_indices = np.argmax(

        probabilities,

        axis=1

    )


    predicted_emotions = np.asarray(

        label_encoder.inverse_transform(
            prediction_indices
        )

    )


    emotion_counts = (

        pd.Series(
            predicted_emotions
        )

        .value_counts()

    )


    overall_emotion = str(

        emotion_counts.index[0]

    )


    overall_count = int(

        emotion_counts.iloc[0]

    )


    overall_share = (

        overall_count
        /
        len(valid_values)
        *
        100

    )


    row_confidences = (

        np.max(
            probabilities,
            axis=1
        )
        *
        100

    )


    average_confidence = float(

        np.mean(
            row_confidences
        )

    )


    average_probabilities = np.mean(

        probabilities,

        axis=0

    )


    distribution = (

        emotion_counts

        .reindex(

            label_encoder.classes_,

            fill_value=0

        )

    )


    return {

        "overall_emotion":
            overall_emotion,

        "overall_share":
            float(
                overall_share
            ),

        "average_confidence":
            average_confidence,

        "average_probabilities":
            average_probabilities,

        "emotion_counts":
            distribution,

        "valid_rows":
            int(
                len(valid_values)
            ),

        "invalid_rows":
            int(
                invalid_rows
            ),

        "total_rows":
            int(
                total_rows
            ),

    }


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       MAIN APP
       ======================================================== */

    .stApp {

        background:
            radial-gradient(
                circle at 10% 0%,
                #122c6f 0%,
                #071a35 28%,
                #03111f 67%,
                #020b14 100%
            );

        color:
            #ffffff;

    }


    [data-testid="stHeader"] {

        background:
            transparent;

    }


    [data-testid="stToolbar"] {

        background:
            transparent;

    }


    .block-container {

        max-width:
            1240px;

        padding-top:
            2.2rem;

        padding-bottom:
            3rem;

    }


    h1,
    h2,
    h3,
    h4,
    h5 {

        color:
            #ffffff !important;

        font-weight:
            750 !important;

    }


    p,
    label {

        color:
            #d6e7f6 !important;

    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {

        background:
            linear-gradient(
                180deg,
                #06172c 0%,
                #03101f 100%
            );

        border-right:
            1px solid #163d63;

    }


    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button {

        background:
            linear-gradient(
                135deg,
                #123657,
                #0b2440
            );

        color:
            #ffffff;

        border:
            1px solid #28618b;

        border-radius:
            11px;

        min-height:
            44px;

        font-weight:
            650;

        box-shadow:
            0 5px 18px rgba(0,0,0,.16);

        transition:
            all .2s ease;

    }


    .stButton > button:hover {

        background:
            linear-gradient(
                135deg,
                #16496f,
                #0d2f4f
            );

        border-color:
            #58dbff;

        color:
            #58dbff;

        transform:
            translateY(-1px);

    }


    /* ========================================================
       METRICS
       ======================================================== */

    [data-testid="stMetric"] {

        background:
            linear-gradient(
                145deg,
                #0c2844,
                #071b30
            );

        border:
            1px solid #26587d;

        border-radius:
            15px;

        padding:
            14px 16px;

        min-height:
            105px;

    }


    [data-testid="stMetricLabel"] {

        color:
            #8eb0ca !important;

    }


    [data-testid="stMetricValue"] {

        color:
            #58dbff !important;

    }


    /* ========================================================
       HERO
       ======================================================== */

    .hero-title {

        font-size:
            42px;

        line-height:
            1.05;

        font-weight:
            850;

        color:
            #ffffff;

    }


    .hero-gradient {

        background:
            linear-gradient(
                90deg,
                #ffffff,
                #52d9ff 45%,
                #a27cff
            );

        -webkit-background-clip:
            text;

        -webkit-text-fill-color:
            transparent;

    }


    .hero-subtitle {

        color:
            #91b3cd;

        font-size:
            14px;

        line-height:
            1.75;

    }


    .small-label {

        color:
            #55baff;

        font-size:
            9px;

        font-weight:
            800;

        letter-spacing:
            1.4px;

        text-transform:
            uppercase;

    }


    /* ========================================================
       BRAIN VISUAL
       ======================================================== */

    .brain-stage {

        position:
            relative;

        height:
            175px;

        overflow:
            hidden;

        border-radius:
            22px;

        background:
            radial-gradient(
                circle at center,
                rgba(64,155,255,.17) 0%,
                rgba(64,155,255,.05) 30%,
                transparent 68%
            );

    }


    .brain-glow {

        position:
            absolute;

        left:
            50%;

        top:
            50%;

        transform:
            translate(-50%,-50%);

        width:
            125px;

        height:
            125px;

        border-radius:
            50%;

        display:
            flex;

        align-items:
            center;

        justify-content:
            center;

        font-size:
            76px;

        background:
            radial-gradient(
                circle,
                rgba(99,103,255,.35),
                rgba(16,91,180,.10) 48%,
                transparent 75%
            );

        box-shadow:
            0 0 45px rgba(73,141,255,.55),
            0 0 90px rgba(83,54,255,.22);

    }


    .pulse-left,
    .pulse-right {

        position:
            absolute;

        top:
            83px;

        width:
            44%;

        height:
            40px;

        opacity:
            .9;

    }


    .pulse-left {

        left:
            0;

    }


    .pulse-right {

        right:
            0;

    }


    .pulse-wave {

        width:
            100%;

        height:
            100%;

        border-top:
            2px solid #39c7ff;

        border-radius:
            50%;

        transform:
            skewX(-28deg);

        filter:
            drop-shadow(0 0 5px #39c7ff);

    }


    /* ========================================================
       CARDS
       ======================================================== */

    .premium-card {

        background:
            linear-gradient(
                145deg,
                rgba(14,42,69,.97),
                rgba(5,20,35,.97)
            );

        border:
            1px solid #275a7f;

        border-radius:
            18px;

        padding:
            20px;

        box-shadow:
            0 14px 38px rgba(0,0,0,.16);

    }


    /* ========================================================
       POPOVERS
       ======================================================== */

    div[data-testid="stPopover"] > button {

        width:
            100% !important;

        min-height:
            102px !important;

        padding:
            14px !important;

        text-align:
            left !important;

        white-space:
            pre-line !important;

        background:
            linear-gradient(
                145deg,
                #0d2945,
                #081c30
            ) !important;

        border:
            1px solid #285a7e !important;

        border-radius:
            15px !important;

        color:
            #ffffff !important;

        font-weight:
            700 !important;

        box-shadow:
            0 9px 25px rgba(0,0,0,.13);

    }


    div[data-testid="stPopover"] > button:hover {

        background:
            linear-gradient(
                145deg,
                #123d60,
                #0b233c
            ) !important;

        border-color:
            #52d7ff !important;

        color:
            #58dbff !important;

    }


    /* ========================================================
       EXPANDERS
       ======================================================== */

    div[data-testid="stExpander"] {

        background:
            #071c30;

        border:
            1px solid #295a7d;

        border-radius:
            15px;

    }


    /* ========================================================
       FILE UPLOADER
       ======================================================== */

    [data-testid="stFileUploader"] {

        background:
            #081d31;

        border:
            1px dashed #3e739c;

        border-radius:
            15px;

        padding:
            12px;

    }


    /* ========================================================
       CHAT
       ======================================================== */

    div[data-testid="stChatMessage"] {

        background:
            #0a2239 !important;

        border:
            1px solid #29577a !important;

        border-radius:
            14px !important;

        margin-bottom:
            8px !important;

    }


    div[data-testid="stChatMessage"] p {

        color:
            #deedfa !important;

        font-size:
            13px !important;

        line-height:
            1.5 !important;

    }


    div[data-testid="stChatInput"] {

        border:
            1px solid #285a7e !important;

        background:
            #071d31 !important;

        border-radius:
            13px !important;

    }


    /* ========================================================
       PROGRESS
       ======================================================== */

    .stProgress > div > div > div > div {

        background:
            #35b8ff;

    }


    /* ========================================================
       DATAFRAME
       ======================================================== */

    [data-testid="stDataFrame"] {

        border-radius:
            12px;

        overflow:
            hidden;

        border:
            1px solid #204866;

    }


    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    html(
        """
        <div style="
            padding:7px 4px 20px 4px;
        ">

            <div style="
                font-size:39px;
                line-height:1;
            ">
                🧠
            </div>

            <div style="
                font-size:27px;
                font-weight:850;
                margin-top:8px;
                color:#ffffff;
            ">
                Neuro<span style="color:#6674ff;">
                    Pulse
                </span>
            </div>

            <div style="
                color:#7fa4bf;
                font-size:10px;
                line-height:1.7;
                margin-top:6px;
            ">
                EEG Emotion Detection &<br>
                Mental Wellness Assistant
            </div>

        </div>
        """
    )


    navigation = {

        "🏠  Home":
            "Home",

        "📤  Upload Dataset":
            "Upload Dataset",

        "😊  Emotion Analysis":
            "Emotion Analysis",

        "〽️  EEG Visualization":
            "EEG Visualization",

        "🕐  History":
            "History",

        "⚙️  Settings":
            "Settings",

    }


    labels = list(
        navigation.keys()
    )


    current_label = next(

        (

            label

            for label, value
            in navigation.items()

            if value == st.session_state.page

        ),

        labels[0]

    )


    selected_label = st.radio(

        "Navigation",

        labels,

        index=
        labels.index(
            current_label
        ),

        label_visibility=
        "collapsed"

    )


    st.session_state.page = (

        navigation[
            selected_label
        ]

    )


    page = st.session_state.page


    html(
        """
        <div style="
            margin-top:30px;
            padding:16px 4px;
            text-align:center;
            border-top:1px solid #173b59;
            color:#6687a4;
            font-size:9px;
            line-height:1.9;
        ">

            CNN + Attention<br>

            Mental Wellness Support<br><br>

            4 Channels × 7 Features

        </div>
        """
    )


# ============================================================
# WELLNESS ASSISTANT
# ============================================================

def wellness_assistant_panel():

    emotion = (
        st.session_state.last_prediction
    )

    confidence = (
        st.session_state.last_confidence
    )

    analysis_type = (

        st.session_state.last_analysis_type

        or

        "EEG Analysis"

    )


    if emotion is None:

        st.info(

            "Run an EEG analysis to start "
            "your wellness conversation."

        )

        return


    confidence_text = (

        "Not Available"

        if confidence is None

        else
        f"{float(confidence):.1f}%"

    )


    current_key = (

        f"{analysis_type}|"
        f"{emotion}|"
        f"{confidence_text}"

    )


    # ========================================================
    # SYSTEM INSTRUCTION
    # ========================================================

    system_instruction = """

You are the NeuroPulse AI Wellness Assistant.

Give supportive, friendly and VERY concise wellness guidance.

Normal answers must be no more than 3 short sentences.

Use simple language.

The EEG classification is experimental and is NOT a medical diagnosis.

Never diagnose anxiety, depression, stress disorders, or any other
medical or psychological condition.

Never claim that an emotion prediction is certain.

You can discuss relaxation, breathing, sleep, study stress,
mindfulness, healthy routines, emotions and general wellbeing.

If the user describes immediate danger or self-harm, encourage them
to contact a trusted person and local emergency services immediately.

"""


    # ========================================================
    # AUTOMATIC FIRST MESSAGE
    # ========================================================

    if (

        gemini_client is not None

        and

        st.session_state.wellness_recommendation_key
        !=
        current_key

    ):

        recommendation_prompt = f"""

NeuroPulse just completed an EEG emotion classification.

Detected emotion:
{emotion}

Confidence:
{confidence_text}

Analysis type:
{analysis_type}

Write the first chatbot message.

Keep it to 2 or 3 short sentences.

Briefly acknowledge the detected result.

State that this is an experimental EEG result
and not a medical diagnosis.

Give only one or two practical wellness suggestions.

Do not diagnose anything.

Do not say the emotion is certain.

"""


        try:

            with st.spinner(

                "🤖 Preparing wellness support..."

            ):

                interaction = (

                    gemini_client
                    .interactions
                    .create(

                        model=
                        "gemini-3.5-flash-lite",

                        input=
                        recommendation_prompt,

                        system_instruction=
                        system_instruction,

                    )

                )


            first_message = compact_ai_text(

                interaction.output_text,

                max_sentences=3,

                max_chars=520

            )


            st.session_state.wellness_chat_history = [

                {

                    "role":
                    "assistant",

                    "content":
                    first_message

                }

            ]


            st.session_state.gemini_interaction_id = (

                interaction.id

            )


            st.session_state.wellness_recommendation_key = (

                current_key

            )


        except Exception:

            st.session_state.wellness_chat_history = [

                {

                    "role":
                    "assistant",

                    "content":
                    (

                        f"The EEG result shows a "
                        f"**{emotion}** pattern with "
                        f"**{confidence_text}** confidence. "
                        "This is an experimental result, "
                        "not a medical diagnosis."

                    )

                }

            ]


            st.session_state.wellness_recommendation_key = (

                current_key

            )


    # ========================================================
    # HEADER
    # ========================================================

    html(

        f"""

        <div class="premium-card">

            <div class="small-label">
                AI WELLNESS ASSISTANT
            </div>

            <div style="
                color:#ffffff;
                font-size:21px;
                font-weight:800;
                margin-top:7px;
            ">

                ✨ Your AI Wellness Assistant

            </div>

            <div style="
                color:#8eafc8;
                font-size:11px;
                line-height:1.6;
                margin-top:6px;
            ">

                Personalized wellness support based on your EEG result.

            </div>

            <div style="
                margin-top:7px;
                color:#8fb1cc;
                font-size:11px;
            ">

                Current result:

                <b style="color:#ffffff;">

                    {emotion_icon(emotion)}
                    {emotion}

                </b>

                &nbsp; • &nbsp;

                Confidence:

                <b style="color:#58dbff;">

                    {confidence_text}

                </b>

            </div>

        </div>

        """

    )


    # ========================================================
    # CHAT HISTORY
    # ========================================================

    chat_container = st.container(

        height=285,

        border=True

    )


    with chat_container:

        if not st.session_state.wellness_chat_history:

            st.caption(

                "Your AI recommendation will appear here."

            )

        else:

            for message in (

                st.session_state.wellness_chat_history

            ):

                with st.chat_message(

                    message["role"]

                ):

                    st.markdown(

                        message["content"]

                    )


    # ========================================================
    # CHAT INPUT
    # ========================================================

    user_message = st.chat_input(

        "Ask your wellness question...",

        key=
        "neuropulse_chat_input"

    )


    if user_message:

        user_message = user_message.strip()


        if not user_message:

            return


        st.session_state.wellness_chat_history.append(

            {

                "role":
                "user",

                "content":
                user_message

            }

        )


        response_text = (

            "Gemini AI is not connected right now."

        )


        if gemini_client is not None:

            context = f"""

Current NeuroPulse context:

Detected EEG emotion:
{emotion}

Confidence:
{confidence_text}

Analysis type:
{analysis_type}

User message:
{user_message}

Respond in no more than 3 short sentences.

Answer the user's question directly.

Use simple, friendly language.

Give at most 1 or 2 practical suggestions.

Do not provide a long explanation unless the user asks.

Do not diagnose any medical or psychological condition.

The EEG classification is experimental and not a medical diagnosis.

"""


            request_data = {

                "model":
                "gemini-3.5-flash-lite",

                "input":
                context,

                "system_instruction":
                system_instruction,

            }


            previous_id = (

                st.session_state
                .gemini_interaction_id

            )


            if previous_id is not None:

                request_data[
                    "previous_interaction_id"
                ] = previous_id


            try:

                with st.spinner(

                    "🤖 NeuroPulse AI is thinking..."

                ):

                    interaction = (

                        gemini_client
                        .interactions
                        .create(

                            **request_data

                        )

                    )


                response_text = compact_ai_text(

                    interaction.output_text,

                    max_sentences=3,

                    max_chars=520

                )


                st.session_state.gemini_interaction_id = (

                    interaction.id

                )


            except Exception as exc:

                response_text = (

                    "I couldn't connect to Gemini right now. "
                    f"Error: {exc}"

                )


        st.session_state.wellness_chat_history.append(

            {

                "role":
                "assistant",

                "content":
                response_text

            }

        )


        st.rerun()


    # ========================================================
    # CLEAR CHAT
    # ========================================================

    if st.button(

        "🗑️ Clear Conversation",

        key=
        "wellness_clear_button",

        use_container_width=True

    ):

        st.session_state.wellness_chat_history = []

        st.session_state.gemini_interaction_id = None

        st.rerun()


# ============================================================
# HOME PAGE
# ============================================================

def home_page():

    # ========================================================
    # HEADER
    # ========================================================

    header_left, header_right = st.columns(

        [4, 1],

        gap="large"

    )


    with header_left:

        html(

            """

            <div style="
                padding-top:4px;
            ">

                <div class="small-label">

                    EEG EMOTION DETECTION & WELLNESS

                </div>


                <div class="hero-title"
                     style="
                        margin-top:8px;
                     ">

                    Better Understanding<br>

                    for a

                    <span class="hero-gradient">

                        Healthier You

                    </span>

                </div>


                <div class="hero-subtitle"
                     style="
                        margin-top:13px;
                        max-width:680px;
                     ">

                    Upload EEG feature data, detect emotional
                    patterns using the CNN + Attention model,
                    and receive general wellness support from
                    NeuroPulse AI.

                </div>

            </div>

            """

        )


    with header_right:

        html(

            """

            <div style="
                display:flex;
                align-items:center;
                justify-content:flex-end;
                gap:9px;
                padding-top:5px;
            ">

                <div style="
                    width:34px;
                    height:34px;
                    border-radius:50%;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    background:#20a5f5;
                    color:#061b2d;
                    font-size:18px;
                    font-weight:900;
                    box-shadow:0 0 22px rgba(45,175,255,.4);
                ">

                    ●

                </div>


                <div>

                    <div style="
                        color:#ffffff;
                        font-size:12px;
                        font-weight:750;
                    ">

                        NeuroPulse

                    </div>


                    <div style="
                        color:#45dfb1;
                        font-size:10px;
                        margin-top:2px;
                    ">

                        ● Online

                    </div>

                </div>

            </div>

            """

        )


    # ========================================================
    # HERO VISUAL + SYSTEM STATUS
    # ========================================================

    visual_col, status_col = st.columns(

        [1.8, 1],

        gap="large"

    )


    with visual_col:

        with st.container(
            border=True
        ):

            html(

                """

                <div class="brain-stage">

                    <div class="pulse-left">

                        <div class="pulse-wave"></div>

                    </div>


                    <div class="brain-glow">

                        🧠

                    </div>


                    <div class="pulse-right">

                        <div class="pulse-wave"></div>

                    </div>

                </div>

                """

            )


    with status_col:

        with st.container(
            border=True
        ):

            st.markdown(
                "### 🟢 System Status"
            )


            if model is not None:

                st.success(
                    "🟢 CNN + Attention Ready"
                )

            else:

                st.error(
                    "🔴 Model Not Available"
                )


            if gemini_client is not None:

                st.info(
                    "🤖 Gemini AI Connected"
                )

            else:

                st.warning(
                    "🤖 Gemini AI Not Connected"
                )


            st.caption(
                "4 EEG Channels • 28 Features • 5 Emotion Classes"
            )


    # ========================================================
    # OVERVIEW
    # ========================================================

    st.write("")


    st.markdown(
        "### 📌 NeuroPulse Overview"
    )


    card1, card2, card3, card4 = st.columns(

        4,

        gap="medium"

    )


    # ========================================================
    # EEG CHANNELS CARD
    # ========================================================

    with card1:

        with st.popover(

            "🧠 EEG Channels\n\n4",

            use_container_width=True

        ):

            st.markdown(
                "### 🧠 EEG Channels"
            )


            st.write(
                "NeuroPulse processes four EEG channels:"
            )


            for channel in CHANNELS:

                st.markdown(
                    f"• **{channel}**"
                )


            st.info(
                "4 channels × 7 features form the model input."
            )


    # ========================================================
    # INPUT FEATURES CARD
    # ========================================================

    with card2:

        with st.popover(

            "📊 Input Features\n\n28",

            use_container_width=True

        ):

            st.markdown(
                "### 📊 Input Features"
            )


            feature_df = pd.DataFrame(

                {

                    "Feature":
                    [

                        "Delta",

                        "Theta",

                        "Alpha",

                        "Beta",

                        "Gamma",

                        "Mean",

                        "Standard Deviation",

                    ]

                }

            )


            st.dataframe(

                feature_df,

                use_container_width=True,

                hide_index=True

            )


            st.success(
                "4 Channels × 7 Features = 28"
            )


            st.markdown(
                "#### Model Input Shape"
            )


            st.code(
                "(4, 7)"
            )


    # ========================================================
    # DETECTABLE EMOTIONS CARD
    # ========================================================

    with card3:

        with st.popover(

            "😊 Detectable Emotions\n\n5",

            use_container_width=True

        ):

            st.markdown(
                "### 😊 Detectable Emotions"
            )


            st.write(
                "The trained CNN + Attention model predicts:"
            )


            for icon, emotion_name in EMOTIONS:

                html(

                    f"""

                    <div style="
                        display:flex;
                        align-items:center;
                        gap:10px;
                        padding:9px;
                        margin:5px 0;
                        background:#091f34;
                        border:1px solid #245271;
                        border-radius:10px;
                    ">

                        <span style="
                            font-size:22px;
                        ">

                            {icon}

                        </span>

                        <span style="
                            color:#ffffff;
                            font-size:13px;
                            font-weight:700;
                        ">

                            {emotion_name}

                        </span>

                    </div>

                    """

                )


            st.info(
                "These are the five output classes."
            )


    # ========================================================
    # LOADED ROWS CARD
    # ========================================================

    with card4:

        if st.session_state.dataset is None:

            loaded_rows = "—"

        else:

            loaded_rows = str(

                len(
                    st.session_state.dataset
                )

            )


        with st.popover(

            f"📁 Loaded Rows\n\n{loaded_rows}",

            use_container_width=True

        ):

            st.markdown(
                "### 📁 Loaded Rows"
            )


            if st.session_state.dataset is None:

                st.warning(
                    "No EEG dataset loaded."
                )


                st.write(
                    "Upload a CSV file to start."
                )


            else:

                dataset = (
                    st.session_state.dataset
                )


                st.success(
                    "✅ EEG Dataset Loaded"
                )


                st.write(

                    f"**File:** "
                    f"{st.session_state.dataset_name}"

                )


                st.write(

                    f"**Rows:** "
                    f"{len(dataset)}"

                )


                st.write(

                    f"**Columns:** "
                    f"{len(dataset.columns)}"

                )


                available = sum(

                    feature in dataset.columns

                    for feature in EEG_FEATURES

                )


                st.write(

                    f"**EEG Features:** "
                    f"{available}/28"

                )


                if available == 28:

                    st.success(
                        "All required features are available."
                    )

                else:

                    st.error(
                        "Some EEG features are missing."
                    )


    # ========================================================
    # HOW NEUROPULSE WORKS
    # ========================================================

    st.write("")


    workflow_col, system_col = st.columns(

        [1.65, 1],

        gap="large"

    )


    # ========================================================
    # WORKFLOW CARD
    # ========================================================

    with workflow_col:

        with st.container(
            border=True
        ):

            st.markdown(
                "### ⚙️ How NeuroPulse Works"
            )


            st.write(

                "EEG feature data is standardized using "
                "the saved scaler and passed to the "
                "trained CNN + Attention model."

            )


            step1, step2, step3 = st.columns(
                3
            )


            with step1:

                html(

                    """

                    <div style="
                        text-align:center;
                        padding:8px;
                    ">

                        <div style="
                            font-size:28px;
                        ">
                            ☁️
                        </div>

                        <div style="
                            color:#58dbff;
                            font-size:12px;
                            font-weight:800;
                            margin-top:5px;
                        ">
                            01
                        </div>

                        <div style="
                            color:#ffffff;
                            font-weight:750;
                            margin-top:4px;
                        ">
                            Upload EEG data
                        </div>

                        <div style="
                            color:#7e9fb8;
                            font-size:10px;
                            margin-top:5px;
                        ">
                            Load your CSV file
                        </div>

                    </div>

                    """

                )


            with step2:

                html(

                    """

                    <div style="
                        text-align:center;
                        padding:8px;
                    ">

                        <div style="
                            font-size:28px;
                        ">
                            🧠
                        </div>

                        <div style="
                            color:#a477ff;
                            font-size:12px;
                            font-weight:800;
                            margin-top:5px;
                        ">
                            02
                        </div>

                        <div style="
                            color:#ffffff;
                            font-weight:750;
                            margin-top:4px;
                        ">
                            Detect emotion
                        </div>

                        <div style="
                            color:#7e9fb8;
                            font-size:10px;
                            margin-top:5px;
                        ">
                            Model analyzes EEG
                        </div>

                    </div>

                    """

                )


            with step3:

                html(

                    """

                    <div style="
                        text-align:center;
                        padding:8px;
                    ">

                        <div style="
                            font-size:28px;
                        ">
                            💙
                        </div>

                        <div style="
                            color:#4ce0bd;
                            font-size:12px;
                            font-weight:800;
                            margin-top:5px;
                        ">
                            03
                        </div>

                        <div style="
                            color:#ffffff;
                            font-weight:750;
                            margin-top:4px;
                        ">
                            Get wellness support
                        </div>

                        <div style="
                            color:#7e9fb8;
                            font-size:10px;
                            margin-top:5px;
                        ">
                            Receive AI guidance
                        </div>

                    </div>

                    """

                )


    # ========================================================
    # EEG SYSTEM
    # ========================================================

    with system_col:

        with st.expander(

            "🧠  EEG System",

            expanded=True

        ):

            status_top, shape_top = st.columns(
                [1, 1]
            )


            with status_top:

                if model is not None:

                    st.success(
                        "System Ready"
                    )

                else:

                    st.error(
                        "System Not Ready"
                    )


            with shape_top:

                st.caption(
                    "4 Channels × 7 Features"
                )


            st.markdown(
                "**Channels**"
            )


            channel_columns = st.columns(
                4
            )


            for index, channel in enumerate(
                CHANNELS
            ):

                with channel_columns[index]:

                    html(

                        f"""

                        <div style="
                            text-align:center;
                            padding:7px 3px;
                            background:#08213a;
                            border:1px solid #245778;
                            border-radius:8px;
                            color:#dcecff;
                            font-size:10px;
                            font-weight:700;
                        ">

                            {channel}

                        </div>

                        """

                    )


            st.markdown(
                "**Detectable Emotions**"
            )


            emotion_columns = st.columns(
                5
            )


            for index, (icon, emotion_name) in enumerate(
                EMOTIONS
            ):

                with emotion_columns[index]:

                    html(

                        f"""

                        <div style="
                            text-align:center;
                            padding:6px 2px;
                            background:#071f34;
                            border:1px solid #275978;
                            border-radius:8px;
                        ">

                            <div style="
                                font-size:18px;
                            ">

                                {icon}

                            </div>

                            <div style="
                                color:#dcecff;
                                font-size:7px;
                                font-weight:700;
                                margin-top:2px;
                            ">

                                {emotion_name}

                            </div>

                        </div>

                        """

                    )


            if st.session_state.dataset is not None:

                st.caption(

                    f"Current dataset: "
                    f"{len(st.session_state.dataset)} rows"

                )

            else:

                st.caption(
                    "Current dataset: Not loaded"
                )


    # ========================================================
    # LATEST RESULT
    # ========================================================

    st.write("")


    latest_col, assistant_col = st.columns(

        [1.65, 1],

        gap="large"

    )


    # ========================================================
    # LATEST RESULT CARD
    # ========================================================

    with latest_col:

        with st.container(
            border=True
        ):

            st.markdown(
                "### 📊 Latest Result"
            )


            if st.session_state.last_prediction is None:

                html(

                    """

                    <div style="
                        border:1px dashed #285d7e;
                        border-radius:14px;
                        background:#061a2d;
                        padding:24px 12px;
                        text-align:center;
                    ">

                        <div style="
                            width:55px;
                            height:55px;
                            margin:0 auto;
                            display:flex;
                            align-items:center;
                            justify-content:center;
                            border-radius:50%;
                            background:#0a2b47;
                            border:1px solid #2c709b;
                            font-size:26px;
                        ">

                            🧠

                        </div>

                        <div style="
                            margin-top:8px;
                            color:#ffffff;
                            font-size:17px;
                            font-weight:750;
                        ">

                            No prediction yet

                        </div>

                        <div style="
                            margin-top:5px;
                            color:#7c9db7;
                            font-size:10px;
                        ">

                            Upload a dataset and run an
                            analysis to see results here.

                        </div>

                    </div>

                    """

                )


            else:

                latest_left, latest_right = st.columns(
                    2
                )


                with latest_left:

                    html(

                        f"""

                        <div style="
                            text-align:center;
                            padding:6px;
                        ">

                            <div style="
                                font-size:58px;
                            ">

                                {emotion_icon(
                                    st.session_state.last_prediction
                                )}

                            </div>

                            <div style="
                                font-size:26px;
                                font-weight:800;
                                color:#ffffff;
                            ">

                                {st.session_state.last_prediction}

                            </div>

                        </div>

                        """

                    )


                with latest_right:

                    st.metric(

                        "Confidence",

                        f"{st.session_state.last_confidence:.1f}%"

                    )


                    st.caption(

                        st.session_state.last_analysis_type
                        or
                        "EEG Analysis"

                    )


                st.info(

                    wellness_message(

                        st.session_state.last_prediction

                    )

                )


    # ========================================================
    # AI ASSISTANT PREVIEW
    # ========================================================

    with assistant_col:

        with st.container(
            border=True
        ):

            st.markdown(
                "### ✨ Your AI Wellness Assistant"
            )


            if st.session_state.last_prediction is None:

                html(

                    """

                    <div style="
                        text-align:center;
                        padding:7px 3px 10px;
                    ">

                        <div style="
                            font-size:34px;
                        ">
                            💬
                        </div>

                        <div style="
                            color:#ffffff;
                            font-size:14px;
                            font-weight:750;
                            margin-top:4px;
                        ">
                            Personalized wellness support
                        </div>

                        <div style="
                            color:#7d9eb8;
                            font-size:10px;
                            line-height:1.5;
                            margin-top:5px;
                        ">
                            Run an analysis to start
                            your wellness conversation.
                        </div>

                    </div>

                    """

                )


            else:

                html(

                    f"""

                    <div style="
                        background:#081f34;
                        border:1px solid #255a7d;
                        border-radius:11px;
                        padding:10px;
                        margin:4px 0 11px;
                    ">

                        <div style="
                            color:#79a7cb;
                            font-size:8px;
                            text-transform:uppercase;
                            letter-spacing:1px;
                        ">

                            Current EEG result

                        </div>

                        <div style="
                            color:#ffffff;
                            font-size:14px;
                            font-weight:750;
                            margin-top:4px;
                        ">

                            {emotion_icon(
                                st.session_state.last_prediction
                            )}

                            {st.session_state.last_prediction}

                        </div>

                        <div style="
                            color:#59d9ff;
                            font-size:10px;
                            margin-top:4px;
                        ">

                            {st.session_state.last_confidence:.1f}%
                            confidence

                        </div>

                    </div>

                    """

                )


                st.caption(

                    "Get personalized wellness support "
                    "based on your EEG result."

                )


            if st.button(

                "💬  Wellness Chat",

                use_container_width=True,

                key=
                "home_wellness_chat"

            ):

                st.session_state.page = (
                    "Emotion Analysis"
                )

                st.rerun()


    # ========================================================
    # QUICK ACTIONS
    # ========================================================

    st.write("")


    st.markdown(
        "### ⚡ Quick Actions"
    )


    action_left, action_right = st.columns(
        2
    )


    with action_left:

        if st.button(

            "📤  Upload EEG Dataset",

            use_container_width=True,

            key=
            "home_upload"

        ):

            st.session_state.page = (
                "Upload Dataset"
            )

            st.rerun()


    with action_right:

        if st.button(

            "🧠  Analyze Emotion",

            use_container_width=True,

            key=
            "home_analyze"

        ):

            st.session_state.page = (
                "Emotion Analysis"
            )

            st.rerun()


# ============================================================
# UPLOAD PAGE
# ============================================================

def upload_page():

    page_header(

        "Upload EEG Dataset",

        "Load a CSV containing the 28 EEG features required by NeuroPulse."

    )


    st.info(

        "Expected input: "
        "4 EEG channels × 7 features = 28 features."

    )


    uploaded_file = st.file_uploader(

        "Choose your EEG CSV file",

        type=[
            "csv"
        ],

        key=
        "eeg_dataset_upload"

    )


    if uploaded_file is None:

        if st.session_state.dataset is not None:

            st.success(

                f"Current dataset: "
                f"{st.session_state.dataset_name}"

            )


            st.dataframe(

                st.session_state.dataset.head(
                    10
                ),

                use_container_width=True

            )


        return


    try:

        df = pd.read_csv(
            uploaded_file
        )


        missing = validate_eeg_features(
            df
        )


        if missing:

            st.error(

                "The uploaded dataset is missing "
                "required EEG features."

            )


            st.write(
                missing
            )


            return


        st.session_state.dataset = df

        st.session_state.dataset_name = (
            uploaded_file.name
        )


        clear_prediction_results()


        st.success(

            "✅ Dataset uploaded and validated successfully."

        )


        c1, c2, c3 = st.columns(
            3
        )


        with c1:

            st.metric(
                "Rows",
                len(df)
            )


        with c2:

            st.metric(
                "Columns",
                len(df.columns)
            )


        with c3:

            st.metric(
                "EEG Features",
                "28 / 28"
            )


        st.subheader(
            "Dataset Preview"
        )


        st.dataframe(

            df.head(
                10
            ),

            use_container_width=True

        )


        if st.button(

            "🙂 Continue to Emotion Analysis",

            use_container_width=True,

            key=
            "continue_analysis"

        ):

            st.session_state.page = (
                "Emotion Analysis"
            )

            st.rerun()


    except Exception as exc:

        st.error(

            f"Could not read dataset: {exc}"

        )


# ============================================================
# EMOTION ANALYSIS PAGE
# ============================================================

def emotion_analysis_page():

    page_header(

        "Emotion Analysis",

        "Analyze the complete EEG dataset or an individual EEG row."

    )


    df = st.session_state.dataset


    if df is None:

        st.warning(
            "Please upload an EEG dataset first."
        )


        if st.button(

            "📤 Upload Dataset",

            use_container_width=True,

            key=
            "analysis_upload"

        ):

            st.session_state.page = (
                "Upload Dataset"
            )

            st.rerun()


        return


    missing = validate_eeg_features(
        df
    )


    if missing:

        st.error(
            "Missing EEG features:"
        )


        st.write(
            missing
        )


        return


    st.success(

        f"Dataset: "
        f"{st.session_state.dataset_name} "
        f"• {len(df)} rows"

    )


    # ========================================================
    # ENTIRE DATASET
    # ========================================================

    st.subheader(
        "🧠 Entire EEG Dataset"
    )


    st.write(

        "Run the CNN + Attention model across "
        "all valid EEG rows to calculate the "
        "dominant dataset-level emotion."

    )


    if st.button(

        "🧠 Analyze Entire Dataset",

        type=
        "primary",

        use_container_width=True,

        key=
        "analyze_entire_dataset"

    ):

        with st.spinner(

            "Running CNN + Attention across the EEG dataset..."

        ):

            try:

                result = predict_entire_dataset(
                    df
                )


                st.session_state.dataset_prediction = (
                    result["overall_emotion"]
                )


                st.session_state.dataset_prediction_share = (
                    result["overall_share"]
                )


                st.session_state.dataset_average_confidence = (
                    result["average_confidence"]
                )


                st.session_state.dataset_probabilities = (
                    result["average_probabilities"].tolist()
                )


                st.session_state.dataset_valid_rows = (
                    result["valid_rows"]
                )


                st.session_state.dataset_total_rows = (
                    result["total_rows"]
                )


                st.session_state.dataset_emotion_counts = (
                    result["emotion_counts"].to_dict()
                )


                st.session_state.last_prediction = (
                    result["overall_emotion"]
                )


                st.session_state.last_confidence = (
                    result["average_confidence"]
                )


                st.session_state.last_probabilities = (
                    result["average_probabilities"].tolist()
                )


                st.session_state.last_analysis_type = (
                    "Entire Dataset"
                )


                reset_wellness_chat()


                save_prediction(

                    0,

                    result["overall_emotion"],

                    result["average_confidence"]

                )


                st.success(
                    "✅ Entire dataset analysis completed."
                )


            except Exception as exc:

                st.error(
                    "Dataset analysis failed."
                )


                st.exception(
                    exc
                )


    # ========================================================
    # INDIVIDUAL EEG ROW
    # ========================================================

    st.divider()


    st.subheader(
        "🔎 Individual EEG Row"
    )


    st.caption(

        "Choose a specific row or select one randomly."

    )


    # IMPORTANT:
    # This button must be before number_input.
    # Otherwise Streamlit can produce the widget state error.
    if st.button(

        "🎲 Select Random EEG Row",

        use_container_width=True,

        key=
        "random_eeg_row"

    ):

        random_index = int(

            np.random.randint(

                0,
                len(df)

            )

        )


        st.session_state.eeg_row_number = (

            random_index + 1

        )


        st.session_state.selected_row = (

            random_index

        )


        st.rerun()


    default_row = min(

        int(
            st.session_state.eeg_row_number
        ),

        len(df)

    )


    row_number = st.number_input(

        "EEG Row Number",

        min_value=
        1,

        max_value=
        len(df),

        value=
        default_row,

        step=
        1,

        key=
        "eeg_row_number"

    )


    selected_index = (

        int(row_number)
        -
        1

    )


    st.session_state.selected_row = (
        selected_index
    )


    if st.button(

        "🔬 Analyze Selected EEG Row",

        use_container_width=True,

        key=
        "analyze_selected_row"

    ):

        with st.spinner(

            "Analyzing selected EEG row..."

        ):

            try:

                (

                    emotion,

                    confidence,

                    probabilities

                ) = predict_emotion(

                    df,

                    selected_index

                )


                st.session_state.last_prediction = (
                    emotion
                )


                st.session_state.last_confidence = (
                    confidence
                )


                st.session_state.last_probabilities = (
                    probabilities.tolist()
                )


                st.session_state.last_analysis_type = (
                    "Individual Row"
                )


                st.session_state.selected_row = (
                    selected_index
                )


                reset_wellness_chat()


                save_prediction(

                    selected_index + 1,

                    emotion,

                    confidence

                )


                st.success(

                    f"✅ Row {selected_index + 1} "
                    f"analyzed successfully."

                )


            except Exception as exc:

                st.error(
                    "Individual prediction failed."
                )


                st.exception(
                    exc
                )


    # ========================================================
    # SHARED RESULT AREA
    # ========================================================

    if st.session_state.last_prediction is not None:

        st.divider()


        emotion = (
            st.session_state.last_prediction
        )


        confidence = (
            st.session_state.last_confidence
        )


        probabilities = (
            st.session_state.last_probabilities
        )


        if (

            st.session_state.last_analysis_type
            ==
            "Entire Dataset"

        ):

            secondary_text = (

                f"{st.session_state.dataset_valid_rows}/"
                f"{st.session_state.dataset_total_rows}"
                " valid EEG rows"

            )

        else:

            secondary_text = (

                f"EEG Row "
                f"{st.session_state.selected_row + 1}"

            )


        result_left, result_right = st.columns(

            [1, 1.4],

            gap="large"

        )


        # ====================================================
        # EMOTION DETECTED
        # ====================================================

        with result_left:

            st.subheader(
                "🧠 Emotion Detected"
            )


            with st.container(
                border=True
            ):

                html(

                    f"""

                    <div style="
                        text-align:center;
                        padding:20px 10px;
                    ">

                        <div style="
                            font-size:72px;
                        ">

                            {emotion_icon(emotion)}

                        </div>

                        <div style="
                            font-size:34px;
                            font-weight:800;
                            color:#ffffff;
                        ">

                            {emotion}

                        </div>

                        <div style="
                            margin-top:10px;
                            color:#58dbff;
                            font-size:18px;
                            font-weight:700;
                        ">

                            Confidence:
                            {confidence:.1f}%

                        </div>

                        <div style="
                            margin-top:7px;
                            color:#7fa5cf;
                            font-size:11px;
                        ">

                            CNN + Attention •
                            {secondary_text}

                        </div>

                    </div>

                    """

                )


            st.info(

                wellness_message(
                    emotion
                )

            )


            st.caption(
                "Model Probabilities"
            )


            if (

                probabilities is not None

                and

                label_encoder is not None

            ):

                for label, probability in zip(

                    label_encoder.classes_,

                    probabilities

                ):

                    probability = float(
                        probability
                    )


                    st.write(

                        f"**{label}** — "
                        f"{probability * 100:.1f}%"

                    )


                    st.progress(

                        min(

                            max(
                                probability,
                                0.0
                            ),

                            1.0

                        )

                    )


        # ====================================================
        # WELLNESS ASSISTANT
        # ====================================================

        with result_right:

            wellness_assistant_panel()


        # ====================================================
        # DATASET DISTRIBUTION
        # ====================================================

        if (

            st.session_state.last_analysis_type
            ==
            "Entire Dataset"

        ):

            st.divider()


            st.subheader(

                "Emotion Distribution Across Dataset"

            )


            counts = (

                st.session_state
                .dataset_emotion_counts

                or {}

            )


            distribution_data = []


            for label in label_encoder.classes_:

                count = int(

                    counts.get(

                        label,

                        0

                    )

                )


                percentage = (

                    count
                    /
                    st.session_state.dataset_valid_rows
                    *
                    100

                    if st.session_state.dataset_valid_rows

                    else 0

                )


                distribution_data.append(

                    {

                        "Emotion":
                        label,

                        "Rows":
                        count,

                        "Percentage":
                        percentage

                    }

                )


            distribution_df = pd.DataFrame(

                distribution_data

            )


            st.dataframe(

                distribution_df,

                use_container_width=True,

                hide_index=True

            )


            st.bar_chart(

                distribution_df.set_index(

                    "Emotion"

                )["Rows"],

                use_container_width=True

            )


# ============================================================
# EEG VISUALIZATION
# ============================================================

def eeg_visualization_page():

    page_header(

        "EEG Visualization",

        "Explore EEG channels and frequency-band features."

    )


    df = st.session_state.dataset


    if df is None:

        st.warning(
            "Upload a dataset first."
        )


        if st.button(

            "📤 Upload Dataset",

            use_container_width=True,

            key=
            "visualization_upload"

        ):

            st.session_state.page = (
                "Upload Dataset"
            )

            st.rerun()


        return


    channel_map = {

        "TP9": [

            "TP9_delta",

            "TP9_theta",

            "TP9_alpha",

            "TP9_beta",

            "TP9_gamma",

        ],

        "AF7": [

            "AF7_delta",

            "AF7_theta",

            "AF7_alpha",

            "AF7_beta",

            "AF7_gamma",

        ],

        "AF8": [

            "AF8_delta",

            "AF8_theta",

            "AF8_alpha",

            "AF8_beta",

            "AF8_gamma",

        ],

        "TP10": [

            "TP10_delta",

            "TP10_theta",

            "TP10_alpha",

            "TP10_beta",

            "TP10_gamma",

        ],

    }


    selected_channel = st.selectbox(

        "Select EEG channel",

        list(
            channel_map.keys()
        ),

        key=
        "visualization_channel"

    )


    available_columns = [

        column

        for column in channel_map[
            selected_channel
        ]

        if column in df.columns

    ]


    if available_columns:

        plot_df = (

            df[
                available_columns
            ]

            .apply(

                pd.to_numeric,

                errors="coerce"

            )

        )


        st.subheader(

            f"{selected_channel} Signal Features"

        )


        st.line_chart(

            plot_df.head(
                100
            ),

            use_container_width=True

        )


    st.subheader(
        "Average Frequency Bands"
    )


    band_values = {}


    for band in BANDS:

        columns = [

            column

            for column in EEG_FEATURES

            if column.endswith(
                "_" + band
            )

            and column in df.columns

        ]


        if columns:

            numeric_data = (

                df[
                    columns
                ]

                .apply(

                    pd.to_numeric,

                    errors="coerce"

                )

            )


            band_values[band] = float(

                numeric_data
                .mean()
                .mean()

            )


    if band_values:

        band_df = pd.DataFrame(

            {

                "Band":
                    list(
                        band_values.keys()
                    ),

                "Average Value":
                    list(
                        band_values.values()
                    ),

            }

        )


        st.bar_chart(

            band_df.set_index(
                "Band"
            ),

            use_container_width=True

        )


        st.dataframe(

            band_df,

            use_container_width=True,

            hide_index=True

        )


# ============================================================
# HISTORY
# ============================================================

def history_page():

    page_header(

        "Prediction History",

        "Review stored EEG predictions and progress."

    )


    history = get_history()


    if history.empty:

        st.info(

            "No predictions have been stored yet."

        )

        return


    history["timestamp"] = pd.to_datetime(

        history[
            "timestamp"
        ]

    )


    c1, c2, c3 = st.columns(
        3
    )


    with c1:

        st.metric(

            "Total Predictions",

            len(history)

        )


    with c2:

        st.metric(

            "Most Recorded Emotion",

            history[
                "emotion"
            ]
            .value_counts()
            .index[0]

        )


    with c3:

        st.metric(

            "Average Confidence",

            f"{history['confidence'].mean():.1f}%"

        )


    st.subheader(
        "Prediction Records"
    )


    history_display = history.copy()


    history_display["Analysis"] = (

        history_display[
            "row_number"
        ]

        .apply(

            lambda value:

            "Entire Dataset"

            if int(value) == 0

            else
            f"EEG Row {int(value)}"

        )

    )


    st.dataframe(

        history_display[

            [

                "id",

                "timestamp",

                "Analysis",

                "emotion",

                "confidence"

            ]

        ],

        use_container_width=True,

        hide_index=True

    )


    if os.path.exists(

        HISTORY_CSV_FILE

    ):

        with open(

            HISTORY_CSV_FILE,

            "rb"

        ) as file:

            st.download_button(

                "⬇️ Download Prediction History",

                data=
                file.read(),

                file_name=
                "neuropulse_prediction_history.csv",

                mime=
                "text/csv",

                use_container_width=True,

                key=
                "download_history"

            )


    st.subheader(
        "Emotion Distribution"
    )


    st.bar_chart(

        history[
            "emotion"
        ].value_counts(),

        use_container_width=True

    )


    st.subheader(
        "Daily Progress"
    )


    daily = (

        history

        .assign(

            date=
            history[
                "timestamp"
            ].dt.date

        )

        .groupby(
            "date"
        )

        .size()

    )


    if not daily.empty:

        st.line_chart(

            daily,

            use_container_width=True

        )


    st.subheader(
        "Weekly Progress"
    )


    week_start = (

        datetime.now()

        -
        timedelta(
            days=7
        )

    )


    weekly = history[

        history[
            "timestamp"
        ]

        >=
        week_start

    ]


    if weekly.empty:

        st.info(

            "No predictions recorded "
            "during the last 7 days."

        )

    else:

        weekly_counts = (

            weekly

            .assign(

                date=
                weekly[
                    "timestamp"
                ].dt.date

            )

            .groupby(
                "date"
            )

            .size()

        )


        st.line_chart(

            weekly_counts,

            use_container_width=True

        )


# ============================================================
# SETTINGS
# ============================================================

def settings_page():

    page_header(

        "Settings",

        "NeuroPulse model, AI connection, and application controls."

    )


    left, right = st.columns(
        2
    )


    with left:

        st.markdown(
            "### 🧠 Model"
        )


        st.write(
            "**Architecture:** CNN + Attention"
        )


        st.write(
            "**Input:** 4 Channels × 7 Features"
        )


        st.write(
            "**Features:** 28"
        )


        st.write(
            "**Output Classes:** 5"
        )


        st.write(
            "**Classes:** ANGRY, CALM, FEAR, HAPPY, SAD"
        )


    with right:

        st.markdown(
            "### 🤖 AI Assistant"
        )


        if gemini_client is not None:

            st.success(
                "Gemini AI Connected"
            )

        else:

            st.warning(
                "Gemini AI Not Connected"
            )


        st.write(
            "API key is read from GEMINI_API_KEY."
        )


    st.divider()


    if model_error:

        st.error(
            "Model loading error"
        )


        st.code(
            model_error
        )

    else:

        st.success(

            "✅ CNN + Attention model "
            "loaded successfully."

        )


    st.divider()


    st.subheader(
        "Application Controls"
    )


    if st.button(

        "🔄 Reload Model",

        use_container_width=True,

        key=
        "settings_reload_model"

    ):

        load_neuropulse_model.clear()

        st.rerun()


    if st.button(

        "🗑️ Clear Prediction History",

        use_container_width=True,

        key=
        "settings_clear_history"

    ):

        conn = sqlite3.connect(
            DB_FILE
        )


        conn.execute(
            "DELETE FROM predictions"
        )


        conn.commit()

        conn.close()


        if os.path.exists(

            HISTORY_CSV_FILE

        ):

            os.remove(

                HISTORY_CSV_FILE

            )


        clear_prediction_results()


        st.success(
            "Prediction history cleared."
        )


    if st.button(

        "🏠 Return Home",

        use_container_width=True,

        key=
        "settings_home"

    ):

        st.session_state.page = (
            "Home"
        )

        st.rerun()


# ============================================================
# PAGE ROUTER
# ============================================================

if page == "Home":

    home_page()


elif page == "Upload Dataset":

    upload_page()


elif page == "Emotion Analysis":

    emotion_analysis_page()


elif page == "EEG Visualization":

    eeg_visualization_page()


elif page == "History":

    history_page()


elif page == "Settings":

    settings_page()


else:

    home_page()


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.caption(

    "NeuroPulse Research Project • "
    "CNN + Attention • "
    "Experimental, non-diagnostic wellness support"

)