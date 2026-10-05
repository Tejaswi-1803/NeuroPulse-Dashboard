import numpy as np
import pandas as pd


# ============================================================
# EXPECTED EEG FEATURES
# ============================================================

EXPECTED_FEATURES = [

    # TP9
    "TP9_delta",
    "TP9_theta",
    "TP9_alpha",
    "TP9_beta",
    "TP9_gamma",
    "TP9_mean",
    "TP9_std",

    # AF7
    "AF7_delta",
    "AF7_theta",
    "AF7_alpha",
    "AF7_beta",
    "AF7_gamma",
    "AF7_mean",
    "AF7_std",

    # AF8
    "AF8_delta",
    "AF8_theta",
    "AF8_alpha",
    "AF8_beta",
    "AF8_gamma",
    "AF8_mean",
    "AF8_std",

    # TP10
    "TP10_delta",
    "TP10_theta",
    "TP10_alpha",
    "TP10_beta",
    "TP10_gamma",
    "TP10_mean",
    "TP10_std"
]


# ============================================================
# CHECK DATASET
# ============================================================

def validate_dataset(df):

    missing_columns = [
        column
        for column in EXPECTED_FEATURES
        if column not in df.columns
    ]

    if missing_columns:

        return False, (
            "Missing EEG columns: "
            + ", ".join(missing_columns)
        )

    return True, "Dataset format is correct."


# ============================================================
# PREPARE EEG DATA
# ============================================================

def prepare_data(df, scaler):

    # Select only the required 28 EEG features
    X = df[EXPECTED_FEATURES].copy()

    # Convert values to numeric
    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Check missing values
    if X.isnull().any().any():

        return None, (
            "The dataset contains missing or "
            "non-numeric EEG values."
        )

    # Convert to NumPy
    X = X.values.astype(np.float32)

    # Scale using the SAME scaler used during training
    X_scaled = scaler.transform(X)

    # CNN expects:
    # samples × 4 channels × 7 features
    X_reshaped = X_scaled.reshape(
        -1,
        4,
        7
    )

    return X_reshaped, None


# ============================================================
# PREDICT EMOTION
# ============================================================

def predict_emotion(
    df,
    model,
    scaler,
    label_encoder
):

    # Validate columns
    valid, message = validate_dataset(df)

    if not valid:
        return None, message

    # Prepare data
    X, error = prepare_data(
        df,
        scaler
    )

    if error:
        return None, error

    # CNN + Attention prediction
    probabilities = model.predict(
        X,
        verbose=0
    )

    # Highest probability emotion
    predicted_indices = np.argmax(
        probabilities,
        axis=1
    )

    predicted_emotions = label_encoder.inverse_transform(
        predicted_indices
    )

    # Build results
    results = []

    for i in range(len(df)):

        emotion = predicted_emotions[i]

        confidence = (
            probabilities[i][predicted_indices[i]]
            * 100
        )

        results.append({

            "emotion": emotion,

            "confidence": round(
                float(confidence),
                2
            ),

            "ANGRY": round(
                float(probabilities[i][0] * 100),
                2
            ),

            "CALM": round(
                float(probabilities[i][1] * 100),
                2
            ),
            
            "FEAR": round(
                float(probabilities[i][2] * 100),
                2
            ),

            "HAPPY": round(
                float(probabilities[i][3] * 100),
                2
            ),

            "SAD": round(
                float(probabilities[i][4] * 100),
                2
            )
        })

    return results, None