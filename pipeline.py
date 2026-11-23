import json
import joblib
import numpy as np
import pandas as pd
import librosa

from features import extract_features, SAMPLE_RATE, SAMPLES_PER_SEGMENT, NUM_SEGMENTS


def load_artifacts(models_dir="models"):
    """Charge le meilleur modèle, le scaler, l'encodeur et l'ordre des colonnes."""
    scaler = joblib.load(f"{models_dir}/scaler.pkl")
    le = joblib.load(f"{models_dir}/label_encoder.pkl")
    feature_cols = joblib.load(f"{models_dir}/feature_cols.pkl")
    with open(f"{models_dir}/best_model.json") as f:
        best = json.load(f)["best"]

    if best == "cnn":
        from tensorflow.keras.models import load_model
        model = load_model(f"{models_dir}/cnn_model.h5", compile=False)
    else:
        model = joblib.load(f"{models_dir}/{best}.pkl")

    return {"model": model, "best": best, "scaler": scaler, "le": le, "feature_cols": feature_cols}


def predict_genre(file, artifacts):
    """Renvoie (genre prédit, probabilités par genre) pour un fichier audio."""
    y, _ = librosa.load(file, sr=SAMPLE_RATE, duration=30)

    rows = []
    for s in range(NUM_SEGMENTS):
        seg = y[s * SAMPLES_PER_SEGMENT:(s + 1) * SAMPLES_PER_SEGMENT]
        if len(seg) < SAMPLES_PER_SEGMENT:
            break
        rows.append(extract_features(seg))
    if not rows:
        return None, None   # fichier trop court (< 3 s)

    X = pd.DataFrame(rows)[artifacts["feature_cols"]].values
    X = artifacts["scaler"].transform(X)

    if artifacts["best"] == "cnn":
        proba = artifacts["model"].predict(X[..., np.newaxis], verbose=0)
    else:
        proba = artifacts["model"].predict_proba(X)

    mean_proba = proba.mean(axis=0)
    return artifacts["le"].classes_[mean_proba.argmax()], mean_proba
