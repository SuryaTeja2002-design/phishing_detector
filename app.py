"""
app.py
------
Flask REST API + web dashboard for the Phishing URL Detector.

Endpoints:
  GET  /                    → Main dashboard
  POST /predict             → Full analysis (ML + cyber + explainability)
  POST /retrain             → Submit a labeled URL for online retraining
  GET  /model/stats         → Model performance metrics
  GET  /model/importance    → Feature importance data
  GET  /history             → Recent predictions (in-memory)
  POST /batch               → Batch URL analysis

Run with:
    python app.py
"""

import os
import json
import pickle
import datetime
import numpy as np
from flask import Flask, request, jsonify, render_template

from feature_extractor import features_to_vector, extract_features, get_feature_names
from cyber_analysis import analyze_url
from train_model import explain_prediction, compute_uncertainty
from online_learner import OnlineLearner
import pandas as pd

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

# In-memory prediction history (last 100)
prediction_history = []

# ── Load model ────────────────────────────────────────────────────────────────

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("model.pkl not found — run: python train_model.py")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

try:
    MODEL_DATA = load_model()
    ENSEMBLE    = MODEL_DATA["ensemble"]
    MODELS      = MODEL_DATA["individual_models"]
    ONLINE      = MODEL_DATA["online_learner"]
    FEATURE_NAMES = MODEL_DATA["feature_names"]
    PERM_IMP    = MODEL_DATA["permutation_importance"]
    MODEL_METRICS = MODEL_DATA["model_metrics"]
    PERM_IMP_DF = pd.DataFrame(PERM_IMP)
    print(f"✓ Models loaded — Ensemble AUC: {MODEL_DATA['ensemble_auc']:.4f}")
except Exception as e:
    print(f"⚠ {e}")
    MODEL_DATA = ENSEMBLE = MODELS = ONLINE = None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if ENSEMBLE is None:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 500

    data = request.get_json()
    if not data or "url" not in data:
        return jsonify({"error": "Missing 'url' in request body"}), 400

    url = data["url"].strip()
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    try:
        feat_vec = features_to_vector(url)
        feat_dict = extract_features(url)
        X = np.array([feat_vec])

        # Ensemble prediction
        pred = int(ENSEMBLE.predict(X)[0])
        proba = ENSEMBLE.predict_proba(X)[0]
        phish_prob = float(proba[1])

        # Per-model predictions (for comparison chart)
        per_model = {}
        for name, model in MODELS.items():
            try:
                p = float(model.predict_proba(X)[0][1])
                per_model[name] = round(p * 100, 1)
            except Exception:
                per_model[name] = None

        # Online model prediction
        online_result = ONLINE.predict(url) if ONLINE else {}

        # Uncertainty
        uncertainty = compute_uncertainty(MODELS, X)

        # Explainability (top feature contributions)
        explanations = explain_prediction(url, ENSEMBLE, None, FEATURE_NAMES, PERM_IMP_DF)

        # Cyber analysis
        cyber = analyze_url(url)

        # Risk level
        if phish_prob >= 0.75:
            risk = "HIGH"
        elif phish_prob >= 0.40:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        result = {
            "url": url,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "prediction": "phishing" if pred == 1 else "legitimate",
            "phishing_probability": round(phish_prob * 100, 1),
            "risk_level": risk,

            # Multi-model comparison
            "per_model_probabilities": per_model,

            # Online model
            "online_model": online_result,

            # Uncertainty
            "uncertainty": {
                "std": uncertainty["std"],
                "confidence_level": uncertainty["confidence"],
                "model_agreement": round((1 - uncertainty["std"]) * 100, 1),
            },

            # Explainability
            "top_features": explanations,

            # Features
            "features": {k: round(v, 4) for k, v in feat_dict.items()},

            # Cyber
            "cyber": {
                "whois": cyber["whois"],
                "dns": cyber["dns"],
                "ssl": cyber["ssl"],
                "threat_intel": cyber["threat_intel"],
                "summary_flags": cyber["summary_flags"],
                "cyber_risk_score": cyber["cyber_risk_score"],
            },
        }

        # Store in history
        prediction_history.append({
            "url": url,
            "prediction": result["prediction"],
            "phishing_probability": result["phishing_probability"],
            "risk_level": risk,
            "timestamp": result["timestamp"],
        })
        if len(prediction_history) > 100:
            prediction_history.pop(0)

        return jsonify(result)

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/retrain", methods=["POST"])
def retrain():
    """Submit a labeled URL for online model retraining."""
    if ONLINE is None:
        return jsonify({"error": "Online model not loaded"}), 500

    data = request.get_json()
    if not data or "url" not in data or "label" not in data:
        return jsonify({"error": "Required: url (str) and label (0=legit, 1=phishing)"}), 400

    url = data["url"].strip()
    label = int(data["label"])
    if label not in (0, 1):
        return jsonify({"error": "label must be 0 or 1"}), 400

    result = ONLINE.update(url, label)
    result["message"] = f"Online model updated with new {'phishing' if label==1 else 'legitimate'} sample"
    result["online_stats"] = ONLINE.get_stats()
    return jsonify(result)


@app.route("/model/stats")
def model_stats():
    if MODEL_DATA is None:
        return jsonify({"error": "Model not loaded"}), 500
    return jsonify({
        "model_metrics": MODEL_METRICS,
        "ensemble_auc": MODEL_DATA.get("ensemble_auc"),
        "training_size": MODEL_DATA.get("training_size"),
        "feature_count": len(FEATURE_NAMES),
        "online_learner_stats": ONLINE.get_stats() if ONLINE else {},
    })


@app.route("/model/importance")
def feature_importance():
    if MODEL_DATA is None:
        return jsonify({"error": "Model not loaded"}), 500
    return jsonify({
        "feature_importance": PERM_IMP,
        "method": "Permutation Importance (SHAP-equivalent)",
        "description": (
            "Each feature's importance is measured by randomly shuffling its values "
            "and measuring the drop in ROC-AUC. Larger drop = more important feature."
        ),
    })


@app.route("/history")
def history():
    return jsonify({
        "count": len(prediction_history),
        "predictions": list(reversed(prediction_history[-50:]))
    })


@app.route("/batch", methods=["POST"])
def batch():
    if ENSEMBLE is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()
    if not data or "urls" not in data:
        return jsonify({"error": "Missing 'urls'"}), 400

    results = []
    for url in data["urls"][:50]:
        url = url.strip()
        if not url.startswith(("http://", "https://")):
            url = "http://" + url
        try:
            X = np.array([features_to_vector(url)])
            pred = int(ENSEMBLE.predict(X)[0])
            prob = float(ENSEMBLE.predict_proba(X)[0][1])
            risk = "HIGH" if prob >= 0.75 else ("MEDIUM" if prob >= 0.40 else "LOW")
            results.append({
                "url": url,
                "prediction": "phishing" if pred == 1 else "legitimate",
                "phishing_probability": round(prob * 100, 1),
                "risk_level": risk,
            })
        except Exception as e:
            results.append({"url": url, "error": str(e)})

    return jsonify({"count": len(results), "results": results})


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": ENSEMBLE is not None,
        "online_learner": ONLINE.get_stats() if ONLINE else None,
    })


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if ENSEMBLE is None:
        print("\n⚠  Model not found — training now...\n")
        import subprocess, sys
        subprocess.run([sys.executable, "train_model.py"], check=True)
        MODEL_DATA = load_model()
        ENSEMBLE = MODEL_DATA["ensemble"]
        MODELS = MODEL_DATA["individual_models"]
        ONLINE = MODEL_DATA["online_learner"]
        PERM_IMP_DF = pd.DataFrame(MODEL_DATA["permutation_importance"])

    app.run(debug=True, port=5000)