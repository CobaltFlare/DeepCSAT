"""
=======================================================================
DEEP-CSAT  –  Flask Deployment API
=======================================================================
Run:  python app.py          (development)
      gunicorn app:app        (production)

POST /predict
  Body (JSON): {
    "channel_name": "Inbound",
    "category": "Product Queries",
    "Sub-category": "Product Specific Information",
    "Customer_City": "Mumbai",
    "Product_category": "Electronics",
    "Agent Shift": "Morning",
    "Tenure Bucket": ">90",
    "Item_price": 1500.0,
    "connected_handling_time": 8.0,
    "response_time_min": 12.0,
    "issue_hour": 10,
    "issue_dayofweek": 1,
    "issue_month": 8,
    "Customer Remarks": "product working great happy service"
  }
=======================================================================
"""

import re
import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data (first run)
for pkg in ["stopwords", "wordnet"]:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

# ── Load model artefact
MODEL_PATH = "deep_csat_model.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file '{MODEL_PATH}' not found. "
        "Run deep_csat_pipeline.py first to train and save the model."
    )

loaded       = joblib.load(MODEL_PATH)
model        = loaded["model"]
scaler       = loaded["scaler"]
le_dict      = loaded["le_dict"]
tfidf        = loaded["tfidf"]
features     = loaded["features"]
num_features = loaded["num_features"]
cat_features = [c for c in features
                if not c.startswith("tfidf_") and c not in num_features]

STOP_WORDS  = set(stopwords.words("english"))
lemmatizer  = WordNetLemmatizer()

app = Flask(__name__)


def preprocess_text(text: str) -> str:
    if not isinstance(text, str) or text.strip() == "":
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [lemmatizer.lemmatize(w) for w in text.split()
              if w not in STOP_WORDS and len(w) > 2]
    return " ".join(tokens)


def build_feature_row(data: dict) -> pd.DataFrame:
    """Convert raw request dict → aligned feature DataFrame."""
    row = {
        "channel_name":           data.get("channel_name", ""),
        "category":               data.get("category", ""),
        "Sub-category":           data.get("Sub-category", ""),
        "Customer_City":          data.get("Customer_City", ""),
        "Product_category":       data.get("Product_category", ""),
        "Agent Shift":            data.get("Agent Shift", ""),
        "Tenure Bucket":          data.get("Tenure Bucket", ""),
        "Item_price":             float(data.get("Item_price", 0) or 0),
        "connected_handling_time": float(data.get("connected_handling_time", 0) or 0),
        "response_time_min":      float(data.get("response_time_min", 0) or 0),
        "issue_hour":             int(data.get("issue_hour", 0) or 0),
        "issue_dayofweek":        int(data.get("issue_dayofweek", 0) or 0),
        "issue_month":            int(data.get("issue_month", 1) or 1),
        "remarks_clean":          preprocess_text(
                                    data.get("Customer Remarks", ""))
    }

    df = pd.DataFrame([row])

    # Encode categoricals
    for col in cat_features:
        if col in le_dict:
            le = le_dict[col]
            val = str(df[col].iloc[0])
            df[col] = le.transform([val])[0] if val in le.classes_ else 0

    # TF-IDF
    tfidf_arr = tfidf.transform(df["remarks_clean"].fillna("")).toarray()
    tfidf_df  = pd.DataFrame(
        tfidf_arr,
        columns=[f"tfidf_{c}" for c in tfidf.get_feature_names_out()]
    )

    struct_cols = [c for c in features
                   if c in df.columns and not c.startswith("tfidf_")]
    df_struct = df[struct_cols].copy()
    df_final  = pd.concat([df_struct.reset_index(drop=True),
                            tfidf_df.reset_index(drop=True)], axis=1)

    # Ensure all feature columns exist
    for col in features:
        if col not in df_final.columns:
            df_final[col] = 0
    df_final = df_final[features]

    # Scale numerics
    df_final[num_features] = scaler.transform(df_final[num_features])
    return df_final


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "DEEP-CSAT Prediction API",
        "version": "1.0",
        "endpoints": {
            "POST /predict": "Predict customer satisfaction (CSAT)"
        }
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "Empty request body"}), 400

        df_input = build_feature_row(data)

        pred  = int(model.predict(df_input)[0])
        prob  = float(model.predict_proba(df_input)[0][1])
        label = "Satisfied" if pred == 1 else "Not Satisfied"

        return jsonify({
            "prediction":        pred,
            "label":             label,
            "confidence":        round(prob, 4),
            "confidence_pct":    f"{prob:.2%}"
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
