"""
DEEP-CSAT Streamlit App
Predicts eCommerce Customer Satisfaction (CSAT Score)
"""

import re
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

warnings.filterwarnings("ignore")

# ── Download NLTK data
for pkg in ["stopwords", "wordnet"]:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

STOP_WORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ───────────────────────────────────────────────
# PAGE CONFIG
# ───────────────────────────────────────────────
st.set_page_config(
    page_title="DEEP-CSAT Predictor",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ───────────────────────────────────────────────
# CUSTOM CSS
# ───────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .block-container { padding: 2rem 3rem; }
    .stMetric { background: #1e2130; border-radius: 12px; padding: 1rem; }
    .predict-box {
        background: linear-gradient(135deg, #1e3a5f, #0d2137);
        border-radius: 16px; padding: 2rem; margin-top: 1rem;
        border: 1px solid #2e5070;
    }
    .satisfied   { color: #2ecc71; font-size: 2.5rem; font-weight: 800; }
    .unsatisfied { color: #e74c3c; font-size: 2.5rem; font-weight: 800; }
    h1 { color: #5dade2; }
    h2, h3 { color: #aed6f1; }
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────
# HELPER FUNCTIONS
# ───────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    if not isinstance(text, str) or text.strip() == "":
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [lemmatizer.lemmatize(w) for w in text.split()
              if w not in STOP_WORDS and len(w) > 2]
    return " ".join(tokens)


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    """Load the trained model artefact (cached)."""
    if not os.path.exists("deep_csat_model.pkl"):
        return None
    return joblib.load("deep_csat_model.pkl")


def predict_csat(loaded, input_dict: dict):
    model        = loaded["model"]
    scaler       = loaded["scaler"]
    le_dict      = loaded["le_dict"]
    tfidf        = loaded["tfidf"]
    features     = loaded["features"]
    num_features = loaded["num_features"]
    cat_features = [c for c in features
                    if not c.startswith("tfidf_") and c not in num_features]

    row = {
        "channel_name":            input_dict.get("channel_name", ""),
        "category":                input_dict.get("category", ""),
        "Sub-category":            input_dict.get("Sub-category", ""),
        "Customer_City":           input_dict.get("Customer_City", ""),
        "Product_category":        input_dict.get("Product_category", ""),
        "Agent Shift":             input_dict.get("Agent Shift", "Morning"),
        "Tenure Bucket":           input_dict.get("Tenure Bucket", ">90"),
        "Item_price":              float(input_dict.get("Item_price", 0) or 0),
        "connected_handling_time": float(input_dict.get("connected_handling_time", 5) or 5),
        "response_time_min":       float(input_dict.get("response_time_min", 10) or 10),
        "issue_hour":              int(input_dict.get("issue_hour", 10) or 10),
        "issue_dayofweek":         int(input_dict.get("issue_dayofweek", 0) or 0),
        "issue_month":             int(input_dict.get("issue_month", 1) or 1),
        "remarks_clean":           preprocess_text(input_dict.get("Customer Remarks", ""))
    }

    df = pd.DataFrame([row])

    for col in cat_features:
        if col in le_dict:
            le  = le_dict[col]
            val = str(df[col].iloc[0])
            df[col] = le.transform([val])[0] if val in le.classes_ else 0

    tfidf_arr    = tfidf.transform(df["remarks_clean"].fillna("")).toarray()
    tfidf_cols   = [f"tfidf_{c}" for c in tfidf.get_feature_names_out()]
    tfidf_df     = pd.DataFrame(tfidf_arr, columns=tfidf_cols)

    struct_cols  = [c for c in features
                    if c in df.columns and not c.startswith("tfidf_")]
    df_struct    = df[struct_cols].copy()
    df_final     = pd.concat([df_struct.reset_index(drop=True),
                               tfidf_df.reset_index(drop=True)], axis=1)

    for col in features:
        if col not in df_final.columns:
            df_final[col] = 0
    df_final = df_final[features]

    df_final[num_features] = scaler.transform(df_final[num_features])

    pred = int(model.predict(df_final)[0])
    prob = float(model.predict_proba(df_final)[0][1])
    return pred, prob


# ───────────────────────────────────────────────
# HEADER
# ───────────────────────────────────────────────
st.title("⭐ DEEP-CSAT — Customer Satisfaction Predictor")
st.markdown("**eCommerce Customer Support AI** • Predict `Satisfied` or `Not Satisfied` in real-time")
st.divider()

# ───────────────────────────────────────────────
# SIDEBAR — About
# ───────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/star.png", width=80)
    st.markdown("## About")
    st.info(
        "**DEEP-CSAT** predicts whether a customer will leave a "
        "satisfied (4-5 ⭐) or unsatisfied (1-3 ⭐) rating after "
        "a customer support interaction."
    )
    st.markdown("### Model")
    st.success("✅ XGBoost Classifier")
    st.markdown("**Features used:**")
    st.markdown("""
    - Channel & Category  
    - Agent Shift & Tenure  
    - Response & Handling Time  
    - Product & City  
    - Customer Remarks (NLP)
    """)
    st.markdown("---")
    st.caption("Project: DEEP-CSAT | AlmaBetter Capstone")

# ───────────────────────────────────────────────
# LOAD MODEL
# ───────────────────────────────────────────────
loaded = load_model()

if loaded is None:
    st.error(
        "⚠️ **Model not found!** `deep_csat_model.pkl` is missing.\n\n"
        "Please run `python deep_csat_pipeline.py` first to train & save the model, "
        "then push `deep_csat_model.pkl` to your GitHub repo."
    )
    st.stop()

# ───────────────────────────────────────────────
# TABS
# ───────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔮 Predict CSAT", "📊 About the Model"])

# ═══════════ TAB 1: PREDICT ═══════════
with tab1:

    st.subheader("📝 Enter Support Ticket Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        channel = st.selectbox("Channel Name", [
            "Inbound", "Outcall", "Chat", "Email", "Social Media"
        ])
        category = st.selectbox("Category", [
            "Product Queries", "Service Issue", "Billing Issue",
            "Returns & Refunds", "Delivery Issue", "Others"
        ])
        sub_cat = st.selectbox("Sub-Category", [
            "Product Specific Information", "Life Insurance",
            "Technical Support", "Payment Issue",
            "Order Tracking", "General Inquiry"
        ])
        city = st.text_input("Customer City", value="Mumbai")

    with col2:
        product_cat = st.selectbox("Product Category", [
            "Electronics", "Clothing", "Home & Kitchen",
            "Books", "Sports", "Grocery", "Beauty", "Toys"
        ])
        agent_shift = st.selectbox("Agent Shift", [
            "Morning", "Afternoon", "Evening", "Night"
        ])
        tenure = st.selectbox("Agent Tenure Bucket", [
            ">90", "61-90", "31-60", "0-30",
            "On Job Training"
        ])
        item_price = st.number_input("Item Price (₹)", min_value=0.0,
                                      max_value=100000.0, value=1500.0, step=100.0)

    with col3:
        handling_time = st.slider("Handling Time (min)", 0, 120, 8)
        response_time = st.slider("Response Time (min)", 0, 180, 12)
        issue_hour    = st.slider("Issue Hour (0–23)", 0, 23, 10)
        issue_dow     = st.selectbox("Day of Week",
                                      ["Monday","Tuesday","Wednesday",
                                       "Thursday","Friday","Saturday","Sunday"])
        issue_month   = st.selectbox("Month", list(range(1, 13)),
                                      format_func=lambda m: [
                                          "Jan","Feb","Mar","Apr","May","Jun",
                                          "Jul","Aug","Sep","Oct","Nov","Dec"][m-1])

    remarks = st.text_area(
        "Customer Remarks (optional)",
        placeholder="e.g. The agent was very helpful and resolved my issue quickly.",
        height=100
    )

    dow_map = {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,
               "Friday":4,"Saturday":5,"Sunday":6}

    st.markdown("")
    predict_btn = st.button("🔮 Predict CSAT", type="primary", use_container_width=True)

    if predict_btn:
        input_dict = {
            "channel_name":            channel,
            "category":                category,
            "Sub-category":            sub_cat,
            "Customer_City":           city,
            "Product_category":        product_cat,
            "Agent Shift":             agent_shift,
            "Tenure Bucket":           tenure,
            "Item_price":              item_price,
            "connected_handling_time": handling_time,
            "response_time_min":       response_time,
            "issue_hour":              issue_hour,
            "issue_dayofweek":         dow_map[issue_dow],
            "issue_month":             issue_month,
            "Customer Remarks":        remarks
        }

        with st.spinner("Predicting..."):
            pred, prob = predict_csat(loaded, input_dict)

        st.divider()
        r1, r2, r3 = st.columns([1, 1, 1])

        with r1:
            label = "✅ Satisfied" if pred == 1 else "❌ Not Satisfied"
            css_class = "satisfied" if pred == 1 else "unsatisfied"
            st.markdown(
                f'<div class="predict-box">'
                f'<div style="font-size:1rem;color:#aaa;">Prediction</div>'
                f'<div class="{css_class}">{label}</div>'
                f'<div style="color:#aaa;margin-top:0.5rem;">Based on ticket details</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        with r2:
            st.metric("Confidence", f"{prob:.2%}")
            # Gauge bar
            fig, ax = plt.subplots(figsize=(4, 0.7))
            ax.barh([0], [prob], color="#2ecc71" if pred == 1 else "#e74c3c",
                    height=0.6)
            ax.barh([0], [1 - prob], left=[prob], color="#1e2130", height=0.6)
            ax.set_xlim(0, 1); ax.axis("off")
            fig.patch.set_alpha(0)
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with r3:
            st.metric("Not-Satisfied Probability", f"{1-prob:.2%}")
            st.info(
                "**Tip:** Response time and handling time are the strongest "
                "predictors of CSAT."
            )

# ═══════════ TAB 2: ABOUT ═══════════
with tab2:
    st.subheader("📊 Model Performance Summary")

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Best Model",    "XGBoost")
    mc2.metric("Accuracy",      "~81%")
    mc3.metric("ROC-AUC Score", "~0.87")

    st.divider()
    st.markdown("""
    ### 🔑 Top Features (Importance)

    | Feature | Importance |
    |---------|-----------|
    | `response_time_min` | ⭐⭐⭐⭐⭐ |
    | `connected_handling_time` | ⭐⭐⭐⭐⭐ |
    | `Tenure Bucket` | ⭐⭐⭐⭐ |
    | `channel_name` | ⭐⭐⭐ |
    | `Agent Shift` | ⭐⭐⭐ |
    | `Item_price` | ⭐⭐ |
    | NLP (TF-IDF terms) | ⭐⭐ |
    
    ### 📋 ML Pipeline Steps
    1. **Data Wrangling** — parse timestamps, strip whitespace, drop duplicates  
    2. **Feature Engineering** — response time, hour/DOW/month from timestamps  
    3. **NLP** — lemmatize + TF-IDF (top 50 terms) on Customer Remarks  
    4. **Preprocessing** — median imputation, IQR outlier capping, label encoding, StandardScaler  
    5. **SMOTE** — handle class imbalance  
    6. **Models** — Logistic Regression, Random Forest (GridSearchCV), XGBoost  
    7. **Saved** → `deep_csat_model.pkl`

    ### 🧪 Hypothesis Tests Performed
    - **H1** Welch's t-test: response time differs between satisfied vs not satisfied  
    - **H2** Chi-Square: CSAT is NOT independent of Agent Shift  
    - **H3** One-Way ANOVA: handling time differs significantly across channels  
    """)

    st.divider()
    st.caption("DEEP-CSAT | AlmaBetter ML Capstone | Dataset: eCommerce Customer Support")
