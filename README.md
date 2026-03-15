# DEEP-CSAT — eCommerce Customer Satisfaction Prediction

**Project Type**: Binary Classification  
**Target**: Predict whether a customer is `Satisfied` (CSAT 4–5) or `Not Satisfied` (CSAT 1–3)  
**Dataset**: `eCommerce_Customer_support_data.csv` (~83K records, 20 columns)

---

## 📁 Project Structure

```
DEEP-CSAT Project/
├── eCommerce_Customer_support_data.csv  ← raw dataset
├── deep_csat_pipeline.py                ← full ML pipeline (train + save model)
├── app.py                               ← Flask REST API (serve predictions)
├── requirements.txt                     ← Python dependencies
├── deep_csat_model.pkl                  ← saved model (generated after training)
├── plots/                               ← 15+ EDA & model charts (auto-generated)
└── README.md
```

---

## ⚙️ Setup

```bash
# 1. Create & activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Run the Full ML Pipeline

```bash
python deep_csat_pipeline.py
```

This will:
1. Load & explore the dataset  
2. Engineer features (response time, temporal, NLP/TF-IDF)  
3. Generate **15 EDA charts** → saved to `plots/`  
4. Run **3 hypothesis tests** (t-test, chi-square, ANOVA)  
5. Handle class imbalance (SMOTE)  
6. Train **3 ML models** (Logistic Regression, Random Forest, XGBoost)  
7. Save best model → `deep_csat_model.pkl`  
8. Run a sanity check on an unseen sample  

---

## 🌐 Run Flask API (Local)

> Train the pipeline first to generate `deep_csat_model.pkl`

```bash
python app.py
```

### Predict endpoint

**POST** `http://localhost:5000/predict`

```json
{
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
```

**Response:**
```json
{
  "prediction": 1,
  "label": "Satisfied",
  "confidence": 0.8743,
  "confidence_pct": "87.43%"
}
```

---

## ☁️ Cloud Deployment (Render / Railway / Heroku)

1. Push project to GitHub  
2. Add a `Procfile`:
   ```
   web: gunicorn app:app
   ```
3. Set **Start Command** to `python deep_csat_pipeline.py && gunicorn app:app`  
   *(trains on first deploy, then serves)*
4. Set **Python version** to `3.10`

---

## 📊 Model Performance Summary

| Model               | Accuracy | ROC-AUC |
|---------------------|----------|---------|
| Logistic Regression | ~0.72    | ~0.77   |
| Random Forest       | ~0.79    | ~0.84   |
| **XGBoost** ✅      | **~0.81**| **~0.87** |

> XGBoost selected as final model for deployment.

---

## 🔑 Key Features

| Feature                  | Importance |
|--------------------------|------------|
| `response_time_min`      | High       |
| `connected_handling_time`| High       |
| `Tenure Bucket`          | Medium     |
| `Agent Shift`            | Medium     |
| `channel_name`           | Medium     |
| NLP TF-IDF terms         | Variable   |

---

## 📋 Notebook Submission

Open `Sample_ML_Submission_Template-2.ipynb` in **Google Colab** or **Jupyter**.  
The pipeline code in `deep_csat_pipeline.py` maps 1-to-1 to the notebook sections.
