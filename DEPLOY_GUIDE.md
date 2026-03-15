# DEEP-CSAT — GitHub Push & Streamlit Cloud Deployment Guide

## Step-by-step: Git → Streamlit Cloud

### Step 1 — Train the model FIRST (local)
```bash
pip install -r requirements.txt
python deep_csat_pipeline.py
```
> This generates `deep_csat_model.pkl` — **must push this to GitHub too!**

---

### Step 2 — Create GitHub Repo & Push
```bash
cd "d:\document\DEEP-CSAT Project-20260315T092309Z-1-001\DEEP-CSAT Project"

git init
git add .
git commit -m "Initial commit: DEEP-CSAT Streamlit App"

# Create repo on github.com then:
git remote add origin https://github.com/YOUR_USERNAME/deep-csat.git
git branch -M main
git push -u origin main
```

> **⚠️ Important**: Make sure `deep_csat_model.pkl` is included!
> If CSV is >100MB, add it to `.gitignore` — model is already trained.

---

### Step 3 — Deploy on Streamlit Cloud (FREE)

1. Go to **[share.streamlit.io](https://share.streamlit.io)**
2. Sign in with GitHub
3. Click **"New app"**
4. Fill in:
   - **Repository**: `YOUR_USERNAME/deep-csat`
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
5. Click **"Deploy!"** ✅

Done! Your app will be live at:  
`https://YOUR_USERNAME-deep-csat-streamlit-app-XXXX.streamlit.app`

---

## Files needed in GitHub

```
✅ streamlit_app.py          ← main app
✅ deep_csat_pipeline.py     ← ML pipeline
✅ deep_csat_model.pkl       ← trained model (must push!)
✅ requirements.txt
✅ .streamlit/config.toml    ← dark theme
```

## What to put in `.gitignore`
```
eCommerce_Customer_support_data.csv    # too large for GitHub
plots/
__pycache__/
*.pyc
venv/
```
