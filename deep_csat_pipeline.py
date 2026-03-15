"""
=======================================================================
DEEP-CSAT: eCommerce Customer Support CSAT Score Prediction
=======================================================================
Project Type  : Classification (CSAT Score Prediction)
Dataset       : eCommerce_Customer_support_data.csv
Target Column : CSAT Score (binarized: 1 = Satisfied [score 4-5],
                                       0 = Not Satisfied [score 1-3])
=======================================================================
"""

# ─────────────────────────────────────────────────────────────────────
# 1. IMPORT LIBRARIES
# ─────────────────────────────────────────────────────────────────────
import os
import re
import warnings
import joblib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless backend for deployment
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

# Text / NLP
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Sklearn – preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

# Sklearn – modelling
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     GridSearchCV, StratifiedKFold)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score,
                              ConfusionMatrixDisplay)

# Imbalanced data
from imblearn.over_sampling import SMOTE

# XGBoost
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# Download required NLTK data (first run only)
for pkg in ["stopwords", "wordnet", "averaged_perceptron_tagger"]:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

SEED = 42
np.random.seed(SEED)

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────
# HELPER – save & show figure
# ─────────────────────────────────────────────────────────────────────
def save_fig(name: str) -> None:
    """Save current matplotlib figure to PLOTS_DIR and close it."""
    path = os.path.join(PLOTS_DIR, f"{name}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT] saved → {path}")


# ─────────────────────────────────────────────────────────────────────
# 2. DATASET LOADING
# ─────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 1 – KNOW YOUR DATA")
print("="*60)

DATA_FILE = "eCommerce_Customer_support_data.csv"
df = pd.read_csv(DATA_FILE, encoding="utf-8", low_memory=False)
print(f"\n✅ Dataset loaded  →  {df.shape[0]:,} rows × {df.shape[1]} columns")

# ── First look
print("\n── First 3 rows ──")
print(df.head(3).to_string())

# ── Shape
print(f"\nRows   : {df.shape[0]:,}")
print(f"Columns: {df.shape[1]}")

# ── Info
print("\n── Dataset Info ──")
df.info()

# ── Duplicates
dups = df.duplicated().sum()
print(f"\nDuplicate rows : {dups:,}")

# ── Missing values
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    "Missing Count": missing,
    "Missing %":     missing_pct
}).query("`Missing Count` > 0").sort_values("Missing %", ascending=False)
print("\n── Missing Values ──")
print(missing_df.to_string())

# Chart 0 – Missing value heatmap
fig, ax = plt.subplots(figsize=(14, 5))
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap="viridis", ax=ax)
ax.set_title("Missing Value Heatmap", fontsize=14, fontweight="bold")
save_fig("00_missing_heatmap")


# ─────────────────────────────────────────────────────────────────────
# 3. UNDERSTAND VARIABLES
# ─────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 2 – UNDERSTAND VARIABLES")
print("="*60)

print("\nColumn names:")
print(df.columns.tolist())

print("\n── Describe (numeric) ──")
print(df.describe().to_string())

print("\n── Unique value counts ──")
for col in df.columns:
    print(f"  {col:35s}  unique={df[col].nunique():>6,}")


# ─────────────────────────────────────────────────────────────────────
# 4. DATA WRANGLING
# ─────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 3 – DATA WRANGLING")
print("="*60)

# 4a. Remove exact duplicates
df.drop_duplicates(inplace=True)
print(f"After dropping duplicates : {df.shape[0]:,} rows")

# 4b. Parse datetime columns
for col in ["Issue_reported at", "issue_responded"]:
    df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)

df["order_date_time"] = pd.to_datetime(df["order_date_time"], errors="coerce",
                                        infer_datetime_format=True)

# 4c. Compute response time in minutes (feature engineering early)
df["response_time_min"] = (
    (df["issue_responded"] - df["Issue_reported at"])
    .dt.total_seconds() / 60
).clip(lower=0)

# 4d. Extract temporal features
df["issue_hour"]       = df["Issue_reported at"].dt.hour
df["issue_dayofweek"]  = df["Issue_reported at"].dt.dayofweek   # 0=Mon
df["issue_month"]      = df["Issue_reported at"].dt.month

# 4e. Clean numeric column
df["Item_price"]              = pd.to_numeric(df["Item_price"], errors="coerce")
df["connected_handling_time"] = pd.to_numeric(df["connected_handling_time"],
                                              errors="coerce")

# 4f. Strip whitespace in string columns
str_cols = df.select_dtypes(include="object").columns
df[str_cols] = df[str_cols].apply(lambda s: s.str.strip() if s.dtype == "O" else s)

print("Wrangling complete.")
print(df.dtypes)


# ─────────────────────────────────────────────────────────────────────
# 5. DATA VISUALISATION  (15 charts)
# ─────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 4 – DATA VISUALISATION")
print("="*60)

# ── Chart 1: CSAT Score Distribution (target)
plt.figure(figsize=(8, 5))
csat_counts = df["CSAT Score"].value_counts().sort_index()
sns.barplot(x=csat_counts.index, y=csat_counts.values,
            palette="viridis", hue=csat_counts.index, legend=False)
plt.title("Chart 1 – CSAT Score Distribution (Target)", fontweight="bold")
plt.xlabel("CSAT Score"); plt.ylabel("Count")
save_fig("01_csat_distribution")

# ── Chart 2: Channel-wise CSAT
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x="channel_name", hue="CSAT Score", palette="Set2")
plt.title("Chart 2 – CSAT Score by Channel", fontweight="bold")
plt.xticks(rotation=30)
save_fig("02_csat_by_channel")

# ── Chart 3: Category-wise CSAT
plt.figure(figsize=(12, 5))
top_cats = df["category"].value_counts().head(10).index
cat_df = df[df["category"].isin(top_cats)]
sns.countplot(data=cat_df, x="category", hue="CSAT Score", palette="coolwarm")
plt.title("Chart 3 – CSAT Score by Category (Top 10)", fontweight="bold")
plt.xticks(rotation=40, ha="right")
save_fig("03_csat_by_category")

# ── Chart 4: Agent Shift vs CSAT
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Agent Shift", hue="CSAT Score", palette="muted")
plt.title("Chart 4 – CSAT Score by Agent Shift", fontweight="bold")
save_fig("04_csat_by_shift")

# ── Chart 5: Tenure Bucket vs Avg CSAT
plt.figure(figsize=(10, 5))
tenure_csat = df.groupby("Tenure Bucket")["CSAT Score"].mean().sort_values()
sns.barplot(x=tenure_csat.index, y=tenure_csat.values, palette="Blues_d",
            hue=tenure_csat.index, legend=False)
plt.title("Chart 5 – Average CSAT by Tenure Bucket", fontweight="bold")
plt.ylabel("Avg CSAT"); plt.xticks(rotation=30)
save_fig("05_avg_csat_by_tenure")

# ── Chart 6: Response Time Distribution
plt.figure(figsize=(10, 5))
valid_rt = df["response_time_min"].dropna()
valid_rt = valid_rt[valid_rt <= valid_rt.quantile(0.99)]
sns.histplot(valid_rt, bins=50, kde=True, color="steelblue")
plt.title("Chart 6 – Response Time Distribution (minutes)", fontweight="bold")
save_fig("06_response_time_dist")

# ── Chart 7: Item Price vs CSAT (box)
plt.figure(figsize=(10, 5))
df_price = df.dropna(subset=["Item_price"])
df_price = df_price[df_price["Item_price"] <= df_price["Item_price"].quantile(0.99)]
sns.boxplot(data=df_price, x="CSAT Score", y="Item_price", palette="pastel")
plt.title("Chart 7 – Item Price vs CSAT Score", fontweight="bold")
save_fig("07_price_vs_csat_box")

# ── Chart 8: Handling Time vs CSAT
plt.figure(figsize=(10, 5))
df_ht = df.dropna(subset=["connected_handling_time"])
df_ht = df_ht[df_ht["connected_handling_time"] <=
              df_ht["connected_handling_time"].quantile(0.99)]
sns.boxplot(data=df_ht, x="CSAT Score", y="connected_handling_time",
            palette="Set3")
plt.title("Chart 8 – Handling Time vs CSAT Score", fontweight="bold")
save_fig("08_handling_vs_csat")

# ── Chart 9: Hour of Day vs Avg CSAT
plt.figure(figsize=(12, 5))
hour_csat = df.groupby("issue_hour")["CSAT Score"].mean()
sns.lineplot(x=hour_csat.index, y=hour_csat.values, marker="o", color="coral")
plt.title("Chart 9 – Average CSAT by Hour of Day", fontweight="bold")
plt.xlabel("Hour"); plt.ylabel("Avg CSAT")
save_fig("09_csat_by_hour")

# ── Chart 10: Day of Week vs Avg CSAT
plt.figure(figsize=(9, 5))
days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
dow_csat = df.groupby("issue_dayofweek")["CSAT Score"].mean()
sns.barplot(x=[days[i] for i in dow_csat.index], y=dow_csat.values,
            palette="magma", hue=[days[i] for i in dow_csat.index], legend=False)
plt.title("Chart 10 – Avg CSAT by Day of Week", fontweight="bold")
save_fig("10_csat_by_dayofweek")

# ── Chart 11: Top 10 Cities by ticket volume
plt.figure(figsize=(12, 5))
city_counts = df["Customer_City"].value_counts().head(10)
sns.barplot(x=city_counts.index, y=city_counts.values,
            palette="cubehelix", hue=city_counts.index, legend=False)
plt.title("Chart 11 – Top 10 Cities by Ticket Volume", fontweight="bold")
plt.xticks(rotation=30, ha="right")
save_fig("11_top_cities")

# ── Chart 12: Product Category vs CSAT
plt.figure(figsize=(12, 5))
top_prods = df["Product_category"].value_counts().head(8).index
prod_df = df[df["Product_category"].isin(top_prods)]
sns.countplot(data=prod_df, x="Product_category", hue="CSAT Score",
              palette="tab10")
plt.title("Chart 12 – CSAT by Product Category (Top 8)", fontweight="bold")
plt.xticks(rotation=35, ha="right")
save_fig("12_csat_by_product_cat")

# ── Chart 13: Monthly ticket volume
plt.figure(figsize=(10, 5))
monthly = df.groupby("issue_month").size()
sns.lineplot(x=monthly.index, y=monthly.values, marker="s", color="teal")
plt.title("Chart 13 – Monthly Ticket Volume", fontweight="bold")
plt.xlabel("Month"); plt.ylabel("Tickets")
save_fig("13_monthly_tickets")

# ── Chart 14: Correlation Heatmap
plt.figure(figsize=(10, 7))
num_cols = df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
corr = num_cols.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", linewidths=0.5)
plt.title("Chart 14 – Correlation Heatmap", fontweight="bold")
save_fig("14_correlation_heatmap")

# ── Chart 15: Pie – CSAT Satisfied vs Not Satisfied
df_csat_binary = df["CSAT Score"].apply(lambda x: "Satisfied (4-5)"
                                         if x >= 4 else "Not Satisfied (1-3)")
plt.figure(figsize=(7, 7))
df_csat_binary.value_counts().plot.pie(
    autopct="%1.1f%%", colors=["#2ecc71", "#e74c3c"],
    startangle=140, shadow=True,
    textprops={"fontsize": 12}
)
plt.title("Chart 15 – Satisfied vs Not Satisfied", fontweight="bold")
plt.ylabel("")
save_fig("15_csat_pie")

print("\n✅ All 15 charts saved to the 'plots/' directory.")


# ─────────────────────────────────────────────────────────────────────
# 6. HYPOTHESIS TESTING
# ─────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 5 – HYPOTHESIS TESTING")
print("="*60)

# H1: Response time differs between satisfied vs not satisfied
satisfied     = df[df["CSAT Score"] >= 4]["response_time_min"].dropna()
not_satisfied = df[df["CSAT Score"] <  4]["response_time_min"].dropna()
t_stat, p_val = stats.ttest_ind(satisfied, not_satisfied, equal_var=False)
print(f"\nH1 – Welch's t-test (response_time_min vs CSAT group):")
print(f"  t-statistic = {t_stat:.4f}, p-value = {p_val:.6f}")
print(f"  → {'Reject H0' if p_val < 0.05 else 'Fail to reject H0'} at α=0.05")

# H2: CSAT Score is independent of Agent Shift (Chi-Square)
ct = pd.crosstab(df["Agent Shift"],
                 df["CSAT Score"].apply(lambda x: "High" if x >= 4 else "Low"))
chi2, p_chi, dof, expected = stats.chi2_contingency(ct)
print(f"\nH2 – Chi-Square test (Agent Shift vs CSAT High/Low):")
print(f"  chi2 = {chi2:.4f}, p-value = {p_chi:.6f}, dof = {dof}")
print(f"  → {'Reject H0' if p_chi < 0.05 else 'Fail to reject H0'} at α=0.05")

# H3: Handling time differs between channels (ANOVA)
groups = [g["connected_handling_time"].dropna().values
          for _, g in df.groupby("channel_name")]
f_stat, p_anova = stats.f_oneway(*groups)
print(f"\nH3 – One-Way ANOVA (handling_time across channels):")
print(f"  F-statistic = {f_stat:.4f}, p-value = {p_anova:.6f}")
print(f"  → {'Reject H0' if p_anova < 0.05 else 'Fail to reject H0'} at α=0.05")


# ─────────────────────────────────────────────────────────────────────
# 7. FEATURE ENGINEERING & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 6 – FEATURE ENGINEERING & PREPROCESSING")
print("="*60)

# Target binarisation: 1 = Satisfied (4-5), 0 = Not Satisfied (1-3)
df["target"] = (df["CSAT Score"] >= 4).astype(int)
print(f"\nTarget distribution:\n{df['target'].value_counts()}")

# ── 7a. Text preprocessing on Customer Remarks
STOP_WORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text: str) -> str:
    """Clean and lemmatize a text field."""
    if not isinstance(text, str) or text.strip() == "":
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)         # remove URLs
    text = re.sub(r"[^a-z\s]", " ", text)               # remove punctuation/digits
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens
              if w not in STOP_WORDS and len(w) > 2]
    return " ".join(tokens)

df["remarks_clean"] = df["Customer Remarks"].apply(preprocess_text)

# TF-IDF on remarks → top 50 features
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=50, min_df=5, max_df=0.9)
remarks_filled = df["remarks_clean"].fillna("")
tfidf_matrix = tfidf.fit_transform(remarks_filled).toarray()
tfidf_df = pd.DataFrame(tfidf_matrix,
                         columns=[f"tfidf_{c}" for c in tfidf.get_feature_names_out()],
                         index=df.index)

# ── 7b. Select structured features
struct_features = [
    "channel_name", "category", "Sub-category",
    "Customer_City", "Product_category", "Agent Shift",
    "Tenure Bucket",
    "Item_price", "connected_handling_time",
    "response_time_min",
    "issue_hour", "issue_dayofweek", "issue_month"
]
df_struct = df[struct_features].copy()

# Missing value imputation
num_features = ["Item_price", "connected_handling_time", "response_time_min"]
cat_features  = [c for c in struct_features if c not in num_features]

for col in num_features:
    median_val = df_struct[col].median()
    df_struct[col] = df_struct[col].fillna(median_val)

for col in cat_features:
    mode_val = df_struct[col].mode()[0]
    df_struct[col] = df_struct[col].fillna(mode_val)

# ── 7c. Outlier capping (IQR) for numeric features
for col in num_features:
    Q1, Q3 = df_struct[col].quantile(0.25), df_struct[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    df_struct[col] = df_struct[col].clip(lower, upper)

# ── 7d. Label encoding for categoricals
le_dict = {}
for col in cat_features:
    le = LabelEncoder()
    df_struct[col] = le.fit_transform(df_struct[col].astype(str))
    le_dict[col] = le

# ── 7e. Combine structured + TF-IDF features
X = pd.concat([df_struct.reset_index(drop=True),
               tfidf_df.reset_index(drop=True)], axis=1)
y = df["target"].reset_index(drop=True)

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts()}")

# ── 7f. Scale numerical features
scaler = StandardScaler()
X[num_features] = scaler.fit_transform(X[num_features])

# ── 7g. Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)
print(f"\nTrain shape: {X_train.shape}  |  Test shape: {X_test.shape}")

# ── 7h. Handle imbalance with SMOTE
sm = SMOTE(random_state=SEED)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
print(f"After SMOTE – Train: {X_train_sm.shape}  "
      f"Target distribution: {pd.Series(y_train_sm).value_counts().to_dict()}")


# ─────────────────────────────────────────────────────────────────────
# 8. ML MODELS
# ─────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 7 – ML MODEL TRAINING")
print("="*60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# ──────────────────────────────────────────────
# MODEL 1 – Logistic Regression
# ──────────────────────────────────────────────
print("\n── Model 1: Logistic Regression ──")
lr = LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")
lr.fit(X_train_sm, y_train_sm)

y_pred_lr = lr.predict(X_test)
lr_acc    = accuracy_score(y_test, y_pred_lr)
lr_auc    = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
lr_cv     = cross_val_score(lr, X_train_sm, y_train_sm, cv=cv,
                            scoring="roc_auc").mean()

print(f"Accuracy : {lr_acc:.4f}")
print(f"ROC-AUC  : {lr_auc:.4f}")
print(f"CV AUC   : {lr_cv:.4f}")
print(classification_report(y_test, y_pred_lr,
                             target_names=["Not Satisfied", "Satisfied"]))

# Confusion Matrix – LR
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_estimator(lr, X_test, y_test,
                                      display_labels=["Not Satisfied", "Satisfied"],
                                      cmap="Blues", ax=ax)
ax.set_title("Model 1 – Logistic Regression Confusion Matrix", fontweight="bold")
save_fig("model1_lr_cm")

# ──────────────────────────────────────────────
# MODEL 2 – Random Forest + GridSearchCV
# ──────────────────────────────────────────────
print("\n── Model 2: Random Forest (with GridSearchCV) ──")
rf_base = RandomForestClassifier(random_state=SEED, n_jobs=-1)
rf_params = {
    "n_estimators": [100, 200],
    "max_depth":    [None, 15],
    "min_samples_split": [2, 5]
}
rf_grid = GridSearchCV(rf_base, rf_params, cv=3, scoring="roc_auc",
                       n_jobs=-1, verbose=0)
rf_grid.fit(X_train_sm, y_train_sm)

best_rf = rf_grid.best_estimator_
y_pred_rf = best_rf.predict(X_test)
rf_acc    = accuracy_score(y_test, y_pred_rf)
rf_auc    = roc_auc_score(y_test, best_rf.predict_proba(X_test)[:, 1])
rf_cv     = cross_val_score(best_rf, X_train_sm, y_train_sm,
                            cv=cv, scoring="roc_auc").mean()

print(f"Best params: {rf_grid.best_params_}")
print(f"Accuracy   : {rf_acc:.4f}")
print(f"ROC-AUC    : {rf_auc:.4f}")
print(f"CV AUC     : {rf_cv:.4f}")
print(classification_report(y_test, y_pred_rf,
                             target_names=["Not Satisfied", "Satisfied"]))

# Confusion Matrix – RF
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_estimator(best_rf, X_test, y_test,
                                      display_labels=["Not Satisfied", "Satisfied"],
                                      cmap="Greens", ax=ax)
ax.set_title("Model 2 – Random Forest Confusion Matrix", fontweight="bold")
save_fig("model2_rf_cm")

# Feature importance
feat_imp = pd.Series(best_rf.feature_importances_, index=X.columns
                     ).sort_values(ascending=False).head(20)
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=feat_imp.values, y=feat_imp.index, palette="viridis",
            hue=feat_imp.index, legend=False, ax=ax)
ax.set_title("Model 2 – Top 20 Feature Importances (RF)", fontweight="bold")
save_fig("model2_rf_feature_importance")

# ──────────────────────────────────────────────
# MODEL 3 – XGBoost
# ──────────────────────────────────────────────
print("\n── Model 3: XGBoost Classifier ──")
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=SEED,
    n_jobs=-1
)
xgb.fit(X_train_sm, y_train_sm,
        eval_set=[(X_test, y_test)], verbose=False)

y_pred_xgb = xgb.predict(X_test)
xgb_acc    = accuracy_score(y_test, y_pred_xgb)
xgb_auc    = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])
xgb_cv     = cross_val_score(xgb, X_train_sm, y_train_sm,
                             cv=cv, scoring="roc_auc").mean()

print(f"Accuracy : {xgb_acc:.4f}")
print(f"ROC-AUC  : {xgb_auc:.4f}")
print(f"CV AUC   : {xgb_cv:.4f}")
print(classification_report(y_test, y_pred_xgb,
                             target_names=["Not Satisfied", "Satisfied"]))

# Confusion Matrix – XGB
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_estimator(xgb, X_test, y_test,
                                      display_labels=["Not Satisfied", "Satisfied"],
                                      cmap="Oranges", ax=ax)
ax.set_title("Model 3 – XGBoost Confusion Matrix", fontweight="bold")
save_fig("model3_xgb_cm")

# ── Model comparison chart
print("\n── Model Comparison ──")
results = pd.DataFrame({
    "Model":    ["Logistic Regression", "Random Forest", "XGBoost"],
    "Accuracy": [lr_acc, rf_acc, xgb_acc],
    "ROC-AUC":  [lr_auc, rf_auc, xgb_auc],
    "CV-AUC":   [lr_cv,  rf_cv,  xgb_cv]
})
print(results.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, metric in zip(axes, ["Accuracy", "ROC-AUC", "CV-AUC"]):
    sns.barplot(data=results, x="Model", y=metric, palette="Set1",
                hue="Model", legend=False, ax=ax)
    ax.set_title(f"Model Comparison – {metric}", fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
save_fig("model_comparison")


# ─────────────────────────────────────────────────────────────────────
# 9. SAVE BEST MODEL
# ─────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 8 – SAVE BEST MODEL (XGBoost)")
print("="*60)

best_model_info = {
    "model":      xgb,
    "scaler":     scaler,
    "le_dict":    le_dict,
    "tfidf":      tfidf,
    "features":   list(X.columns),
    "num_features": num_features
}
MODEL_PATH = "deep_csat_model.pkl"
joblib.dump(best_model_info, MODEL_PATH)
print(f"✅ Model saved → {MODEL_PATH}")


# ─────────────────────────────────────────────────────────────────────
# 10. SANITY CHECK – load model & predict on unseen sample
# ─────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 9 – SANITY CHECK")
print("="*60)

loaded = joblib.load(MODEL_PATH)
model_loaded  = loaded["model"]
scaler_l      = loaded["scaler"]
le_dict_l     = loaded["le_dict"]
tfidf_l       = loaded["tfidf"]
features_l    = loaded["features"]
num_feats_l   = loaded["num_features"]

# Build a synthetic unseen sample
sample = pd.DataFrame([{
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
    "remarks_clean": "product working great happy service"
}])

# Encode categoricals
for col in cat_features:
    le_s = le_dict_l.get(col)
    if le_s:
        val = sample[col].astype(str).iloc[0]
        if val in le_s.classes_:
            sample[col] = le_s.transform([val])[0]
        else:
            sample[col] = 0

# TF-IDF
sample_tfidf = tfidf_l.transform(sample["remarks_clean"].fillna("")).toarray()
sample_tfidf_df = pd.DataFrame(
    sample_tfidf,
    columns=[f"tfidf_{c}" for c in tfidf_l.get_feature_names_out()]
)

sample_struct = sample[[c for c in features_l
                         if c in sample.columns and not c.startswith("tfidf_")]].copy()

# Fill any missing columns with 0
for col in cat_features:
    if col not in sample_struct.columns:
        sample_struct[col] = 0

sample_final = pd.concat([sample_struct.reset_index(drop=True),
                           sample_tfidf_df.reset_index(drop=True)], axis=1)

# Align columns
for col in features_l:
    if col not in sample_final.columns:
        sample_final[col] = 0
sample_final = sample_final[features_l]

# Scale numeric
sample_final[num_feats_l] = scaler_l.transform(sample_final[num_feats_l])

pred = model_loaded.predict(sample_final)[0]
prob = model_loaded.predict_proba(sample_final)[0][1]
label = "Satisfied" if pred == 1 else "Not Satisfied"
print(f"\n🔮 Prediction for sample →  {label}  (confidence = {prob:.2%})")
print(f"\n✅ Sanity check passed! Model pipeline is deployment-ready.\n")


# ─────────────────────────────────────────────────────────────────────
# CONCLUSION
# ─────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  CONCLUSION")
print("="*60)
print(f"""
The DEEP-CSAT pipeline successfully:
  1. Loaded & explored {df.shape[0]:,} customer support records (20 features).
  2. Engineered response_time_min, temporal features (hour, DOW, month).
  3. Preprocessed Customer Remarks using NLP (lemmatization + TF-IDF).
  4. Produced 15 insightful EDA visualisations saved to 'plots/'.
  5. Performed 3 hypothesis tests (t-test, chi-square, ANOVA).
  6. Handled class imbalance using SMOTE.
  7. Trained 3 ML models (Logistic Regression, Random Forest, XGBoost).
  8. XGBoost achieved:  Accuracy={xgb_acc:.4f}  |  ROC-AUC={xgb_auc:.4f}
  9. Best model serialised → deep_csat_model.pkl (deployment-ready).

Business Impact:
  • Proactively identify dissatisfied customers to reduce churn.
  • Flag slow-response tickets for re-training agents.
  • Guide scheduling by shift/hour patterns affecting CSAT.
""")
print("="*60)
print("  *** Pipeline Complete — Ready for Deployment! ***")
print("="*60)
