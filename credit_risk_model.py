# =============================================================================
# Credit Risk & Fraud Strategy Analytics Pipeline
# Dataset: Home Credit Default Risk (Kaggle) — application_train.csv
# Goal: End-to-end fraud strategy: rules, models, monitoring, scorecards
# =============================================================================

import os
import logging
import warnings
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    precision_recall_curve, roc_curve, f1_score, precision_score, recall_score
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap

warnings.filterwarnings("ignore")

# =============================================================================
# SECTION 1: Configuration
# =============================================================================

CONFIG = {
    "DATA_PATH": "application_train.csv",
    "TARGET_COL": "TARGET",
    "RANDOM_SEED": 42,
    "TEST_SIZE": 0.2,

    "XGB_PARAMS": {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "aucpr",
        "random_state": 42,
        "verbosity": 0,
    },
    "RF_PARAMS": {
        "n_estimators": 200,
        "max_depth": 8,
        "random_state": 42,
        "n_jobs": -1,
    },

    # Business cost matrix — false negatives (missed defaults) cost far more
    # than false positives (declined good customers) in credit/fraud contexts
    "COST_FN": 5000,
    "COST_FP": 200,

    # Operational decision threshold
    "DECISION_THRESHOLD": 0.3,

    # Rule-based strategy thresholds
    "RULE_INCOME_MIN": 45_000,
    "RULE_CREDIT_RATIO_MAX": 0.45,
    "RULE_EXT_SOURCE_MIN": 0.35,
    "RULE_PEER_DEFAULTS_MAX": 2,

    # Risk score tiers for operational scorecard
    "SCORE_BINS": [0, 0.10, 0.25, 0.50, 1.01],
    "SCORE_LABELS": ["Low", "Medium", "High", "Very High"],
    "SCORE_ACTIONS": {
        "Low": "Auto-Approve",
        "Medium": "Manual Review",
        "High": "Decline",
        "Very High": "Decline + Alert",
    },

    "OUTPUT_DIR": "outputs",
    "LOG_FILE": "outputs/credit_risk_model.log",
}

CATEGORICAL_COLS = [
    "CODE_GENDER",
    "NAME_CONTRACT_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_INCOME_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
]

NUMERIC_COLS = [
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH",
    "REGION_POPULATION_RELATIVE",
    "CNT_FAM_MEMBERS", "CNT_CHILDREN",
    "HOUR_APPR_PROCESS_START",
    "OWN_CAR_AGE",
    "DEF_30_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE",
    "OBS_30_CNT_SOCIAL_CIRCLE", "OBS_60_CNT_SOCIAL_CIRCLE",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
]

EDUCATION_ORDER = {
    "Lower secondary": 0,
    "Secondary / secondary special": 1,
    "Incomplete higher": 2,
    "Higher education": 3,
    "Academic degree": 4,
}


def setup_logging(cfg: dict) -> logging.Logger:
    os.makedirs(cfg["OUTPUT_DIR"], exist_ok=True)
    logger = logging.getLogger("credit_risk")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s", "%H:%M:%S")

    fh = logging.FileHandler(cfg["LOG_FILE"], mode="w")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# =============================================================================
# SECTION 2: Data Loading & EDA
# =============================================================================

def load_data(cfg: dict, logger: logging.Logger) -> pd.DataFrame:
    logger.info("Loading data from %s", cfg["DATA_PATH"])
    df = pd.read_csv(cfg["DATA_PATH"])
    default_rate = df[cfg["TARGET_COL"]].mean() * 100
    logger.info("Shape: %s | Default rate: %.2f%% | Memory: %.1f MB",
                df.shape, default_rate, df.memory_usage(deep=True).sum() / 1e6)
    return df


def run_eda(df: pd.DataFrame, cfg: dict, logger: logging.Logger) -> None:
    logger.info("Running EDA...")
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle("Home Credit — Exploratory Data Analysis", fontsize=14, fontweight="bold")

    # Missing value heatmap (top 20 columns by missingness)
    ax = axes[0, 0]
    miss_pct = df.isnull().mean().sort_values(ascending=False).head(20)
    ax.barh(miss_pct.index[::-1], miss_pct.values[::-1], color="steelblue")
    ax.set_xlabel("Missing fraction")
    ax.set_title("Top 20 Columns by Missingness")
    ax.set_xlim(0, 1)

    # Default rate by CODE_GENDER
    ax = axes[0, 1]
    if "CODE_GENDER" in df.columns:
        dr = df.groupby("CODE_GENDER")["TARGET"].mean().sort_values()
        ax.bar(dr.index, dr.values * 100, color=["steelblue", "tomato", "gray"][:len(dr)])
        ax.set_ylabel("Default Rate (%)")
        ax.set_title("Default Rate by Gender")

    # Default rate by NAME_EDUCATION_TYPE
    ax = axes[0, 2]
    if "NAME_EDUCATION_TYPE" in df.columns:
        dr = df.groupby("NAME_EDUCATION_TYPE")["TARGET"].mean().sort_values()
        ax.barh(dr.index, dr.values * 100, color="coral")
        ax.set_xlabel("Default Rate (%)")
        ax.set_title("Default Rate by Education")

    # Default rate by NAME_CONTRACT_TYPE
    ax = axes[0, 3]
    if "NAME_CONTRACT_TYPE" in df.columns:
        dr = df.groupby("NAME_CONTRACT_TYPE")["TARGET"].mean().sort_values()
        ax.bar(dr.index, dr.values * 100, color="mediumpurple")
        ax.set_ylabel("Default Rate (%)")
        ax.set_title("Default Rate by Contract Type")

    # AMT_INCOME_TOTAL distribution
    ax = axes[1, 0]
    for label, grp in df.groupby("TARGET"):
        vals = grp["AMT_INCOME_TOTAL"].clip(upper=500_000).dropna()
        ax.hist(vals, bins=50, alpha=0.5, density=True,
                label="Default" if label == 1 else "No Default")
    ax.set_xlabel("Annual Income (clipped at 500k)")
    ax.set_title("Income Distribution")
    ax.legend(fontsize=8)

    # AMT_CREDIT distribution
    ax = axes[1, 1]
    for label, grp in df.groupby("TARGET"):
        vals = grp["AMT_CREDIT"].clip(upper=2_000_000).dropna()
        ax.hist(vals, bins=50, alpha=0.5, density=True,
                label="Default" if label == 1 else "No Default")
    ax.set_xlabel("Loan Amount (clipped at 2M)")
    ax.set_title("Loan Amount Distribution")
    ax.legend(fontsize=8)

    # DAYS_EMPLOYED histogram — expose 365243 sentinel
    ax = axes[1, 2]
    days = df["DAYS_EMPLOYED"].dropna()
    ax.hist(days.clip(-5000, 0), bins=60, color="darkgreen", alpha=0.7)
    anomaly_count = (df["DAYS_EMPLOYED"] == 365243).sum()
    ax.set_xlabel("Days Employed (negative = past)")
    ax.set_title(f"Employment Tenure\n(365243 anomaly: {anomaly_count:,} rows excluded)")

    # Correlation with TARGET
    ax = axes[1, 3]
    num_cols = df.select_dtypes(include="number").columns.tolist()
    corr = df[num_cols].corr()["TARGET"].drop("TARGET").abs().nlargest(12)
    ax.barh(corr.index[::-1], corr.values[::-1], color="teal")
    ax.set_xlabel("|Pearson Correlation| with TARGET")
    ax.set_title("Top 12 Features Correlated with Default")

    plt.tight_layout()
    out_path = os.path.join(cfg["OUTPUT_DIR"], "eda_plots.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    logger.info("EDA plot saved to %s", out_path)


# =============================================================================
# SECTION 3: Feature Engineering
# =============================================================================

def engineer_features(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    logger.info("Engineering features...")
    df = df.copy()

    # --- Anomaly flag: 365243 encodes unemployed/pensioner status ---
    df["DAYS_EMPLOYED_ANOMALY"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

    # --- Debt burden ratios (core credit risk KPIs) ---
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1)
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1)
    df["CREDIT_GOODS_RATIO"] = df["AMT_CREDIT"] / (df["AMT_GOODS_PRICE"] + 1)
    df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / (df["CNT_FAM_MEMBERS"] + 1)

    # --- Age and employment stability ---
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365
    df["EMPLOYMENT_YEARS"] = -df["DAYS_EMPLOYED"] / 365
    df["EMPLOYMENT_STABILITY_FLAG"] = (df["EMPLOYMENT_YEARS"] > 2).astype(int)
    df["AGE_EMPLOYMENT_RATIO"] = df["EMPLOYMENT_YEARS"] / (df["AGE_YEARS"] + 1)

    # --- External credit bureau score aggregates (most predictive signals) ---
    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    available_ext = [c for c in ext_cols if c in df.columns]
    if available_ext:
        df["EXT_SOURCE_MEAN"] = df[available_ext].mean(axis=1)
        df["EXT_SOURCE_MIN"] = df[available_ext].min(axis=1)
        df["EXT_SOURCE_STD"] = df[available_ext].std(axis=1).fillna(0)
        df["EXT_SOURCE_2_MISSING"] = df["EXT_SOURCE_2"].isnull().astype(int)

    # --- Social network risk indicator (fraud ring signal) ---
    if "DEF_30_CNT_SOCIAL_CIRCLE" in df.columns and "OBS_30_CNT_SOCIAL_CIRCLE" in df.columns:
        df["SOCIAL_CIRCLE_DEFAULT_RATE"] = (
            df["DEF_30_CNT_SOCIAL_CIRCLE"] / (df["OBS_30_CNT_SOCIAL_CIRCLE"] + 1)
        )

    # --- Categorical encoding ---
    # Ordinal encode education (meaningful ordering)
    if "NAME_EDUCATION_TYPE" in df.columns:
        df["EDUCATION_ORDINAL"] = df["NAME_EDUCATION_TYPE"].map(EDUCATION_ORDER).fillna(1)

    # Frequency encode high-cardinality categoricals
    for col in ["OCCUPATION_TYPE", "NAME_INCOME_TYPE", "NAME_FAMILY_STATUS",
                "NAME_HOUSING_TYPE"]:
        if col in df.columns:
            freq = df[col].value_counts(normalize=True)
            df[col + "_FREQ"] = df[col].map(freq).fillna(0)

    # Label encode binary/low-cardinality categoricals
    for col in ["CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "NAME_CONTRACT_TYPE"]:
        if col in df.columns:
            le = LabelEncoder()
            df[col + "_ENC"] = le.fit_transform(df[col].astype(str))

    # --- Impute numeric NaNs with median ---
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    logger.info("Feature engineering complete. Shape: %s", df.shape)
    return df


def get_feature_list(df: pd.DataFrame) -> list:
    exclude = {"TARGET", "SK_ID_CURR"}
    exclude |= set(CATEGORICAL_COLS)
    exclude |= {"NAME_INCOME_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE",
                "OCCUPATION_TYPE", "NAME_EDUCATION_TYPE"}
    return [c for c in df.select_dtypes(include="number").columns if c not in exclude]


# =============================================================================
# SECTION 4: Rule-Based Fraud Strategy
# =============================================================================
# INTERVIEW NOTE: In production, rules run at the API layer before the model
# scores. Rules are cheaper to audit, explain to regulators, and deploy than
# ML models. They form the baseline strategy that a score model improves upon.

def apply_rule_strategy(df: pd.DataFrame, cfg: dict) -> pd.Series:
    rules = pd.DataFrame(index=df.index)
    rules["r1_low_income"] = (df["AMT_INCOME_TOTAL"] < cfg["RULE_INCOME_MIN"]).astype(int)
    rules["r2_high_debt"] = (df.get("CREDIT_INCOME_RATIO", pd.Series(0, index=df.index))
                             > cfg["RULE_CREDIT_RATIO_MAX"]).astype(int)
    rules["r3_unemployed"] = df.get("DAYS_EMPLOYED_ANOMALY",
                                     pd.Series(0, index=df.index)).astype(int)
    rules["r4_low_bureau"] = (df.get("EXT_SOURCE_MEAN", pd.Series(1, index=df.index))
                               < cfg["RULE_EXT_SOURCE_MIN"]).astype(int)
    rules["r5_peer_defaults"] = (
        df.get("DEF_30_CNT_SOCIAL_CIRCLE", pd.Series(0, index=df.index))
        >= cfg["RULE_PEER_DEFAULTS_MAX"]
    ).astype(int)
    return rules


def evaluate_rule_strategy(df: pd.DataFrame, cfg: dict,
                            logger: logging.Logger) -> dict:
    logger.info("Evaluating rule-based strategy...")
    rules = apply_rule_strategy(df, cfg)
    y_true = df["TARGET"]

    rule_names = {
        "r1_low_income": f"Income < ${cfg['RULE_INCOME_MIN']:,}",
        "r2_high_debt": f"Credit/Income > {cfg['RULE_CREDIT_RATIO_MAX']}",
        "r3_unemployed": "Unemployed/Pensioner flag",
        "r4_low_bureau": f"Bureau score < {cfg['RULE_EXT_SOURCE_MIN']}",
        "r5_peer_defaults": f"Peer defaults >= {cfg['RULE_PEER_DEFAULTS_MAX']}",
    }

    print("\n" + "=" * 70)
    print("RULE-BASED STRATEGY EVALUATION")
    print("=" * 70)
    print(f"{'Rule':<35} {'Hit%':>6} {'Precision':>10} {'Recall':>8}")
    print("-" * 70)

    rule_metrics = {}
    for rule_col, rule_label in rule_names.items():
        flagged = rules[rule_col]
        hit_rate = flagged.mean() * 100
        if flagged.sum() > 0:
            prec = precision_score(y_true, flagged, zero_division=0)
            rec = recall_score(y_true, flagged, zero_division=0)
        else:
            prec = rec = 0.0
        rule_metrics[rule_col] = {"hit_rate": hit_rate, "precision": prec, "recall": rec}
        print(f"  {rule_label:<33} {hit_rate:>5.1f}%  {prec:>9.3f}  {rec:>7.3f}")

    combined = (rules.sum(axis=1) >= 1).astype(int)
    comb_hit = combined.mean() * 100
    comb_prec = precision_score(y_true, combined, zero_division=0)
    comb_rec = recall_score(y_true, combined, zero_division=0)
    comb_f1 = f1_score(y_true, combined, zero_division=0)
    approval_rate = 1 - combined.mean()

    print("-" * 70)
    print(f"  {'ANY rule triggered (OR logic)':<33} {comb_hit:>5.1f}%  "
          f"{comb_prec:>9.3f}  {comb_rec:>7.3f}")
    print("=" * 70)
    print(f"  Approval rate with rules only: {approval_rate:.1%}")
    print(f"  Combined F1: {comb_f1:.4f}")
    print(f"\n  Interpretation: Rules alone catch {comb_rec:.1%} of defaults while")
    print(f"  declining {comb_hit:.1f}% of all applicants. A score model layered on")
    print(f"  top improves recall while maintaining a manageable false positive rate.")
    print("=" * 70 + "\n")

    return {
        "y_pred": combined,
        "precision": comb_prec,
        "recall": comb_rec,
        "f1": comb_f1,
        "approval_rate": approval_rate,
        "pr_auc": average_precision_score(y_true, combined),
        "individual": rule_metrics,
    }


# =============================================================================
# SECTION 5: Model Training
# =============================================================================

def prepare_model_data(df: pd.DataFrame, cfg: dict, logger: logging.Logger):
    features = get_feature_list(df)
    X = df[features]
    y = df[cfg["TARGET_COL"]]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["TEST_SIZE"],
        random_state=cfg["RANDOM_SEED"], stratify=y
    )
    logger.info("Train: %d rows | Test: %d rows | Features: %d",
                len(X_train), len(X_test), len(features))
    logger.info("Train defaults: %d (%.2f%%) | Test defaults: %d (%.2f%%)",
                y_train.sum(), y_train.mean() * 100,
                y_test.sum(), y_test.mean() * 100)
    return X_train, X_test, y_train, y_test


def train_logistic_regression(X_train, y_train, cfg, logger):
    logger.info("Training Logistic Regression...")
    t0 = time.time()
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=0.1, class_weight="balanced",
                                   max_iter=1000, random_state=cfg["RANDOM_SEED"]))
    ])
    model.fit(X_train, y_train)
    logger.info("  Logistic Regression done in %.1fs", time.time() - t0)
    return model


def train_random_forest(X_train, y_train, cfg, logger):
    logger.info("Training Random Forest...")
    t0 = time.time()
    model = RandomForestClassifier(
        class_weight="balanced_subsample", **cfg["RF_PARAMS"]
    )
    model.fit(X_train, y_train)
    logger.info("  Random Forest done in %.1fs", time.time() - t0)
    return model


def train_xgb_weighted(X_train, y_train, cfg, logger):
    logger.info("Training XGBoost (scale_pos_weight) — CHAMPION...")
    t0 = time.time()
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    params = {**cfg["XGB_PARAMS"], "scale_pos_weight": neg / pos}
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    logger.info("  XGB Weighted done in %.1fs", time.time() - t0)
    return model


def train_xgb_smote(X_train, y_train, cfg, logger):
    logger.info("Training XGBoost + SMOTE — CHALLENGER...")
    t0 = time.time()
    sample_size = min(50_000, len(X_train))
    X_s, y_s = X_train.iloc[:sample_size], y_train.iloc[:sample_size]
    smote = SMOTE(random_state=cfg["RANDOM_SEED"], n_jobs=-1)
    X_res, y_res = smote.fit_resample(X_s, y_s)
    logger.info("  After SMOTE — 0: %d, 1: %d", (y_res == 0).sum(), (y_res == 1).sum())
    model = xgb.XGBClassifier(**cfg["XGB_PARAMS"])
    model.fit(X_res, y_res)
    logger.info("  XGB SMOTE done in %.1fs", time.time() - t0)
    return model


def train_all_models(X_train, y_train, cfg, logger):
    return {
        "Logistic Regression": train_logistic_regression(X_train, y_train, cfg, logger),
        "Random Forest": train_random_forest(X_train, y_train, cfg, logger),
        "XGB Weighted [Champion]": train_xgb_weighted(X_train, y_train, cfg, logger),
        "XGB SMOTE [Challenger]": train_xgb_smote(X_train, y_train, cfg, logger),
    }


# =============================================================================
# SECTION 6: Model Evaluation Suite
# =============================================================================

def compute_ks_statistic(y_true: pd.Series, y_score: np.ndarray) -> float:
    df_ks = pd.DataFrame({"score": y_score, "target": y_true}).sort_values("score", ascending=False)
    total_pos = y_true.sum()
    total_neg = (y_true == 0).sum()
    df_ks["cum_pos"] = df_ks["target"].cumsum() / total_pos
    df_ks["cum_neg"] = (1 - df_ks["target"]).cumsum() / total_neg
    return (df_ks["cum_pos"] - df_ks["cum_neg"]).abs().max() * 100


def evaluate_model(model, X_test, y_test, name, cfg) -> dict:
    y_score = model.predict_proba(X_test)[:, 1]
    threshold = cfg["DECISION_THRESHOLD"]
    y_pred = (y_score >= threshold).astype(int)

    roc_auc = roc_auc_score(y_test, y_score)
    pr_auc = average_precision_score(y_test, y_score)
    ks = compute_ks_statistic(y_test, y_score)
    gini = 2 * roc_auc - 1
    fpr, tpr, _ = roc_curve(y_test, y_score)
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_score)

    return {
        "model_name": name,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "ks_statistic": ks,
        "gini": gini,
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "approval_rate": 1 - y_pred.mean(),
        "fpr": fpr,
        "tpr": tpr,
        "prec_curve": prec_curve,
        "rec_curve": rec_curve,
        "y_score": y_score,
    }


def evaluate_all_models(models, X_test, y_test, cfg, logger) -> dict:
    logger.info("Evaluating all models...")
    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(model, X_test, y_test, name, cfg)

    print("\n" + "=" * 85)
    print("MODEL EVALUATION SUMMARY")
    print("=" * 85)
    print(f"{'Model':<30} {'AUC':>7} {'AP':>7} {'KS':>7} {'Gini':>7} "
          f"{'Prec':>7} {'Recall':>7} {'Appr%':>7}")
    print("-" * 85)
    for name, r in results.items():
        marker = " *" if "Champion" in name else ""
        print(f"  {name + marker:<28} {r['roc_auc']:>7.4f} {r['pr_auc']:>7.4f} "
              f"{r['ks_statistic']:>7.2f} {r['gini']:>7.4f} "
              f"{r['precision']:>7.3f} {r['recall']:>7.3f} "
              f"{r['approval_rate'] * 100:>6.1f}%")
    print("=" * 85 + "\n")
    return results


# =============================================================================
# SECTION 7: Business-Driven Threshold Optimization
# =============================================================================
# INTERVIEW NOTE: AUC optimization is a model metric. Threshold selection is
# a business decision. The cost matrix encodes the firm's actual loss tolerance
# and drives the operational cutoff — not an arbitrary 0.5 default.

def optimize_threshold(model, X_test, y_test, cfg, logger) -> dict:
    logger.info("Optimizing decision threshold via cost matrix...")
    y_score = model.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.05, 0.96, 0.01)
    rows = []
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        fn = ((y_pred == 0) & (y_test == 1)).sum()
        fp = ((y_pred == 1) & (y_test == 0)).sum()
        total_cost = fn * cfg["COST_FN"] + fp * cfg["COST_FP"]
        approval = 1 - y_pred.mean()
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        rows.append({
            "threshold": t, "total_cost": total_cost,
            "approval_rate": approval, "precision": prec,
            "recall": rec, "f1": f1, "fn": fn, "fp": fp
        })

    tdf = pd.DataFrame(rows)
    opt_cost = tdf.loc[tdf["total_cost"].idxmin(), "threshold"]
    opt_f1 = tdf.loc[tdf["f1"].idxmax(), "threshold"]

    # Precision >= 0.6 constrained optimal recall (regulatory scenario)
    constrained = tdf[tdf["precision"] >= 0.6]
    opt_constrained = constrained.loc[constrained["recall"].idxmax(), "threshold"] \
        if not constrained.empty else opt_cost

    opt_row = tdf[tdf["threshold"].round(2) == round(opt_cost, 2)].iloc[0]
    baseline_cost = tdf[tdf["threshold"] == 0.5].iloc[0]["total_cost"] \
        if not tdf[tdf["threshold"] == 0.5].empty else tdf["total_cost"].max()
    savings = (baseline_cost - opt_row["total_cost"]) / len(y_test) * 10_000

    print("\n" + "=" * 65)
    print("THRESHOLD OPTIMIZATION (Cost Matrix: FN=$5k, FP=$200)")
    print("=" * 65)
    print(f"  Min-cost threshold    : {opt_cost:.2f}")
    print(f"  F1-optimal threshold  : {opt_f1:.2f}")
    print(f"  Prec>=0.6 constrained : {opt_constrained:.2f}")
    print(f"\n  At threshold={opt_cost:.2f}:")
    print(f"    Approval rate : {opt_row['approval_rate']:.1%}")
    print(f"    Recall        : {opt_row['recall']:.1%}  (defaults caught)")
    print(f"    Precision     : {opt_row['precision']:.1%}")
    print(f"    Est. savings vs. threshold=0.5: ${savings:,.0f} per 10,000 applicants")
    print("=" * 65 + "\n")

    logger.info("Optimal threshold (cost): %.2f | Savings est: $%.0f per 10k", opt_cost, savings)
    return {
        "threshold_df": tdf,
        "opt_cost": opt_cost,
        "opt_f1": opt_f1,
        "opt_constrained": opt_constrained,
        "y_score": y_score,
    }


# =============================================================================
# SECTION 8: SHAP Feature Interpretability
# =============================================================================

def run_shap_analysis(model, X_train, X_test, cfg, logger) -> np.ndarray:
    logger.info("Running SHAP analysis (n=1000 sample)...")
    sample = X_test.iloc[:1000]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    # Summary beeswarm plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, sample, show=False, max_display=15)
    plt.title("SHAP Feature Importance — XGB Champion Model", fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(cfg["OUTPUT_DIR"], "shap_summary.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    logger.info("SHAP summary plot saved to %s", out_path)

    # Top features table with fraud typology mapping
    mean_shap = np.abs(shap_values).mean(axis=0)
    top_features = pd.Series(mean_shap, index=sample.columns).nlargest(10)

    fraud_typology = {
        "EXT_SOURCE_MEAN": "Aggregated bureau score — low score = thin-file / synthetic identity",
        "EXT_SOURCE_2": "Primary bureau score",
        "EXT_SOURCE_MIN": "Worst bureau score — flags high-risk outliers",
        "EXT_SOURCE_3": "Secondary bureau score",
        "EXT_SOURCE_1": "Tertiary bureau score",
        "CREDIT_INCOME_RATIO": "Debt burden — high ratio = loan stacking / bust-out fraud",
        "DAYS_EMPLOYED": "Employment tenure — short tenure = higher default risk",
        "AGE_YEARS": "Applicant age — younger applicants historically higher risk",
        "AMT_CREDIT": "Loan amount — larger loans = higher loss severity",
        "AMT_ANNUITY": "Monthly obligation — high annuity strains cash flow",
        "EMPLOYMENT_STABILITY_FLAG": "Stable employment indicator",
        "SOCIAL_CIRCLE_DEFAULT_RATE": "Peer default rate — fraud ring signal",
        "DAYS_EMPLOYED_ANOMALY": "Employment misrepresentation flag",
        "CREDIT_GOODS_RATIO": "LTV proxy — high ratio = overextended borrower",
    }

    print("\n" + "=" * 75)
    print("TOP 10 SHAP FEATURES — Fraud Typology Mapping")
    print("=" * 75)
    for feat, val in top_features.items():
        typology = fraud_typology.get(feat, "Risk indicator")
        print(f"  {feat:<30} SHAP={val:.4f}  |  {typology}")
    print("=" * 75 + "\n")

    return mean_shap


# =============================================================================
# SECTION 9: Risk Score Bucketing & Operational Scorecard
# =============================================================================

def build_risk_scorecard(model, X_test, y_test, cfg, logger) -> pd.DataFrame:
    logger.info("Building risk scorecard...")
    y_score = model.predict_proba(X_test)[:, 1]
    score_df = pd.DataFrame({
        "score": y_score,
        "target": y_test.values,
        "AMT_CREDIT": X_test["AMT_CREDIT"].values if "AMT_CREDIT" in X_test.columns else 0,
    })
    score_df["tier"] = pd.cut(
        score_df["score"],
        bins=cfg["SCORE_BINS"],
        labels=cfg["SCORE_LABELS"],
        right=False
    )

    rows = []
    cum_bad = 0
    total_bad = score_df["target"].sum()

    for tier in reversed(cfg["SCORE_LABELS"]):
        g = score_df[score_df["tier"] == tier]
        n = len(g)
        if n == 0:
            continue
        dr = g["target"].mean()
        cum_bad += g["target"].sum()
        avg_credit = g["AMT_CREDIT"].mean() if "AMT_CREDIT" in g else 0
        exp_loss = dr * avg_credit
        rows.append({
            "Risk Tier": tier,
            "Count": n,
            "% Pop": f"{n / len(score_df):.1%}",
            "Default Rate": f"{dr:.1%}",
            "Action": cfg["SCORE_ACTIONS"][tier],
            "Exp Loss/App": f"${exp_loss:,.0f}",
            "Cum Bad Capture": f"{cum_bad / total_bad:.1%}",
        })

    scorecard = pd.DataFrame(rows)

    print("\n" + "=" * 80)
    print("OPERATIONAL RISK SCORECARD")
    print("=" * 80)
    print(scorecard.to_string(index=False))
    print("=" * 80 + "\n")
    logger.info("Scorecard built.")
    return scorecard


# =============================================================================
# SECTION 10: Population Stability Index (PSI) Monitoring
# =============================================================================
# INTERVIEW NOTE: PSI is monitored monthly in production. PSI > 0.2 triggers
# a model review ticket. Feature-level PSI (feature drift) is also monitored
# separately from score-level PSI — this example shows score-level monitoring.

def calculate_psi(expected: np.ndarray, actual: np.ndarray,
                  n_bins: int = 10) -> float:
    bins = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    bins[0] -= 1e-4
    bins[-1] += 1e-4
    eps = 1e-4

    exp_counts = np.histogram(expected, bins=bins)[0]
    act_counts = np.histogram(actual, bins=bins)[0]
    exp_pct = exp_counts / exp_counts.sum() + eps
    act_pct = act_counts / act_counts.sum() + eps

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def psi_status(psi_val: float) -> str:
    if psi_val < 0.1:
        return "STABLE — no action required"
    elif psi_val < 0.2:
        return "MODERATE SHIFT — monitor closely"
    return "ALERT — model recalibration recommended"


def calculate_psi_monitoring(model, X_train, X_test, cfg, logger) -> dict:
    logger.info("Running PSI monitoring simulation...")
    dev_scores = model.predict_proba(X_train.iloc[:10_000])[:, 1]
    prod_scores = model.predict_proba(X_test)[:, 1]

    # Simulate economic stress: shift feature values up 10%
    numeric_feats = X_test.select_dtypes(include="number").columns
    X_stressed = X_test.copy()
    X_stressed[numeric_feats] = X_stressed[numeric_feats] * 1.10
    stressed_scores = model.predict_proba(X_stressed)[:, 1]

    psi_normal = calculate_psi(dev_scores, prod_scores)
    psi_stressed = calculate_psi(dev_scores, stressed_scores)

    print("\n" + "=" * 60)
    print("MONITORING REPORT — Score Distribution Stability (PSI)")
    print("=" * 60)
    print(f"  Dev vs Prod PSI    : {psi_normal:.4f}  — {psi_status(psi_normal)}")
    print(f"  Dev vs Stressed PSI: {psi_stressed:.4f}  — {psi_status(psi_stressed)}")
    print("  Thresholds: <0.10 Stable | 0.10-0.20 Monitor | >0.20 Recalibrate")
    print("=" * 60 + "\n")
    logger.info("PSI normal: %.4f | PSI stressed: %.4f", psi_normal, psi_stressed)

    return {
        "psi_normal": psi_normal,
        "psi_stressed": psi_stressed,
        "dev_scores": dev_scores,
        "prod_scores": prod_scores,
        "stressed_scores": stressed_scores,
        "status_normal": psi_status(psi_normal),
        "status_stressed": psi_status(psi_stressed),
    }


# =============================================================================
# SECTION 11: Champion-Challenger Analysis
# =============================================================================
# INTERVIEW NOTE: In a real C/C test, 10-20% of traffic goes to the challenger
# model. We monitor for 4-8 weeks, checking approval rate delta, loss rate
# delta, and customer experience metrics before promoting to champion.

def champion_challenger_analysis(models, rule_results, eval_results,
                                  cfg, logger) -> pd.DataFrame:
    logger.info("Building champion-challenger comparison table...")
    rows = []

    # Rule-based baseline row
    rows.append({
        "Strategy": "Rule-Based Only",
        "AUC": "N/A",
        "AP": f"{rule_results['pr_auc']:.4f}",
        "KS": "N/A",
        "Gini": "N/A",
        "Precision": f"{rule_results['precision']:.3f}",
        "Recall": f"{rule_results['recall']:.3f}",
        "F1": f"{rule_results['f1']:.4f}",
        "Approval%": f"{rule_results['approval_rate']:.1%}",
    })

    for name, r in eval_results.items():
        label = name
        rows.append({
            "Strategy": label,
            "AUC": f"{r['roc_auc']:.4f}",
            "AP": f"{r['pr_auc']:.4f}",
            "KS": f"{r['ks_statistic']:.2f}",
            "Gini": f"{r['gini']:.4f}",
            "Precision": f"{r['precision']:.3f}",
            "Recall": f"{r['recall']:.3f}",
            "F1": f"{r['f1']:.4f}",
            "Approval%": f"{r['approval_rate']:.1%}",
        })

    cc_df = pd.DataFrame(rows)

    print("\n" + "=" * 100)
    print("CHAMPION-CHALLENGER ANALYSIS")
    print("=" * 100)
    print(cc_df.to_string(index=False))

    champ = eval_results.get("XGB Weighted [Champion]", {})
    chall = eval_results.get("XGB SMOTE [Challenger]", {})
    if champ and chall:
        auc_delta = champ["roc_auc"] - chall["roc_auc"]
        prec_delta = champ["precision"] - chall["precision"]
        rec_delta = chall["recall"] - champ["recall"]
        print(f"\n  Decision: Champion XGB Weighted selected.")
        print(f"  AUC advantage over Challenger: +{auc_delta:.4f}")
        print(f"  Precision advantage: +{prec_delta:.3f} (less friction for good customers)")
        print(f"  Challenger recall advantage: +{rec_delta:.3f} — worth a 10% C/C traffic test.")
    print("=" * 100 + "\n")

    logger.info("Champion-Challenger table complete.")
    return cc_df


# =============================================================================
# SECTION 12: Visualization Dashboard
# =============================================================================

def build_dashboard(eval_results, threshold_results, psi_results,
                    cc_df, cfg, logger) -> None:
    logger.info("Building dashboard...")
    colors = {
        "Logistic Regression": "steelblue",
        "Random Forest": "darkorange",
        "XGB Weighted [Champion]": "green",
        "XGB SMOTE [Challenger]": "red",
    }

    fig = plt.figure(figsize=(22, 26))
    fig.suptitle("Credit Risk & Fraud Strategy — Analytics Dashboard",
                 fontsize=15, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Row 1, Col 0: PR Curves
    ax = fig.add_subplot(gs[0, 0])
    for name, r in eval_results.items():
        ap = r["pr_auc"]
        ax.plot(r["rec_curve"], r["prec_curve"],
                label=f"{name.split('[')[0].strip()} (AP={ap:.3f})",
                color=colors.get(name, "gray"))
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # Row 1, Col 1: ROC Curves
    ax = fig.add_subplot(gs[0, 1])
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    for name, r in eval_results.items():
        auc = r["roc_auc"]
        ax.plot(r["fpr"], r["tpr"],
                label=f"{name.split('[')[0].strip()} (AUC={auc:.3f})",
                color=colors.get(name, "gray"))
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # Row 1, Col 2: Score Distribution (Champion)
    ax = fig.add_subplot(gs[0, 2])
    champ_name = "XGB Weighted [Champion]"
    if champ_name in eval_results:
        from_eval = eval_results[champ_name]
        # We need y_test to split; use threshold_results y_score instead
        y_score = threshold_results["y_score"]
        # We don't have y_test here — plot as overall distribution
        ax.hist(y_score, bins=50, color="steelblue", alpha=0.7, density=True)
        ax.axvline(x=cfg["DECISION_THRESHOLD"], color="red", linestyle="--",
                   label=f"Threshold={cfg['DECISION_THRESHOLD']}")
        ax.set_xlabel("Predicted Default Probability")
        ax.set_ylabel("Density")
        ax.set_title("Score Distribution\n(Champion XGB Weighted)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # Row 2, Col 0: Threshold Cost Curve
    ax = fig.add_subplot(gs[1, 0])
    tdf = threshold_results["threshold_df"]
    ax2_twin = ax.twinx()
    ax.plot(tdf["threshold"], tdf["total_cost"] / 1e6, color="tomato", label="Expected Cost ($M)")
    ax2_twin.plot(tdf["threshold"], tdf["approval_rate"] * 100, color="steelblue",
                  linestyle="--", label="Approval Rate %")
    ax.axvline(x=threshold_results["opt_cost"], color="black", linestyle=":",
               label=f"Optimal={threshold_results['opt_cost']:.2f}")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Expected Cost ($M)", color="tomato")
    ax2_twin.set_ylabel("Approval Rate (%)", color="steelblue")
    ax.set_title("Business Cost vs Threshold\n(FN=$5k, FP=$200)")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7)
    ax.grid(alpha=0.3)

    # Row 2, Col 1: Precision vs Recall by threshold
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(tdf["threshold"], tdf["precision"], color="steelblue", label="Precision")
    ax.plot(tdf["threshold"], tdf["recall"], color="tomato", label="Recall")
    ax.plot(tdf["threshold"], tdf["f1"], color="green", linestyle="--", label="F1")
    ax.axvline(x=cfg["DECISION_THRESHOLD"], color="gray", linestyle=":",
               label=f"Current={cfg['DECISION_THRESHOLD']}")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 by Threshold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Row 2, Col 2: KS Visualization (Champion)
    ax = fig.add_subplot(gs[1, 2])
    champ_r = eval_results.get("XGB Weighted [Champion]", {})
    if champ_r:
        ax.plot(champ_r["fpr"], champ_r["tpr"], color="green", label="TPR (Default capture)")
        ax.plot(champ_r["fpr"], champ_r["fpr"], color="gray", linestyle="--")
        ax.fill_between(champ_r["fpr"],
                        champ_r["tpr"], champ_r["fpr"],
                        alpha=0.15, color="green")
        ks = champ_r["ks_statistic"]
        ax.set_title(f"KS Statistic: {ks:.2f}\n(industry benchmark: KS > 40 = good)")
        ax.set_xlabel("Cumulative Non-Default Rate")
        ax.set_ylabel("Cumulative Default Rate")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # Row 3, Col 0: PSI monitoring bar chart
    ax = fig.add_subplot(gs[2, 0])
    n_bins = 10
    dev_s = psi_results["dev_scores"]
    prod_s = psi_results["prod_scores"]
    bins = np.linspace(0, 1, n_bins + 1)
    dev_hist = np.histogram(dev_s, bins=bins)[0] / len(dev_s)
    prod_hist = np.histogram(prod_s, bins=bins)[0] / len(prod_s)
    x = np.arange(n_bins)
    width = 0.35
    ax.bar(x - width / 2, dev_hist, width, label="Development", color="steelblue", alpha=0.7)
    ax.bar(x + width / 2, prod_hist, width, label="Production", color="coral", alpha=0.7)
    ax.set_xlabel("Score Decile")
    ax.set_ylabel("Population %")
    ax.set_title(f"PSI Monitoring\nDev vs Prod PSI={psi_results['psi_normal']:.4f} — "
                 f"{psi_results['status_normal'].split('—')[0].strip()}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # Row 3, Col 1: Model comparison bar (AUC)
    ax = fig.add_subplot(gs[2, 1])
    names = [n.split("[")[0].strip() for n in eval_results]
    aucs = [r["roc_auc"] for r in eval_results.values()]
    bar_colors = [colors.get(n, "gray") for n in eval_results]
    bars = ax.barh(names, aucs, color=bar_colors, alpha=0.8)
    ax.set_xlabel("ROC-AUC")
    ax.set_title("Model AUC Comparison")
    ax.set_xlim(0.5, max(aucs) + 0.05)
    for bar, val in zip(bars, aucs):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)
    ax.grid(alpha=0.3, axis="x")

    # Row 3, Col 2: Gini comparison
    ax = fig.add_subplot(gs[2, 2])
    ginis = [r["gini"] for r in eval_results.values()]
    bars = ax.barh(names, ginis, color=bar_colors, alpha=0.8)
    ax.set_xlabel("Gini Coefficient")
    ax.set_title("Gini Coefficient Comparison\n(industry benchmark: Gini > 0.5 = good)")
    ax.set_xlim(0, max(ginis) + 0.1)
    for bar, val in zip(bars, ginis):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)
    ax.grid(alpha=0.3, axis="x")

    # Row 4 (full width): Champion-Challenger table
    ax = fig.add_subplot(gs[3, :])
    ax.axis("off")
    table_data = [cc_df.columns.tolist()] + cc_df.values.tolist()
    tbl = ax.table(
        cellText=cc_df.values,
        colLabels=cc_df.columns,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.6)

    # Highlight champion row
    for j in range(len(cc_df.columns)):
        for i, strategy in enumerate(cc_df["Strategy"]):
            if "Champion" in str(strategy):
                tbl[i + 1, j].set_facecolor("#d4f0d4")
    ax.set_title("Champion-Challenger Strategy Comparison", fontsize=11,
                 fontweight="bold", pad=12)

    out_path = os.path.join(cfg["OUTPUT_DIR"], "dashboard.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    logger.info("Dashboard saved to %s", out_path)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    logger = setup_logging(CONFIG)
    logger.info("=== Credit Risk & Fraud Strategy Pipeline Starting ===")

    df_raw = load_data(CONFIG, logger)
    run_eda(df_raw, CONFIG, logger)

    df = engineer_features(df_raw, logger)

    rule_results = evaluate_rule_strategy(df, CONFIG, logger)

    X_train, X_test, y_train, y_test = prepare_model_data(df, CONFIG, logger)

    models = train_all_models(X_train, y_train, CONFIG, logger)

    eval_results = evaluate_all_models(models, X_test, y_test, CONFIG, logger)

    champ_model = models["XGB Weighted [Champion]"]

    threshold_results = optimize_threshold(champ_model, X_test, y_test, CONFIG, logger)

    run_shap_analysis(champ_model, X_train, X_test, CONFIG, logger)

    build_risk_scorecard(champ_model, X_test, y_test, CONFIG, logger)

    psi_results = calculate_psi_monitoring(champ_model, X_train, X_test, CONFIG, logger)

    cc_df = champion_challenger_analysis(
        models, rule_results, eval_results, CONFIG, logger
    )

    build_dashboard(eval_results, threshold_results, psi_results, cc_df, CONFIG, logger)

    logger.info("=== Pipeline complete. Outputs in '%s/' ===", CONFIG["OUTPUT_DIR"])


if __name__ == "__main__":
    main()
