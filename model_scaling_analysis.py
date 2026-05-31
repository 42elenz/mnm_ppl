import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.abspath('.'))

from services.analysis_service import analysis_service
from functions.analysis.statistics import (
    CLINICAL_VARS,
    aggregate_loss_completion,
    aggregate_residualized_loss,
    add_length_residualized_loss,
    add_winsorized_columns,
    clinical_correlations,
    residualize_ppl,
    compute_roc
)

# ── CONFIGURATION ────────────────────────────────────────────────────────────
BASE_DIR = "/Users/lenze/code/HITKIP/mnm_ppl"
MODEL_DATA_DIR = os.path.join(BASE_DIR, "data/02_perplexities/data/02_perplexities_vllm")
METADATA_CSV = os.path.join(BASE_DIR, "data/00_raw_data/speech_metadata.csv")
FIGURE_DIR = os.path.join(BASE_DIR, "figures/scaling_analysis")
os.makedirs(FIGURE_DIR, exist_ok=True)

TARGET_PROMPT = "prompt_healthy"
PREPROCESSING = "without_outliers"

MODELS = {
    "Qwen2.5-32B-Instruct": "Qwen 32B",
    "Qwen2.5-7B-Instruct": "Qwen 7B",
    "leo-mistral-hessianai-7b-chat": "Leo 7B"
}

def fmt_p(p):
    if p < 0.001: return "< 0.001"
    elif p < 0.01: return f"{p:.3f}"
    else: return f"{p:.2f}"

# ── DATA PROCESSING ─────────────────────────────────────────────────────────
all_subjects_data = []
model_results = {}

for model_folder, model_label in MODELS.items():
    MODEL_PREFIX = "lm"
    PPL_VAR = f"{MODEL_PREFIX}_loss_median"
    WC_VAR = f"{MODEL_PREFIX}_total_words"
    MATCH_VARS = [f"{MODEL_PREFIX}_avg_words", "education"]
    CONFOUNDER_VARS = ["sex_num", "age", f"{MODEL_PREFIX}_avg_words", "education"]

    suffix = TARGET_PROMPT.split('_', 1)[1]
    sentence_csv = os.path.join(MODEL_DATA_DIR, model_folder, TARGET_PROMPT, f"TAT_german_utterances_with_perplexities_{suffix}.csv")
    
    if not os.path.exists(sentence_csv):
        print(f"Skipping {model_label}: File not found {sentence_csv}")
        continue

    print(f"Processing {model_label}...")

    df_all, settings = analysis_service(
        sentence_csv=sentence_csv,
        metadata_csv=METADATA_CSV,
        model_name=MODEL_PREFIX,
        match_vars=MATCH_VARS,
        match_k=2,
        requested_clinical_vars=CLINICAL_VARS,
        impute_confounders_cols=CONFOUNDER_VARS,
    )
    
    df = settings[PREPROCESSING].copy()
    df["sex_num"] = df["sex"].map({"M": 0, "F": 1, "m": 0, "f": 1, "male": 0, "female": 1, "Male": 0, "Female": 1})

    sentence_df = pd.read_csv(sentence_csv)
    sentence_df_resid, _ = add_length_residualized_loss(sentence_df)
    
    agg_loss = aggregate_loss_completion(sentence_df_resid, MODEL_PREFIX)
    agg_resid = aggregate_residualized_loss(sentence_df_resid, MODEL_PREFIX)
    
    merge_keys = ["file_id", f"{MODEL_PREFIX}_total_words", f"{MODEL_PREFIX}_avg_words", f"{MODEL_PREFIX}_n_sentences"]
    loss_person_df = agg_loss.merge(agg_resid, on=merge_keys, how="outer")
    
    loss_person_df = add_winsorized_columns(
        loss_person_df,
        cols=[f"{MODEL_PREFIX}_loss_mean", f"{MODEL_PREFIX}_loss_median", f"{MODEL_PREFIX}_loss_resid_mean", f"{MODEL_PREFIX}_loss_resid_median"],
        lower_q=0.01, upper_q=0.99,
    )
    
    loss_person_df["file_id"] = loss_person_df["file_id"].apply(lambda x: str(x).split("_")[0].replace("-", "").replace(" ", "").strip())
    new_loss_cols = [c for c in loss_person_df.columns if c == "file_id" or c not in df.columns]
    df = df.merge(loss_person_df[new_loss_cols], on="file_id", how="left")

    df["log_total_words"] = np.log1p(df[WC_VAR])
    df["log_avg_words"] = np.log1p(df[f"{MODEL_PREFIX}_avg_words"])
    ols_confounders = [c if c != WC_VAR else "log_total_words" for c in CONFOUNDER_VARS]
    if "log_avg_words" not in ols_confounders:
        ols_confounders.append("log_avg_words")

    resid_ppl = residualize_ppl(df, PPL_VAR, ols_confounders, fit_on_all=True)
    resid_col = f"{PPL_VAR}_resid"
    df[resid_col] = np.nan
    df.loc[resid_ppl.index, resid_col] = resid_ppl
    df = df.dropna(subset=[resid_col]).copy()

    # Store metrics for this model
    available_clinical = sorted(list(set([v for v in CLINICAL_VARS if v in df.columns])))
    model_df = df[["file_id", "is_patient", "diagnosis"] + available_clinical + [PPL_VAR, resid_col]].copy()
    model_df = model_df.rename(columns={PPL_VAR: f"{model_label}_Raw", resid_col: f"{model_label}_Resid"})
    model_results[model_label] = model_df

# Merge all into one wide dataframe for easy comparison
comparison_df = None
for label, m_df in model_results.items():
    if comparison_df is None:
        comparison_df = m_df
    else:
        # Avoid duplicating clinical and group info
        cols_to_use = ["file_id", f"{label}_Raw", f"{label}_Resid"]
        comparison_df = comparison_df.merge(m_df[cols_to_use], on="file_id", how="inner")

print(f"Final sample size: {len(comparison_df)}")

# ── ANALYSIS 1: ROC-AUC ─────────────────────────────────────────────────────
print("Running ROC-AUC analysis...")
auc_results = []
groups = {"All Patients": comparison_df, 
          "SSD": comparison_df[comparison_df["diagnosis"].isin(["HC", "SSD"])],
          "MDD": comparison_df[comparison_df["diagnosis"].isin(["HC", "MDD"])],
          "BD": comparison_df[comparison_df["diagnosis"].isin(["HC", "BD"])]}

for group_name, sub_df in groups.items():
    if sub_df["is_patient"].nunique() < 2: continue
    
    for label in MODELS.values():
        for metric_type in ["Raw", "Resid"]:
            col = f"{label}_{metric_type}"
            auc = roc_auc_score(sub_df["is_patient"], sub_df[col])
            auc_results.append({
                "Group": group_name,
                "Model": label,
                "Metric": metric_type,
                "AUC": auc
            })

auc_df = pd.DataFrame(auc_results)
auc_df.to_csv(os.path.join(FIGURE_DIR, "scaling_auc_results.csv"), index=False)

# ── ANALYSIS 2: CLINICAL CORRELATIONS ───────────────────────────────────────
print("Running clinical correlations...")
corr_results = []
patients = comparison_df[comparison_df["is_patient"] == 1].copy()

available_clinical = sorted(list(set([v for v in CLINICAL_VARS if v in patients.columns])))

for label in MODELS.values():
    for metric_type in ["Raw", "Resid"]:
        col = f"{label}_{metric_type}"
        for cvar in available_clinical:
            valid = patients[[col, cvar]].dropna()
            if len(valid) > 10:
                r, p = spearmanr(valid[col], valid[cvar])
                corr_results.append({
                    "Model": label,
                    "Metric": metric_type,
                    "Clinical Variable": cvar,
                    "r": r,
                    "p": p,
                    "n": len(valid)
                })

corr_df = pd.DataFrame(corr_results)
corr_df.to_csv(os.path.join(FIGURE_DIR, "scaling_clinical_correlations.csv"), index=False)


# ── VISUALIZATION ──────────────────────────────────────────────────────────
print("Generating plots...")

# 1. AUC Comparison Bar Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=auc_df[auc_df["Metric"] == "Resid"], x="Group", y="AUC", hue="Model")
plt.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
plt.title("Model Discrimination (AUC) - Residualized Loss")
plt.ylim(0.4, 0.8)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "auc_comparison.png"), dpi=300)

# 2. Correlation Comparison Heatmap
plt.figure(figsize=(10, 8))
pivot_corr = corr_df[corr_df["Metric"] == "Resid"].pivot(index="Clinical Variable", columns="Model", values="r")
sns.heatmap(pivot_corr, annot=True, cmap="RdBu_r", center=0, vmin=-0.4, vmax=0.4)
plt.title("Clinical Correlations (r) - Residualized Loss")
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "correlation_heatmap.png"), dpi=300)

# 3. Highlight Plot: Qwen 7B vs 32B for SAPS
saps_var = "SAPS_PositiveFormaleDenkstoerung_ENIGMA"
if saps_var not in patients.columns:
    # Try alternative names
    potential_saps = [c for c in patients.columns if "SAPS" in c and "Positive" in c]
    if potential_saps:
        saps_var = potential_saps[0]

if saps_var in patients.columns and saps_var in pivot_corr.index:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for i, model in enumerate(["Qwen 32B", "Qwen 7B"]):
        col = f"{model}_Resid"
        sns.regplot(data=patients, x=saps_var, y=col, ax=axes[i], 
                    scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
        r = pivot_corr.loc[saps_var, model]
        axes[i].set_title(f"{model} (r = {r:.3f})")
        axes[i].set_ylabel("Residualized Loss")
        axes[i].set_xlabel(saps_var)
    
    plt.suptitle(f"{saps_var} Correlation: 32B vs 7B")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "saps_correlation_comparison.png"), dpi=300)

# ── SUMMARY TABLE ──────────────────────────────────────────────────────────
summary_table = auc_df[auc_df["Group"] == "All Patients"].pivot(index="Model", columns="Metric", values="AUC")
summary_table.columns = [f"AUC_{c}" for c in summary_table.columns]

# Add SAPS and YMRS correlations if available
for m_var in ["SAPS_PositiveFormaleDenkstoerung_ENIGMA", "YMRS_Sum"]:
    if m_var in pivot_corr.index:
        summary_table[f"r_{m_var}"] = pivot_corr.loc[m_var]

summary_table.to_csv(os.path.join(FIGURE_DIR, "scaling_summary_table.csv"))
print(f"\nScaling Summary Table:\n{summary_table}")


print(f"\nDone! Results saved in {FIGURE_DIR}")
