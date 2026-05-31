import os
import sys
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, spearmanr, kruskal

sys.path.insert(0, os.path.abspath('.'))

from services.analysis_service import analysis_service, load_metadata, build_person_df
from functions.analysis.statistics import (
    CLINICAL_VARS,
    aggregate_loss_completion,
    aggregate_residualized_loss,
    add_length_residualized_loss,
    add_winsorized_columns,
    group_comparison,
    pairwise_vs_hc,
    kruskal_wallis_test,
    clinical_correlations,
    residualize_ppl
)

# ── CONFIGURATION ────────────────────────────────────────────────────────────
BASE_DIR = "/Users/lenze/code/HITKIP/mnm_ppl"
MODEL_DATA_DIR = os.path.join(BASE_DIR, "data/02_perplexities/data/02_perplexities_vllm")
METADATA_CSV = os.path.join(BASE_DIR, "data/00_raw_data/speech_metadata.csv")
FIGURE_DIR = os.path.join(BASE_DIR, "figures/model_ablation")
os.makedirs(FIGURE_DIR, exist_ok=True)

TARGET_PROMPT = "prompt_healthy"
PREPROCESSING = "without_outliers"

def fmt_p(p):
    if p < 0.001: return "< 0.001"
    elif p < 0.01: return f"{p:.3f}"
    else: return f"{p:.2f}"

# Find all model directories
model_dirs = [d for d in os.listdir(MODEL_DATA_DIR) if os.path.isdir(os.path.join(MODEL_DATA_DIR, d))]

all_group_results = []
all_pairwise_results = []
all_clinical_results = []
subject_metrics = []

for model_folder in sorted(model_dirs):
    # Create a nice label for the model
    folder_lower = model_folder.lower()
    if "qwen2.5-32b" in folder_lower:
        model_label = "Qwen 32B"
    elif "qwen2.5-7b" in folder_lower:
        model_label = "Qwen 7B"
    elif "qwen" in folder_lower:
        model_label = "Qwen"
    elif "leo" in folder_lower:
        model_label = "Leo"
    else:
        model_label = model_folder[:10]
        
    # We use a generic 'lm' prefix for the analysis columns to keep the code model-agnostic
    MODEL_PREFIX = "lm"
    PPL_VAR = f"{MODEL_PREFIX}_loss_median"
    WC_VAR = f"{MODEL_PREFIX}_total_words"
    MATCH_VARS = [f"{MODEL_PREFIX}_avg_words", "education"]
    CONFOUNDER_VARS = ["sex_num", "age", f"{MODEL_PREFIX}_avg_words", "education"]

    suffix = TARGET_PROMPT.split('_', 1)[1]
    sentence_csv = os.path.join(MODEL_DATA_DIR, model_folder, TARGET_PROMPT, f"TAT_german_utterances_with_perplexities_{suffix}.csv")
    
    if not os.path.exists(sentence_csv):
        print(f"Skipping {model_folder}: File not found {sentence_csv}")
        continue

    print(f"\n{'='*60}")
    print(f"Processing Model: {model_label} ({model_folder})")
    print(f"{'='*60}")

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

    # Deconfounding (Residualization) - Exactly as in notebook
    df["log_total_words"] = np.log1p(df[WC_VAR])
    df["log_avg_words"] = np.log1p(df[f"{MODEL_PREFIX}_avg_words"])
    
    # Prepare confounders for OLS
    ols_confounders = [c if c != WC_VAR else "log_total_words" for c in CONFOUNDER_VARS]
    if "log_avg_words" not in ols_confounders:
        ols_confounders.append("log_avg_words")

    resid_ppl = residualize_ppl(df, PPL_VAR, ols_confounders, fit_on_all=True)
    resid_col = f"{PPL_VAR}_resid"
    df[resid_col] = np.nan
    df.loc[resid_ppl.index, resid_col] = resid_ppl

    # Lock in the final sample by dropping subjects that could not be residualized (N=371)
    df = df.dropna(subset=[resid_col]).copy()

    prompt_subjects = df[["file_id", "is_patient", "diagnosis", PPL_VAR, resid_col]].copy()
    prompt_subjects.columns = ["file_id", "is_patient", "diagnosis", f"{model_label}_Raw", f"{model_label}_Resid"]
    subject_metrics.append(prompt_subjects)

    for var in [PPL_VAR, resid_col]:
        comp = group_comparison(df, var)
        comp["Model"] = model_label
        comp["Metric"] = "Residualized" if "_resid" in var else "Raw"
        all_group_results.append(comp)

    for var in [PPL_VAR, resid_col]:
        pw = pairwise_vs_hc(df, var)
        for row in pw:
            row["Model"] = model_label
            row["Metric"] = "Residualized" if "_resid" in var else "Raw"
            all_pairwise_results.append(row)

    patients = df[df["is_patient"] == 1].copy()
    available_clinical = [v for v in CLINICAL_VARS if v in patients.columns]
    
    for var in [PPL_VAR, resid_col]:
        corr = clinical_correlations(patients, var, available_clinical)
        corr["Model"] = model_label
        corr["Metric"] = "Residualized" if "_resid" in var else "Raw"
        all_clinical_results.append(corr)

# ── CONSOLIDATION & EXPORT ───────────────────────────────────────────────────
print("\nSaving results...")

final_group_df = pd.DataFrame(all_group_results)
final_group_df["p(MW)_fmt"] = final_group_df["p(MW)"].apply(fmt_p)
group_csv = os.path.join(FIGURE_DIR, "ablation_group_comparison.csv")
final_group_df.to_csv(group_csv, index=False)

final_pairwise_df = pd.DataFrame(all_pairwise_results)
final_pairwise_df["p_fmt"] = final_pairwise_df["p"].apply(fmt_p)
pairwise_csv = os.path.join(FIGURE_DIR, "ablation_pairwise_diagnosis.csv")
final_pairwise_df.to_csv(pairwise_csv, index=False)

final_clinical_df = pd.concat(all_clinical_results, ignore_index=True)
final_clinical_df["p_fmt"] = final_clinical_df["p"].apply(fmt_p)
final_clinical_df["p_fdr_fmt"] = final_clinical_df["p_fdr"].apply(fmt_p)
clinical_csv = os.path.join(FIGURE_DIR, "ablation_clinical_correlations.csv")
final_clinical_df.to_csv(clinical_csv, index=False)

print("Generating publication-ready plots and tables...")

final_group_df["Summary"] = "d=" + final_group_df["Cohens d"].round(2).astype(str) + ", p=" + final_group_df["p(MW)_fmt"]
summary_all = final_group_df[["Model", "Metric", "Summary"]].copy()
summary_all["Comparison"] = "HC vs. All Patients"

final_pairwise_df["Summary"] = "d=" + final_pairwise_df["Cohen's d"].round(2).astype(str) + ", p=" + final_pairwise_df["p_fmt"]
summary_pw = final_pairwise_df[["Model", "Metric", "Diagnosis", "Summary"]].copy()
summary_pw = summary_pw.rename(columns={"Diagnosis": "Comparison"})
summary_pw["Comparison"] = "HC vs. " + summary_pw["Comparison"]

total_summary = pd.concat([summary_all, summary_pw], ignore_index=True)

# Drop any accidental duplicates before pivoting
total_summary = total_summary.drop_duplicates(subset=["Comparison", "Metric", "Model"])

wide_summary = total_summary.pivot(index=["Comparison", "Metric"], columns="Model", values="Summary")

comp_order = ["HC vs. All Patients", "HC vs. SSD", "HC vs. MDD", "HC vs. BD"]
wide_summary = wide_summary.reindex(comp_order, level="Comparison")

summary_table_csv = os.path.join(FIGURE_DIR, "supplementary_table_model_robustness.csv")
wide_summary.to_csv(summary_table_csv)
print(f" - {summary_table_csv}")

clean_clinical_df = final_clinical_df[final_clinical_df["Metric"] == "Raw"].drop_duplicates(subset=["Clinical Variable", "Model"])
heatmap_data = clean_clinical_df.pivot(index="Clinical Variable", columns="Model", values="r")
heatmap_data = heatmap_data.dropna(how="all")

fig, ax = plt.subplots(figsize=(8, 12))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-0.5, vmax=0.5, center=0, square=True, linewidths=.5, cbar_kws={"label": "Spearman correlation (r)"}, ax=ax)
ax.set_title("")
ax.set_ylabel("")
ax.set_xlabel("")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
fig.savefig(os.path.join(FIGURE_DIR, "supplementary_clinical_correlation_heatmap.png"), dpi=300)
print(f" - supplementary_clinical_correlation_heatmap.png")

# 3. Cross-Model Scatter Plots
# ─────────────────────────────────────────────────────────────────────────────
# Merge all subject metrics into one wide dataframe
wide_subjects = subject_metrics[0]
for sm in subject_metrics[1:]:
    wide_subjects = wide_subjects.merge(sm.drop(columns=["is_patient", "diagnosis"]), on="file_id", how="inner")

models_found = [c.replace("_Raw", "") for c in wide_subjects.columns if c.endswith("_Raw")]

if len(models_found) > 1:
    target_label = models_found[0]
    other_models = models_found[1:]

    n_others = len(other_models)
    fig, axes = plt.subplots(1, n_others, figsize=(5 * n_others, 5), squeeze=False)
    axes = axes.flatten()

    for i, (ax, other) in enumerate(zip(axes, other_models)):
        x_col = f"{target_label}_Raw"
        y_col = f"{other}_Raw"
        
        plot_data = wide_subjects[[x_col, y_col, "is_patient"]].dropna()
        if len(plot_data) < 2: continue

        # Scatter plot colored by patient status
        colors = plot_data["is_patient"].map({0: "#3498DB", 1: "#E74C3C"})
        ax.scatter(plot_data[x_col], plot_data[y_col], c=colors, alpha=0.5, s=20, edgecolors="none")
        
        # Add legend to the first plot
        if i == 0:
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', label='HC', markerfacecolor='#3498DB', markersize=8),
                Line2D([0], [0], marker='o', color='w', label='Patient', markerfacecolor='#E74C3C', markersize=8)
            ]
            ax.legend(handles=legend_elements, loc='lower right', frameon=True, fontsize=10)
        
        # Identity line
        lims = [min(plot_data[x_col].min(), plot_data[y_col].min()), max(plot_data[x_col].max(), plot_data[y_col].max())]
        ax.plot(lims, lims, 'k--', alpha=0.3, lw=1)
        
        ax.set_xlabel(f"Loss ({target_label})")
        ax.set_ylabel(f"Loss ({other})")
        
        # Add Spearman r to the corner
        r, p = spearmanr(plot_data[x_col], plot_data[y_col])
        ax.text(0.05, 0.95, f"r = {r:.3f}", transform=ax.transAxes, verticalalignment='top', fontsize=12)
        ax.grid(alpha=0.2)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "supplementary_cross_model_scatter.png"), dpi=300)
    print(f" - supplementary_cross_model_scatter.png")

print(f"Done! All results saved in {FIGURE_DIR}")
