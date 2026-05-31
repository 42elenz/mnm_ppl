import os
import sys
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, spearmanr, kruskal

# Add project root to path
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
# Paths
BASE_DIR = "/Users/lenze/code/HITKIP/mnm_ppl"
# This can be swapped for newer output folders later
MODEL_DATA_DIR = "data/02_perplexities/data/02_perplexities_vllm/Qwen2.5-32B-Instruct"
METADATA_CSV = os.path.join(BASE_DIR, "data/00_raw_data/speech_metadata.csv")
FIGURE_DIR = os.path.join(BASE_DIR, "figures/prompt_ablation")
os.makedirs(FIGURE_DIR, exist_ok=True)

MODEL = "qwen"
PROMPT_TYPES = ["prompt_healthy", "prompt_sick", "prompt_neutral", "prompt_no_prompt"]
PROMPT_LABELS = {
    "prompt_healthy": "Healthy",
    "prompt_sick": "Patient",
    "prompt_neutral": "Neutral",
    "prompt_no_prompt": "No prompt"
}

PREPROCESSING = "without_outliers"  # choice from settings: "with_outliers" or "without_outliers"
PPL_VAR = f"{MODEL}_loss_median"
WC_VAR = f"{MODEL}_total_words"

# Variables for matching and confounders (based on notebook)
MATCH_VARS = [f"{MODEL}_avg_words", "education"]
CONFOUNDER_VARS = ["sex_num", "age", f"{MODEL}_avg_words", "education"]

def fmt_p(p):
    if p < 0.001:
        return "< 0.001"
    elif p < 0.01:
        return f"{p:.3f}"
    else:
        return f"{p:.2f}"

# ── MAIN LOOP ────────────────────────────────────────────────────────────────
all_group_results = []
all_pairwise_results = []
all_clinical_results = []
subject_metrics = [] # To collect subject-level data for cross-prompt scatter

for ptype in PROMPT_TYPES:
    label = PROMPT_LABELS[ptype]
    suffix = ptype.split('_', 1)[1]
    sentence_csv = os.path.join(BASE_DIR, MODEL_DATA_DIR, ptype, f"TAT_german_utterances_with_perplexities_{suffix}.csv")
    
    if not os.path.exists(sentence_csv):
        print(f"Skipping {ptype}: File not found {sentence_csv}")
        continue

    print(f"\n{'='*60}")
    print(f"Processing Prompt: {label} ({ptype})")
    print(f"{'='*60}")

    # 1. Load data & prepare analysis settings
    df_all, settings = analysis_service(
        sentence_csv=sentence_csv,
        metadata_csv=METADATA_CSV,
        model_name=MODEL,
        match_vars=MATCH_VARS,
        match_k=2,
        requested_clinical_vars=CLINICAL_VARS,
        impute_confounders_cols=CONFOUNDER_VARS,
    )
    
    # 2. Select target preprocessing setting
    df = settings[PREPROCESSING].copy()
    print(f"Selected setting: {PREPROCESSING} (N={len(df)})")
    
    # sex_num mapping as in notebook
    df["sex_num"] = df["sex"].map({
        "M": 0, "F": 1, "m": 0, "f": 1,
        "male": 0, "female": 1, "Male": 0, "Female": 1,
    })

    # 3. Add length-residualized loss features (exactly as blueprint)
    sentence_df = pd.read_csv(sentence_csv)
    sentence_df_resid, _ = add_length_residualized_loss(sentence_df)
    
    agg_loss = aggregate_loss_completion(sentence_df_resid, MODEL)
    agg_resid = aggregate_residualized_loss(sentence_df_resid, MODEL)
    
    merge_keys = ["file_id", f"{MODEL}_total_words", f"{MODEL}_avg_words", f"{MODEL}_n_sentences"]
    loss_person_df = agg_loss.merge(agg_resid, on=merge_keys, how="outer")
    
    loss_person_df = add_winsorized_columns(
        loss_person_df,
        cols=[f"{MODEL}_loss_mean", f"{MODEL}_loss_median", f"{MODEL}_loss_resid_mean", f"{MODEL}_loss_resid_median"],
        lower_q=0.01, upper_q=0.99,
    )
    
    # Cleanup file_id for merging as in notebook
    loss_person_df["file_id"] = loss_person_df["file_id"].apply(
        lambda x: str(x).split("_")[0].replace("-", "").replace(" ", "").strip()
    )
    
    # Merge new loss columns into df
    new_loss_cols = [c for c in loss_person_df.columns if c == "file_id" or c not in df.columns]
    df = df.merge(loss_person_df[new_loss_cols], on="file_id", how="left")

    # 4. Deconfounding (Residualization) - Exactly as in notebook
    df["log_total_words"] = np.log1p(df[WC_VAR])
    df["log_avg_words"] = np.log1p(df[f"{MODEL}_avg_words"])
    
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

    # Collect subject-level data for this prompt
    prompt_subjects = df[["file_id", "is_patient", "diagnosis", PPL_VAR, resid_col]].copy()
    prompt_subjects.columns = ["file_id", "is_patient", "diagnosis", f"{label}_Raw", f"{label}_Resid"]
    subject_metrics.append(prompt_subjects)

    # 5. Core Analysis: Group Comparisons
    for var in [PPL_VAR, resid_col]:
        comp = group_comparison(df, var)
        comp["Prompt"] = label
        comp["Metric"] = "Residualized" if "_resid" in var else "Raw"
        all_group_results.append(comp)

    # 6. Pairwise vs HC
    for var in [PPL_VAR, resid_col]:
        pw = pairwise_vs_hc(df, var)
        for row in pw:
            row["Prompt"] = label
            row["Metric"] = "Residualized" if "_resid" in var else "Raw"
            all_pairwise_results.append(row)

    # 7. Clinical Correlations (Patients only)
    patients = df[df["is_patient"] == 1].copy()
    available_clinical = [v for v in CLINICAL_VARS if v in patients.columns]
    
    for var in [PPL_VAR, resid_col]:
        corr = clinical_correlations(patients, var, available_clinical)
        corr["Prompt"] = label
        corr["Metric"] = "Residualized" if "_resid" in var else "Raw"
        all_clinical_results.append(corr)

# ── CONSOLIDATION & EXPORT ───────────────────────────────────────────────────
print("\nSaving results...")

# Group Comparisons
final_group_df = pd.DataFrame(all_group_results)
final_group_df["p(MW)_fmt"] = final_group_df["p(MW)"].apply(fmt_p)
group_csv = os.path.join(FIGURE_DIR, "ablation_group_comparison.csv")
final_group_df.to_csv(group_csv, index=False)

# Pairwise Diagnosis
final_pairwise_df = pd.DataFrame(all_pairwise_results)
final_pairwise_df["p_fmt"] = final_pairwise_df["p"].apply(fmt_p)
pairwise_csv = os.path.join(FIGURE_DIR, "ablation_pairwise_diagnosis.csv")
final_pairwise_df.to_csv(pairwise_csv, index=False)

# Clinical Correlations
final_clinical_df = pd.concat(all_clinical_results, ignore_index=True)
final_clinical_df["p_fmt"] = final_clinical_df["p"].apply(fmt_p)
final_clinical_df["p_fdr_fmt"] = final_clinical_df["p_fdr"].apply(fmt_p)
clinical_csv = os.path.join(FIGURE_DIR, "ablation_clinical_correlations.csv")
final_clinical_df.to_csv(clinical_csv, index=False)

# ── VISUALIZATIONS FOR PUBLICATION ───────────────────────────────────────────
print("Generating publication-ready plots and tables...")

# 1. Consolidated Group Comparison Table (Side-by-Side)
# ─────────────────────────────────────────────────────────────────────────────
# Create display strings: "d=0.33, p<0.001"
final_group_df["Summary"] = "d=" + final_group_df["Cohens d"].round(2).astype(str) + ", p=" + final_group_df["p(MW)_fmt"]
summary_all = final_group_df[["Prompt", "Metric", "Summary"]].copy()
summary_all["Comparison"] = "HC vs. All Patients"

final_pairwise_df["Summary"] = "d=" + final_pairwise_df["Cohen's d"].round(2).astype(str) + ", p=" + final_pairwise_df["p_fmt"]
summary_pw = final_pairwise_df[["Prompt", "Metric", "Diagnosis", "Summary"]].copy()
summary_pw = summary_pw.rename(columns={"Diagnosis": "Comparison"})
summary_pw["Comparison"] = "HC vs. " + summary_pw["Comparison"]

# Combine All and Per-Diagnosis
total_summary = pd.concat([summary_all, summary_pw], ignore_index=True)

# Pivot to wide format: Rows = [Comparison, Metric], Columns = Prompt
wide_summary = total_summary.pivot(index=["Comparison", "Metric"], columns="Prompt", values="Summary")

# Reorder columns to match standard prompt order
cols_order = [PROMPT_LABELS[pt] for pt in PROMPT_TYPES if PROMPT_LABELS[pt] in wide_summary.columns]
wide_summary = wide_summary[cols_order]

# Sort comparisons for better readability (SSD first)
comp_order = ["HC vs. All Patients", "HC vs. SSD", "HC vs. MDD", "HC vs. BD"]
wide_summary = wide_summary.reindex(comp_order, level="Comparison")

summary_table_csv = os.path.join(FIGURE_DIR, "supplementary_table_prompt_robustness.csv")
wide_summary.to_csv(summary_table_csv)
print(f" - {summary_table_csv}")

# 2. Clinical Correlation Heatmap (Raw Metric)
# ─────────────────────────────────────────────────────────────────────────────
# Drop potential duplicates from the clinical list (sometimes caused by input list overlaps)
clean_clinical_df = final_clinical_df[final_clinical_df["Metric"] == "Raw"].drop_duplicates(
    subset=["Clinical Variable", "Prompt"]
)

heatmap_data = clean_clinical_df.pivot(
    index="Clinical Variable", columns="Prompt", values="r"
)
# Reorder columns to match PROMPT_LABELS order
cols_order = [PROMPT_LABELS[pt] for pt in PROMPT_TYPES if PROMPT_LABELS[pt] in heatmap_data.columns]
heatmap_data = heatmap_data[cols_order]

# Drop rows where all prompts are NaN
heatmap_data = heatmap_data.dropna(how="all")

fig, ax = plt.subplots(figsize=(10, 12))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-0.5, vmax=0.5, 
            center=0, square=True, linewidths=.5, cbar_kws={"label": "Spearman correlation (r)"}, ax=ax)
ax.set_title("") # No title for publication
ax.set_ylabel("")
ax.set_xlabel("")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
fig.savefig(os.path.join(FIGURE_DIR, "supplementary_clinical_correlation_heatmap.png"), dpi=300)

# 2. Cross-Prompt Scatter Plots
# ─────────────────────────────────────────────────────────────────────────────
# Merge all subject metrics into one wide dataframe
wide_subjects = subject_metrics[0]
for sm in subject_metrics[1:]:
    wide_subjects = wide_subjects.merge(sm.drop(columns=["is_patient", "diagnosis"]), on="file_id", how="inner")

# We compare everything against 'No prompt' as a baseline
target_label = "No prompt"
other_prompts = [PROMPT_LABELS[pt] for pt in PROMPT_TYPES if PROMPT_LABELS[pt] != target_label]

n_others = len(other_prompts)
fig, axes = plt.subplots(1, n_others, figsize=(5 * n_others, 5))
if n_others == 1: axes = [axes]

for ax, other in zip(axes, other_prompts):
    x_col = f"{target_label}_Raw"
    y_col = f"{other}_Raw"
    
    # Drop NaNs for correlation calculation and plotting
    plot_data = wide_subjects[[x_col, y_col, "is_patient"]].dropna()
    
    if len(plot_data) < 2:
        print(f"Warning: Not enough data for {other} vs {target_label}")
        continue

    # Scatter plot colored by patient status
    colors = plot_data["is_patient"].map({0: "#3498DB", 1: "#E74C3C"})
    ax.scatter(plot_data[x_col], plot_data[y_col], c=colors, alpha=0.5, s=20, edgecolors="none")
    
    # Add legend to the first plot
    if ax == axes[0]:
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='HC', markerfacecolor='#3498DB', markersize=8),
            Line2D([0], [0], marker='o', color='w', label='Patient', markerfacecolor='#E74C3C', markersize=8)
        ]
        ax.legend(handles=legend_elements, loc='lower right', frameon=True, fontsize=10)
    
    # Identity line
    lims = [
        min(plot_data[x_col].min(), plot_data[y_col].min()),
        max(plot_data[x_col].max(), plot_data[y_col].max())
    ]
    ax.plot(lims, lims, 'k--', alpha=0.3, lw=1)
    
    ax.set_xlabel(f"Loss ({target_label})")
    ax.set_ylabel(f"Loss ({other})")
    
    # Add Spearman r to the corner
    r, p = spearmanr(plot_data[x_col], plot_data[y_col])
    ax.text(0.05, 0.95, f"r = {r:.3f}", transform=ax.transAxes, verticalalignment='top', fontsize=12)
    ax.grid(alpha=0.2)

plt.tight_layout()
fig.savefig(os.path.join(FIGURE_DIR, "supplementary_cross_prompt_scatter.png"), dpi=300)

print(f"Done! Results saved in {FIGURE_DIR}")
print(f"Files created:\n - {group_csv}\n - {pairwise_csv}\n - {clinical_csv}")
print(f" - supplementary_clinical_correlation_heatmap.png\n - supplementary_cross_prompt_scatter.png")
