import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import statsmodels.api as sm
from scipy.stats import spearmanr, chi2
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.abspath('.'))

from services.analysis_service import analysis_service
from functions.analysis.statistics import (
    CLINICAL_VARS,
    aggregate_loss_completion,
    aggregate_residualized_loss,
    add_length_residualized_loss,
    add_winsorized_columns,
    residualize_ppl,
    evaluate_incremental_value
)
from functions.plotting.plots import plot_incremental_value_roc

# ── CONFIGURATION ────────────────────────────────────────────────────────────
BASE_DIR = "/Users/lenze/code/HITKIP/mnm_ppl"
MODEL_DATA_DIR = os.path.join(BASE_DIR, "data/02_perplexities/data/02_perplexities_vllm")
METADATA_CSV = os.path.join(BASE_DIR, "data/00_raw_data/speech_metadata.csv")
BASELINES_PARQUET = os.path.join(BASE_DIR, "baselines/features_per_participant.parquet")
FIGURE_DIR = os.path.join(BASE_DIR, "figures") # Matches main script output directory
os.makedirs(FIGURE_DIR, exist_ok=True)

TARGET_PROMPT = "prompt_healthy"
PREPROCESSING = "without_outliers"
MODEL_FOLDER = "Qwen2.5-32B-Instruct"
MODEL_LABEL = "Qwen 32B"

# ── DATA PROCESSING ─────────────────────────────────────────────────────────
print("Loading Qwen 32B Data...")
MODEL_PREFIX = "lm"
MODEL = MODEL_PREFIX # Map MODEL to MODEL_PREFIX for the user's loop
PPL_VAR = f"{MODEL_PREFIX}_loss_median"
WC_VAR = f"{MODEL_PREFIX}_total_words"
MATCH_VARS = [f"{MODEL_PREFIX}_avg_words", "education"]
CONFOUNDER_VARS = ["sex_num", "age", f"{MODEL_PREFIX}_avg_words", "education"]

suffix = TARGET_PROMPT.split('_', 1)[1]
sentence_csv = os.path.join(MODEL_DATA_DIR, MODEL_FOLDER, TARGET_PROMPT, f"TAT_german_utterances_with_perplexities_{suffix}.csv")

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

# Load baselines
print("Loading Baseline Features (Semantic Coherence)...")
baseline_df = pd.read_parquet(BASELINES_PARQUET)
baseline_df["participant_id"] = baseline_df["participant_id"].apply(lambda x: str(x).split("_")[0].replace("-", "").replace(" ", "").strip())

# Identify semantic coherence column
coherence_col = [c for c in baseline_df.columns if "semantic_coherence" in c]
if not coherence_col:
    print("WARNING: semantic_coherence_mean not found in baselines. Checking available columns:")
    print(baseline_df.columns.tolist())
    sys.exit(1)
coherence_col = coherence_col[0]

# Merge
df = df.merge(baseline_df[["participant_id", coherence_col]], left_on="file_id", right_on="participant_id", how="inner")
df = df.dropna(subset=[resid_col, coherence_col]).copy()

print(f"Final merged sample size: {len(df)}")

# ── REQUESTED INCREMENTAL VALUE ANALYSIS LOOP ───────────────────────────────
def fmt_lrt_p(p):
    """Helper to format p-values for the table."""
    if pd.isna(p): return "—"
    if p < 0.001: return "< 0.001"
    return f"{p:.3f}"

# Define the cohorts we want to analyze
cohorts = [
    {"label": "All Patients vs. HC", "diag_filter": None, "file_suffix": "all"},
    {"label": "SSD vs. HC", "diag_filter": "SSD", "file_suffix": "ssd"},
    {"label": "BD vs. HC", "diag_filter": "BD", "file_suffix": "bd"},
    {"label": "MDD vs. HC", "diag_filter": "MDD", "file_suffix": "mdd"}
]

table_rows = []
os.makedirs(os.path.join(FIGURE_DIR, "advanced"), exist_ok=True)

# Run the loops to generate separate ROC figures (without individual titles)
for i, c in enumerate(cohorts):
    print(f"\n==========================================")
    print(f"Running Analysis: {c['label']}")
    print(f"==========================================")
    
    # Filter dataset for specific diagnosis if needed
    if c['diag_filter']:
        df_sub = df[df["diagnosis"].isin(["HC", c['diag_filter']])].copy()
    else:
        df_sub = df.copy()
        
    # 1. Run the evaluation
    results = evaluate_incremental_value(
        df=df_sub, 
        target="is_patient", 
        length_metric=f"{MODEL}_avg_words", 
        ppl_metric=PPL_VAR,
        confounder_cols=CONFOUNDER_VARS,
        proximity_col=coherence_col
    )
    
    # 2. Extract metrics to build the structured table rows
    m = results['models']
    
    # Row 0: LLM Loss Only
    table_rows.append({
        "Cohort": c['label'],
        "Model Specification": "Median LLM Loss Only",
        "AIC": f"{m['Median LLM Loss Only']['aic']:.1f}",
        "CV-AUC": f"{m['Median LLM Loss Only']['auc']:.3f}",
        "LRT p-value (vs Base)": "—"
    })

    # Row 1: Length Only (Baseline)
    table_rows.append({
        "Cohort": "",
        "Model Specification": "Length Only",
        "AIC": f"{m['Length Only']['aic']:.1f}",
        "CV-AUC": f"{m['Length Only']['auc']:.3f}",
        "LRT p-value (vs Base)": "—"
    })
    
    # Row 2: Length + LLM Loss
    table_rows.append({
        "Cohort": "",
        "Model Specification": "Length + Median LLM Loss",
        "AIC": f"{m['Length + Median LLM Loss']['aic']:.1f}",
        "CV-AUC": f"{m['Length + Median LLM Loss']['auc']:.3f}",
        "LRT p-value (vs Base)": fmt_lrt_p(results['lrt_p_vs_length'])
    })
    
    # Row 3: Confounders Only (Baseline)
    table_rows.append({
        "Cohort": "",
        "Model Specification": "Confounders Only (Base)",
        "AIC": f"{m['Confounders Only']['aic']:.1f}",
        "CV-AUC": f"{m['Confounders Only']['auc']:.3f}",
        "LRT p-value (vs Base)": "—"
    })
    
    # Row 4: Confounders + LLM Loss
    table_rows.append({
        "Cohort": "",
        "Model Specification": "Confounders + Median LLM Loss",
        "AIC": f"{m['Confounders + Median LLM Loss']['aic']:.1f}",
        "CV-AUC": f"{m['Confounders + Median LLM Loss']['auc']:.3f}",
        "LRT p-value (vs Base)": fmt_lrt_p(results['lrt_p_vs_confounders'])
    })

    # Row 5: Confounders + Proximity (Baseline)
    table_rows.append({
        "Cohort": "",
        "Model Specification": "Confounders + Proximity",
        "AIC": f"{m['Confounders + Proximity']['aic']:.1f}",
        "CV-AUC": f"{m['Confounders + Proximity']['auc']:.3f}",
        "LRT p-value (vs Base)": "—"
    })

    # Row 6: Confounders + Proximity + LLM Loss
    table_rows.append({
        "Cohort": "",
        "Model Specification": "Confounders + Proximity + Median LLM Loss",
        "AIC": f"{m['Confounders + Proximity + Median LLM Loss']['aic']:.1f}",
        "CV-AUC": f"{m['Confounders + Proximity + Median LLM Loss']['auc']:.3f}",
        "LRT p-value (vs Base)": fmt_lrt_p(results['lrt_p_vs_proximity'])
    })
    
    # 3. Generate the visualization (Show legend on all plots, all titles turned off)
    save_path = os.path.join(FIGURE_DIR, "advanced", f"comprehensive_incremental_value_roc_{c['file_suffix']}.png")
    plot_incremental_value_roc(
        y_true=results['y_true'],
        models_dict=results['models'],
        save_path=save_path,
        show_title=False,         # Turned OFF individual titles
        show_legend=True          # Legend ON for all plots to display AUC numbers
    )
    print(f"Saved ROC plot (no title, legend=True) to: {save_path}")

# 4. Compile and save the final publication table
final_table = pd.DataFrame(table_rows)

os.makedirs("figures/00_main_final_tables_figures", exist_ok=True)
table_save_path = "figures/00_main_final_tables_figures/table_incremental_value.csv"
final_table.to_csv(table_save_path, index=False)

print("\n\n" + "="*60)
print("FINAL PUBLICATION TABLE")
print("="*60)
if 'display' in globals():
    display(final_table)
else:
    print(final_table.to_string(index=False))

# ── GRID COMPILATION AT THE END ─────────────────────────────────────────────
print("\nCombining individual ROC plots into a single publication-ready 2x2 grid...")
GRID_OUTPUT_DIR = "figures/00_main_final_tables_figures/figures"
os.makedirs(GRID_OUTPUT_DIR, exist_ok=True)

images_to_combine = [
    "comprehensive_incremental_value_roc_all.png",
    "comprehensive_incremental_value_roc_ssd.png",
    "comprehensive_incremental_value_roc_bd.png",
    "comprehensive_incremental_value_roc_mdd.png"
]

fig, axes = plt.subplots(
    2, 2, 
    figsize=(16, 16), 
    gridspec_kw={'height_ratios': [1, 1]}
)

panel_labels = ["a", "b", "c", "d"]
panel_titles = []

for idx, img_name in enumerate(images_to_combine):
    ax = axes[idx // 2, idx % 2]
    img_path = os.path.join(FIGURE_DIR, "advanced", img_name)
    
    if os.path.exists(img_path):
        img = mpimg.imread(img_path)
        ax.imshow(img)
    else:
        print(f"Warning: {img_path} not found.")
    
    ax.axis('off')
    
    # Place panel labels (a, b, c, d) and the Subanalyses Title text inside the panel area
    ax.text(-0.02, 1.02, f"{panel_labels[idx]}", transform=ax.transAxes, 
            fontsize=22, va='bottom', ha='left')

plt.tight_layout(pad=1.0)

grid_save_path = os.path.join(GRID_OUTPUT_DIR, "combined_comprehensive_roc_grid.png")
plt.savefig(grid_save_path, dpi=300, bbox_inches='tight')
print(f"Saved compiled 2x2 ROC results grid to: {grid_save_path}")

# ── COMPREHENSIVE CLINICAL VARIABLES LOOP (unchanged logic, saved to figures/) ──
print("\nRunning incremental clinical symptom analysis...")
patients = df[df["is_patient"] == 1].copy()
available_clinical = sorted(list(set([v for v in CLINICAL_VARS if v in patients.columns])))

clinical_results = []

for cvar in available_clinical:
    valid = patients[[cvar, resid_col, coherence_col]].dropna()
    if len(valid) < 15:
        continue
        
    y = valid[cvar]
    X1 = sm.add_constant(valid[coherence_col])
    model1 = sm.OLS(y, X1).fit()
    X2 = sm.add_constant(valid[[coherence_col, resid_col]])
    model2 = sm.OLS(y, X2).fit()
    
    lr_stat = -2 * (model1.llf - model2.llf)
    p_lrt = chi2.sf(lr_stat, df=1)
    
    r_std, p_std = spearmanr(valid[resid_col], valid[cvar])
    
    resid_y_coh = sm.OLS(y, X1).fit().resid
    resid_loss_coh = sm.OLS(valid[resid_col], X1).fit().resid
    r_partial, p_partial = spearmanr(resid_loss_coh, resid_y_coh)
    
    clinical_results.append({
        "Clinical Variable": cvar,
        "n": len(valid),
        "R2_Semantic_Only": model1.rsquared,
        "R2_Full_Model": model2.rsquared,
        "R2_Boost": model2.rsquared - model1.rsquared,
        "LRT_p_value": p_lrt,
        "Spearman_r": r_std,
        "Spearman_p": p_std,
        "Partial_r": r_partial,
        "Partial_p": p_partial
    })

clinical_df = pd.DataFrame(clinical_results)
clinical_df = clinical_df.sort_values(by="Partial_r", key=abs, ascending=False)
clinical_df.to_csv(os.path.join(FIGURE_DIR, "incremental_value", "comprehensive_incremental_clinical.csv"), index=False)

# Clean Heatmap compared Standard vs Partial Correlations
plt.figure(figsize=(8, 10))
heatmap_data = clinical_df.set_index("Clinical Variable")[["Spearman_r", "Partial_r"]]
sns.heatmap(heatmap_data, annot=True, cmap="RdBu_r", center=0, vmin=-0.4, vmax=0.4, fmt=".3f")
plt.title("Standard vs. Partial Correlations\n(Controlling for Semantic Proximity)")
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "incremental_value", "standard_vs_partial_correlation_heatmap.png"), dpi=300)

print(f"\nDone! All analyses, final table, and unified grid plots saved successfully.")
