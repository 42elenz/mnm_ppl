
import os
import sys
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from scipy.stats import zscore

# Ensure project root is in path
sys.path.insert(0, os.path.abspath('.'))

from services.analysis_service import analysis_service
from functions.analysis.statistics import residualize_ppl, CLINICAL_VARS

# ── 1. CONFIGURATION & DATA LOAD ─────────────────────────────────────────────
BASE_DIR = "/Users/lenze/code/HITKIP/mnm_ppl"
METADATA_CSV = os.path.join(BASE_DIR, "data/00_raw_data/speech_metadata.csv")
SENTENCE_CSV = os.path.join(BASE_DIR, "data/02_perplexities/data/02_perplexities_vllm/Qwen2.5-32B-Instruct/prompt_healthy/TAT_german_utterances_with_perplexities_healthy.csv")
MODEL = "qwen"

print("Loading data and applying preprocessing...")
df_all, settings = analysis_service(
    sentence_csv=SENTENCE_CSV,
    metadata_csv=METADATA_CSV,
    model_name=MODEL,
    match_vars=[f"{MODEL}_avg_words", "education"],
    match_k=2,
    requested_clinical_vars=CLINICAL_VARS,
    impute_confounders_cols=["sex_num", "age", f"{MODEL}_avg_words", "education"]
)

# Use 'without_outliers' as per main analysis blueprint
df = settings["without_outliers"].copy()
df["sex_num"] = df["sex"].map({"M": 0, "F": 1, "m": 0, "f": 1, "male": 0, "female": 1, "Male": 0, "Female": 1})

# ── 1b. ADD LOSS COLUMNS (Exactly as notebook) ──────────────────────────────
from functions.analysis.statistics import (
    add_length_residualized_loss, aggregate_loss_completion, aggregate_residualized_loss
)
sentence_df = pd.read_csv(SENTENCE_CSV)
sentence_df_resid, _ = add_length_residualized_loss(sentence_df)
agg_loss = aggregate_loss_completion(sentence_df_resid, MODEL)
agg_resid = aggregate_residualized_loss(sentence_df_resid, MODEL)

merge_keys = ["file_id", f"{MODEL}_total_words", f"{MODEL}_avg_words", f"{MODEL}_n_sentences"]
loss_person_df = agg_loss.merge(agg_resid, on=merge_keys, how="outer")
loss_person_df["file_id"] = loss_person_df["file_id"].apply(lambda x: str(x).split("_")[0].replace("-", "").replace(" ", "").strip())

new_loss_cols = [c for c in loss_person_df.columns if c == "file_id" or c not in df.columns]
df = df.merge(loss_person_df[new_loss_cols], on="file_id", how="left")

# ── 2. RESIDUALIZATION (Locking in N=371) ────────────────────────────────────
PPL_VAR = f"{MODEL}_loss_median"

# Add log transformation for word count
df["log_avg_words"] = np.log1p(df[f"{MODEL}_avg_words"])
ols_confounders = ["sex_num", "age", "education", "log_avg_words"]

resid_ppl = residualize_ppl(df, PPL_VAR, ols_confounders, fit_on_all=True)
resid_col = "residualized_LLM_loss"
df[resid_col] = np.nan
df.loc[resid_ppl.index, resid_col] = resid_ppl

# Drop subjects that could not be residualized (N=371, BD=26)
df = df.dropna(subset=[resid_col]).copy()

# ── 3. DATA FILTERING: PSYCHOSIS-MANIA SPECTRUM (SSD + BD) ───────────────────
cohort = df[df["diagnosis"].isin(["SSD", "BD"])].copy()
print(f"Pooled Cohort N = {len(cohort)} (SSD: {sum(cohort['diagnosis']=='SSD')}, BD: {sum(cohort['diagnosis']=='BD')})")

# Dummy variable (BD=0, SSD=1)
cohort["SSD_v_BD"] = (cohort["diagnosis"] == "SSD").astype(int)

# Standardize the LLM loss so Beta = standardized Beta for the predictor
cohort["loss_z"] = zscore(cohort[resid_col])

# The 18 dependent variables mapped to actual columns in your dataframe
y_map = {
    'SAPS_Sum_ENIGMA': 'SAPS Total',
    'SAPS_PositiveFormaleDenkstoerung_ENIGMA': 'SAPS FTD',
    'SAPS_Wahn_ENIGMA': 'SAPS Delusions',
    'SAPS_Halluzinationen_ENIGMA': 'SAPS Hallucinations',
    'SAPS_BizarresVerhalten_ENIGMA': 'SAPS Bizarre Behavior',
    'SANS_AbulieApathie_ENIGMA': 'SANS Avolition',
    'SANS_AlogieParalogie_ENIGMA': 'SANS Alogia',
    'SANS_Anhedonie_ENIGMA': 'SANS Anhedonia',
    'SANS_Affektverflachung_ENIGMA': 'SANS Affective Flattening',
    'YMRS_Sum': 'YMRS Total',
    'HAMD_Sum17': 'HAMD-17 Total',
    'verbal_IQ': 'Verbal IQ',
    'LexicalVF': 'Lexical Fluency',
    'SemanticVF': 'Semantic Fluency',
    'TLI_Impoverished': 'TLI Impoverished',
    'TLI_Disorganization': 'TLI Disorganization',
    'TLI_Non_specific': 'TLI Nonspecific',
    'GAF': 'GAF Score'
}

results = []

# ── 4. STATISTICAL MODELLING PIPELINE ────────────────────────────────────────
for y_raw, y_label in y_map.items():
    if y_raw not in cohort.columns:
        continue
        
    # Drop NaNs for this specific model
    model_df = cohort[[y_raw, "loss_z", "SSD_v_BD"]].dropna()
    
    if len(model_df) < 15:
        continue
        
    # Standardize Y so that the beta coefficient is fully standardized
    model_df[f"{y_raw}_z"] = zscore(model_df[y_raw])
    
    # Fit OLS: Y ~ Loss + Diagnosis
    formula = f"{y_raw}_z ~ loss_z + SSD_v_BD"
    model = smf.ols(formula, data=model_df).fit()
    
    # Extract metrics for 'loss_z'
    beta = model.params["loss_z"]
    se = model.bse["loss_z"]
    t_stat = model.tvalues["loss_z"]
    p_val = model.pvalues["loss_z"]
    
    results.append({
        "Dependent Variable": y_label,
        "n": len(model_df),
        "Beta (Loss)": beta,
        "Std Error": se,
        "t-stat": t_stat,
        "Raw P-value": p_val
    })

results_df = pd.DataFrame(results)

# ── 5. MULTIPLE COMPARISON CORRECTION ────────────────────────────────────────
_, q_vals, _, _ = multipletests(results_df["Raw P-value"], alpha=0.05, method="fdr_bh")
results_df["FDR q-value"] = q_vals
results_df["Significant? (Y/N)"] = results_df["FDR q-value"].apply(lambda q: "Y" if q < 0.05 else "N")

# Sort by q-value
results_df = results_df.sort_values("FDR q-value").reset_index(drop=True)

# Formatting for terminal output
def format_df(df):
    f_df = df.copy()
    f_df["Beta (Loss)"] = f_df["Beta (Loss)"].apply(lambda x: f"{x:+.3f}")
    f_df["Std Error"] = f_df["Std Error"].apply(lambda x: f"{x:.3f}")
    f_df["t-stat"] = f_df["t-stat"].apply(lambda x: f"{x:+.2f}")
    f_df["Raw P-value"] = f_df["Raw P-value"].apply(lambda x: "< 0.001" if x < 0.001 else f"{x:.3f}")
    f_df["FDR q-value"] = f_df["FDR q-value"].apply(lambda x: "< 0.001" if x < 0.001 else f"{x:.3f}")
    return f_df

print("\n--- Psychosis-Mania Spectrum (SSD + BD) Subanalysis ---")
print("Standardized Regression: Symptom ~ residualized_LLM_loss + Diagnosis_Dummy")
print(format_df(results_df).to_string(index=False))

# Save results
out_path = os.path.join(BASE_DIR, "figures/00_main_final_tables_figures/table_psychosis_spectrum_subanalysis.csv")
results_df.to_csv(out_path, index=False)
print(f"\nResults saved to: {out_path}")
