import os
import sys
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.abspath('.'))

from services.analysis_service import analysis_service
from functions.analysis.statistics import (
    residualize_ppl, 
    CLINICAL_VARS, 
    add_length_residualized_loss, 
    aggregate_loss_completion,
    aggregate_residualized_loss
)

BASE_DIR = "/Users/lenze/code/HITKIP/mnm_ppl"
METADATA_CSV = os.path.join(BASE_DIR, "data/00_raw_data/speech_metadata.csv")
SENTENCE_CSV = os.path.join(BASE_DIR, "data/02_perplexities/data/02_perplexities_vllm/Qwen2.5-32B-Instruct/prompt_healthy/TAT_german_utterances_with_perplexities_healthy.csv")

MODEL = "qwen"

df_all, settings = analysis_service(
    sentence_csv=SENTENCE_CSV,
    metadata_csv=METADATA_CSV,
    model_name=MODEL,
    match_vars=[f"{MODEL}_avg_words", "education"],
    match_k=2,
    requested_clinical_vars=CLINICAL_VARS,
    impute_confounders_cols=["sex_num", "age", f"{MODEL}_avg_words", "education"]
)

# NOTE: analysis_service already returns data with f"{MODEL}_loss_median" inside the 'settings' dict
df = settings["with_outliers"].copy()
df["sex_num"] = df["sex"].map({"M": 0, "F": 1, "m": 0, "f": 1, "male": 0, "female": 1, "Male": 0, "Female": 1})

# Run residualization logic
PPL_VAR = f"{MODEL}_loss_median"
# Let's ensure the column name is exact (sometimes they are renamed in the service)
if PPL_VAR not in df.columns:
    # Try looking for variants
    alt = [c for c in df.columns if "loss_median" in c]
    if alt: PPL_VAR = alt[0]

df["log_avg_words"] = np.log1p(df[f"{MODEL}_avg_words"])
ols_confounders = ["sex_num", "age", "education", "log_avg_words"]

resid_ppl = residualize_ppl(df, PPL_VAR, ols_confounders, fit_on_all=True)

df["resid"] = np.nan
df.loc[resid_ppl.index, "resid"] = resid_ppl

# Which BD patients have missing residuals?
dropped_bd = df[(df["diagnosis"] == "BD") & (df["resid"].isna())]
print("\nDropped BD Subject(s):")
print(dropped_bd[["file_id", "diagnosis", "age", "sex", "education"]])
