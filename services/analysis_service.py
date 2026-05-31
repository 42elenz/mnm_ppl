"""Service that loads perplexity data, merges with demographics, and prepares the three analysis settings."""

from IPython import display
import numpy as np
import pandas as pd
from functions.analysis.statistics import (
    CLINICAL_VARS,
    aggregate_perplexity,
    compute_high_ppl_proportion,
    create_matched_sample,
    create_matched_sample_per_diagnosis,
    winsorize_and_log,
)
from functions.plotting.plots import create_table_demo

TABLE_DIR = "tables"
METADATA_RENAME = {
    "Proband": "file_id",
    "group": "diagnosis",
    "years of education ": "education",  # trailing space in original
    "GAFscore": "GAF",
}

DEFAULT_METADATA_COLS = [
    "file_id",
    "diagnosis",
    "sex",
    "age",
    "education",
    "comorbidity",
]


def _normalize_subject_id(series: pd.Series) -> pd.Series:
    """Normalize subject IDs for robust joins.

    - removes BOM and surrounding whitespace
    - keeps only the prefix before the first underscore
    """
    s = series.astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    return s.str.split("_").str[0]


def _apply_sentence_id_repairs(
    sentence_df: pd.DataFrame,
    id_repairs: dict[str, str] | None,
) -> pd.DataFrame:
    """Optionally repair known ID typos directly in sentence-level data.

    For each mapping source -> target (both interpreted after normalization),
    rows are reassigned as follows:
    - if target already exists, all source rows are moved to target
    - otherwise, rows are split per task and every second row is reassigned
      to target (useful for duplicated transcripts under one wrong ID)
    """
    if not id_repairs:
        return sentence_df

    out = sentence_df.copy()
    for src_raw, dst_raw in id_repairs.items():
        src = str(src_raw).replace("\ufeff", "").strip().split("_")[0]
        dst = str(dst_raw).replace("\ufeff", "").strip().split("_")[0]
        mask_src = out["file_id"] == src
        n_src = int(mask_src.sum())
        if n_src == 0:
            continue

        dst_exists = bool((out["file_id"] == dst).any())
        if dst_exists:
            out.loc[mask_src, "file_id"] = dst
            print(f"ID repair: reassigned {n_src} rows from {src} -> {dst}")
            continue

        if "task" not in out.columns:
            idx = out.loc[mask_src].index[1::2]
            out.loc[idx, "file_id"] = dst
            print(
                f"ID repair: split {n_src} rows from {src} into {src}/{dst} "
                f"(reassigned {len(idx)} rows; no task column)"
            )
            continue

        reassigned_idx: list[int] = []
        for _, grp in out.loc[mask_src].groupby("task", sort=False):
            reassigned_idx.extend(grp.index[1::2].tolist())
        out.loc[reassigned_idx, "file_id"] = dst
        print(
            f"ID repair: split {n_src} rows from {src} into {src}/{dst} "
            f"(reassigned {len(reassigned_idx)} rows by task)"
        )

    return out




def load_metadata(path: str, meta_data_col: list[str] | None = None) -> pd.DataFrame:
    """Load and clean demographics / clinical metadata."""
    meta = pd.read_csv(path)
    meta = meta.rename(columns=METADATA_RENAME)

    # Keep defaults + known clinical variables unless the caller provides a list.
    if meta_data_col is None:
        meta_data_col = list(dict.fromkeys(DEFAULT_METADATA_COLS + CLINICAL_VARS))

    # Allow callers to pass pre-rename names such as GAFscore.
    # Deduplicate after rename mapping so one logical variable appears once.
    requested_cols = list(dict.fromkeys(METADATA_RENAME.get(c, c) for c in meta_data_col))
    available = [c for c in requested_cols if c in meta.columns]
    meta = meta[available].copy()
    if "file_id" in meta.columns:
        meta["file_id"] = _normalize_subject_id(meta["file_id"])
        before = len(meta)
        meta = meta.drop_duplicates(subset=["file_id"], keep="first")
        after = len(meta)
        if after < before:
            print(f"Dropped {before - after} duplicate metadata rows after ID normalization.")
    meta["is_patient"] = (meta["diagnosis"] != "HC").astype(int)
    meta["sex_num"] = (
        meta["sex"]
        .map({"male": 0, "female": 1, "Male": 0, "Female": 1})
    )
    return meta


def build_person_df(
    sentence_df: pd.DataFrame,
    metadata: pd.DataFrame,
    model_name: str = "qwen",
    impute_confounders_cols: None | list[str] = None,
    id_repairs: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Aggregate sentences to person level and merge with metadata.

    Also adds a high-PPL proportion column using the 90th percentile
    of healthy-control sentences as threshold.

    If *impute_confounders_cols* is given, missing values in those columns
    are imputed using KNN (fit on HC, transform everyone) and a short
    missingness report is printed per diagnosis.
    """
    from sklearn.impute import KNNImputer

    sentence_df = sentence_df.copy()
    metadata = metadata.copy()

    sentence_df["file_id"] = _normalize_subject_id(sentence_df["file_id"])
    metadata["file_id"] = _normalize_subject_id(metadata["file_id"])
    sentence_df = _apply_sentence_id_repairs(sentence_df, id_repairs)

    agg = aggregate_perplexity(sentence_df, model_name)
    df = metadata.merge(agg, on="file_id", how="inner")

    missing_after_merge = sorted(set(metadata["file_id"]) - set(df["file_id"]))
    if missing_after_merge:
        preview = ", ".join(missing_after_merge[:20])
        suffix = " ..." if len(missing_after_merge) > 20 else ""
        print(
            f"Missing after metadata↔utterance merge: {len(missing_after_merge)} IDs"
            f" ({preview}{suffix})"
        )

    table1 = create_table_demo(df, model_name=model_name)
    print(table1.to_string(index=False))
    table1.to_csv(f"{TABLE_DIR}/table1_demo.csv", index=False)
    # High-PPL threshold = 90th percentile of control sentences
    ctrl_ids = df.loc[df["is_patient"] == 0, "file_id"].unique()
    ctrl_sents = sentence_df[sentence_df["file_id"].isin(ctrl_ids)]
    threshold = ctrl_sents["perplexity_completion"].quantile(0.90)
    print(f"High-PPL sentence threshold (90th %ile of controls): {threshold:.2f}")

    prop = compute_high_ppl_proportion(sentence_df, threshold, model_name)
    df = df.merge(prop, on="file_id", how="left")

    # ── KNN imputation of confounders ─────────────────────────────
    if impute_confounders_cols:
        # Resolve sex → sex_num for imputation (KNN needs numeric)
        imp_cols = ["sex_num" if c == "sex" else c for c in impute_confounders_cols]
        imp_cols = [c for c in imp_cols if c in df.columns]

        # --- Missingness report ---
        miss = df.groupby("diagnosis")[imp_cols].apply(lambda g: g.isna().sum())
        n_per_dx = df.groupby("diagnosis").size().rename("n")
        miss_report = miss.join(n_per_dx)
        total_missing = df[imp_cols].isna().sum()
        print("\n── Confounder missingness before imputation ──")
        print(f"Total missing per variable:\n{total_missing.to_string()}")
        print(f"\nMissing counts per diagnosis:")
        print(miss_report.to_string())
        print()

        if total_missing.sum() > 0:
            # Fit KNN on HC, transform everyone
            hc_mask = df["is_patient"] == 0
            imputer = KNNImputer(n_neighbors=5, weights="distance")
            imputer.fit(df.loc[hc_mask, imp_cols])
            df[imp_cols] = imputer.transform(df[imp_cols])
            print(f"KNN-imputed {int(total_missing.sum())} missing confounder values "
                  f"(fit on {hc_mask.sum()} HCs, cols={imp_cols})")
        else:
            print("No missing confounder values — skipping imputation.")

    return df


def prepare_three_settings(
    df: pd.DataFrame,
    model_name: str = "qwen",
    match_vars: list[str] = ["age", "sex", "education", "qwen_avg_words"],
    match_k: int = 2,
) -> dict[str, pd.DataFrame]:
    """Return analysis DataFrames for multiple settings.

    Keys: 'with_outliers', 'without_outliers', 'matched', 'matched_without_outliers',
          'matched_per_dx', 'matched_per_dx_without_outliers'.
    """
    ppl_cols = [f"{model_name}_ppl_mean", f"{model_name}_ppl_median"]

    # --- Setting 1: with outliers (raw) ---
    df_raw = df.copy()

    # --- Setting 2: without outliers (winsorized + log) ---
    df_wins = winsorize_and_log(df, ppl_cols)

    # --- Setting 3: matched sample 1:1 (incl. sentence length) ---
    df_matched = create_matched_sample(
        df,
        match_vars=match_vars,
    )

    # --- Setting 4: matched + winsorized ---
    df_matched_wins = winsorize_and_log(df_matched, ppl_cols)

    # --- Setting 5: per-diagnosis 1:k matched ---
    df_matched_k = create_matched_sample_per_diagnosis(
        df,
        match_vars=match_vars,
        k=match_k,
    )

    # --- Setting 6: per-diagnosis matched + winsorized ---
    df_matched_k_wins = winsorize_and_log(df_matched_k, ppl_cols)

    return {
        "with_outliers": df_raw,
        "without_outliers": df_wins,
        "matched": df_matched,
        "matched_without_outliers": df_matched_wins,
        "matched_per_dx": df_matched_k,
        "matched_per_dx_without_outliers": df_matched_k_wins,
    }


def analysis_service(
    sentence_csv: str,
    metadata_csv: str,
    model_name: str = "qwen",
    match_vars: list[str] = ["age", "sex", "education", "qwen_avg_words"],
    match_k: int = 2,
    meta_data_col: list[str] | None = None,
    requested_clinical_vars: list[str] | None = None,
    impute_confounders_cols: None | list[str] = None,
    id_repairs: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """End-to-end: load data → person df → analysis settings.

    Args:
        sentence_csv: path to sentence-level CSV with perplexity columns
        metadata_csv: path to demographics CSV
        model_name: name of the language model
        match_vars: list of variables to use for matching
        match_k: number of HCs to match per patient in per-diagnosis matching
        meta_data_col: explicit metadata columns to keep (optional)
        requested_clinical_vars: extra columns to preserve for correlation analysis

    Returns:
        (df, settings) where *df* is the full person-level DataFrame and
        *settings* is a dict with the analysis DataFrames.
    """
    sentence_df = pd.read_csv(sentence_csv)
    if meta_data_col is None:
        meta_data_col = list(dict.fromkeys(DEFAULT_METADATA_COLS + (requested_clinical_vars or [])))
    metadata = load_metadata(metadata_csv, meta_data_col=meta_data_col)

    df = build_person_df(
        sentence_df,
        metadata,
        model_name,
        impute_confounders_cols=impute_confounders_cols,
        id_repairs=id_repairs,
    )

    settings = prepare_three_settings(df, model_name, match_vars, match_k)

    for name, sdf in settings.items():
        n_pat = (sdf["is_patient"] == 1).sum()
        n_ctrl = (sdf["is_patient"] == 0).sum()
        print(f"  {name}: {len(sdf)} total ({n_pat} patients, {n_ctrl} controls)")

    return df, settings
