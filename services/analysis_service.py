"""Service that loads perplexity data, merges with demographics, and prepares the three analysis settings."""

import numpy as np
import pandas as pd
from functions.analysis.statistics import (
    aggregate_perplexity,
    compute_high_ppl_proportion,
    create_matched_sample,
    winsorize_and_log,
)


METADATA_RENAME = {
    "Proband": "file_id",
    "group": "diagnosis",
    "years of education ": "education",  # trailing space in original
    "GAFscore": "GAF",
}

METADATA_COLS = [
    "file_id", "diagnosis", "sex", "age", "education", "comorbidity",
    "GAF", "verbal_IQ", "TLI_Impoverished", "TLI_Disorganization", "TLI_Non_specific",
]


def load_metadata(path: str) -> pd.DataFrame:
    """Load and clean demographics / clinical metadata."""
    meta = pd.read_csv(path)
    meta = meta.rename(columns=METADATA_RENAME)
    available = [c for c in METADATA_COLS if c in meta.columns]
    meta = meta[available].copy()
    meta["is_patient"] = (meta["diagnosis"] != "HC").astype(int)
    meta["sex_num"] = (
        meta["sex"]
        .map({"male": 0, "female": 1, "Male": 0, "Female": 1})
        .fillna(0.5)
    )
    return meta


def build_person_df(
    sentence_df: pd.DataFrame,
    metadata: pd.DataFrame,
    model_name: str = "qwen",
) -> pd.DataFrame:
    """Aggregate sentences to person level and merge with metadata.

    Also adds a high-PPL proportion column using the 90th percentile
    of healthy-control sentences as threshold.
    """
    agg = aggregate_perplexity(sentence_df, model_name)
    df = metadata.merge(agg, on="file_id", how="inner")

    # High-PPL threshold = 90th percentile of control sentences
    ctrl_ids = df.loc[df["is_patient"] == 0, "file_id"].unique()
    ctrl_sents = sentence_df[sentence_df["file_id"].isin(ctrl_ids)]
    threshold = ctrl_sents["perplexity_completion"].quantile(0.90)
    print(f"High-PPL sentence threshold (90th %ile of controls): {threshold:.2f}")

    prop = compute_high_ppl_proportion(sentence_df, threshold, model_name)
    df = df.merge(prop, on="file_id", how="left")
    return df


def prepare_three_settings(
    df: pd.DataFrame,
    model_name: str = "qwen",
) -> dict[str, pd.DataFrame]:
    """Return the three analysis DataFrames.

    Keys: 'with_outliers', 'without_outliers', 'matched'.
    """
    ppl_cols = [f"{model_name}_ppl_mean", f"{model_name}_ppl_median"]

    # --- Setting 1: with outliers (raw) ---
    df_raw = df.copy()

    # --- Setting 2: without outliers (winsorized + log) ---
    df_wins = winsorize_and_log(df, ppl_cols)

    # --- Setting 3: matched sample ---
    df_matched = create_matched_sample(df)

    return {
        "with_outliers": df_raw,
        "without_outliers": df_wins,
        "matched": df_matched,
    }


def analysis_service(
    sentence_csv: str,
    metadata_csv: str,
    model_name: str = "qwen",
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """End-to-end: load data → person df → three analysis settings.

    Args:
        sentence_csv: path to sentence-level CSV with perplexity columns
        metadata_csv: path to demographics CSV

    Returns:
        (df, settings) where *df* is the full person-level DataFrame and
        *settings* is a dict with the three analysis DataFrames.
    """
    sentence_df = pd.read_csv(sentence_csv)
    metadata = load_metadata(metadata_csv)

    df = build_person_df(sentence_df, metadata, model_name)
    settings = prepare_three_settings(df, model_name)

    for name, sdf in settings.items():
        n_pat = (sdf["is_patient"] == 1).sum()
        n_ctrl = (sdf["is_patient"] == 0).sum()
        print(f"  {name}: {len(sdf)} total ({n_pat} patients, {n_ctrl} controls)")

    return df, settings
