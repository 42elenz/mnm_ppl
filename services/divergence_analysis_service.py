"""Service that loads sentence-level divergence data and prepares analysis settings."""

import numpy as np
import pandas as pd
from functions.analysis.statistics import create_matched_sample, winsorize_and_log


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


def aggregate_divergence(sentence_df: pd.DataFrame, model_name: str = "divergence") -> pd.DataFrame:
    """Aggregate sentence-level divergence to person-level metrics."""
    main = (
        sentence_df.groupby("file_id")
        .agg(
            {
                "divergence": [
                    "mean",
                    "median",
                    "std",
                    lambda x: x.quantile(0.90),
                    lambda x: x.quantile(0.75),
                    lambda x: (x.std() / x.mean()) if x.mean() > 0 else np.nan,
                ],
                "sent_idx": "count",
                "task": pd.Series.nunique,
            }
        )
        .reset_index()
    )

    main.columns = [
        "file_id",
        f"{model_name}_div_mean",
        f"{model_name}_div_median",
        f"{model_name}_div_std",
        f"{model_name}_div_p90",
        f"{model_name}_div_p75",
        f"{model_name}_div_cv",
        f"{model_name}_n_sentences",
        f"{model_name}_n_tasks",
    ]

    # total_words is task-level and repeated for each sentence; sum one value per task.
    words = (
        sentence_df.groupby(["file_id", "task"], as_index=False)["total_words"]
        .max()
        .groupby("file_id", as_index=False)["total_words"]
        .sum()
        .rename(columns={"total_words": f"{model_name}_total_words"})
    )

    out = main.merge(words, on="file_id", how="left")
    out[f"{model_name}_avg_words"] = (
        out[f"{model_name}_total_words"] / out[f"{model_name}_n_sentences"].replace(0, np.nan)
    )
    return out


def compute_high_divergence_proportion(
    sentence_df: pd.DataFrame,
    threshold: float,
    model_name: str = "divergence",
) -> pd.DataFrame:
    """Compute proportion of sentences with divergence above threshold per person."""
    tmp = sentence_df.copy()
    tmp["is_high_div"] = (tmp["divergence"] > threshold).astype(int)
    prop = (
        tmp.groupby("file_id")
        .agg({"is_high_div": ["sum", "mean"]})
        .reset_index()
    )
    prop.columns = [
        "file_id",
        f"{model_name}_n_high_div",
        f"{model_name}_prop_high_div",
    ]
    return prop


def build_person_df(
    sentence_df: pd.DataFrame,
    metadata: pd.DataFrame,
    model_name: str = "divergence",
) -> pd.DataFrame:
    """Aggregate divergence to person level and merge with metadata."""
    agg = aggregate_divergence(sentence_df, model_name)
    df = metadata.merge(agg, on="file_id", how="inner")

    ctrl_ids = df.loc[df["is_patient"] == 0, "file_id"].unique()
    ctrl_sents = sentence_df[sentence_df["file_id"].isin(ctrl_ids)]
    threshold = ctrl_sents["divergence"].quantile(0.90)
    print(f"High-divergence threshold (90th %ile of controls): {threshold:.4f}")

    prop = compute_high_divergence_proportion(sentence_df, threshold, model_name)
    df = df.merge(prop, on="file_id", how="left")
    return df


def prepare_three_settings(
    df: pd.DataFrame,
    model_name: str = "divergence",
) -> dict[str, pd.DataFrame]:
    """Return three analysis DataFrames: raw, winsorized, matched."""
    div_cols = [f"{model_name}_div_mean", f"{model_name}_div_median"]

    df_raw = df.copy()
    df_wins = winsorize_and_log(df, div_cols)
    df_matched = create_matched_sample(df)

    return {
        "with_outliers": df_raw,
        "without_outliers": df_wins,
        "matched": df_matched,
    }


def analysis_service(
    sentence_csv: str,
    metadata_csv: str,
    model_name: str = "divergence",
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """End-to-end: load divergence data, merge metadata, return all settings."""
    sentence_df = pd.read_csv(sentence_csv)
    metadata = load_metadata(metadata_csv)

    df = build_person_df(sentence_df, metadata, model_name)
    settings = prepare_three_settings(df, model_name)

    for name, sdf in settings.items():
        n_pat = (sdf["is_patient"] == 1).sum()
        n_ctrl = (sdf["is_patient"] == 0).sum()
        print(f"  {name}: {len(sdf)} total ({n_pat} patients, {n_ctrl} controls)")

    return df, settings
