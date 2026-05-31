import json
from pathlib import Path

import numpy as np
import pandas as pd

from functions.analysis.statistics import aggregate_perplexity
from services.analysis_service import load_metadata

from baselines.common import (
    FEATURES_PARQUET,
    LEO_NO_PROMPT_CSV,
    METADATA_CSV,
    QWEN_NO_PROMPT_CSV,
    ensure_dirs,
    init_logger,
    set_global_seed,
)
from baselines.feature_avg_words import compute_avg_words_per_sentence
from baselines.feature_content_ratio import compute_content_function_ratio
from baselines.feature_gpt2_perplexity import compute_gpt2_sentence_and_participant
from baselines.feature_mattr import compute_mattr
from baselines.feature_semantic_coherence import compute_semantic_features
from baselines.feature_wordfreq import compute_log_frequency_features


def _build_narrative_table(sentence_df: pd.DataFrame) -> pd.DataFrame:
    nar = (
        sentence_df.groupby("file_id", as_index=False)["utterance"]
        .apply(lambda x: " ".join(x.astype(str).tolist()))
        .rename(columns={"file_id": "participant_id", "utterance": "narrative"})
    )
    return nar


def _print_validation(df: pd.DataFrame, cols: list[str], header: str) -> None:
    print(f"\n=== {header} ===")
    for c in cols:
        if c not in df.columns:
            continue
        s = df[c]
        print(
            f"{c:<30} n={s.notna().sum():>4} missing={s.isna().sum():>4} "
            f"median={np.nanmedian(s):>8.4f}"
        )


def build_features() -> pd.DataFrame:
    set_global_seed(42)
    ensure_dirs()
    init_logger()

    print("\n=== Load core inputs ===")
    sentence_qwen = pd.read_csv(QWEN_NO_PROMPT_CSV)
    sentence_leo = pd.read_csv(LEO_NO_PROMPT_CSV)
    
    # Normalize IDs (e.g., '0006_3' -> '0006') to match metadata
    sentence_qwen["file_id"] = sentence_qwen["file_id"].apply(lambda x: str(x).split("_")[0])
    sentence_leo["file_id"] = sentence_leo["file_id"].apply(lambda x: str(x).split("_")[0])

    metadata = load_metadata(str(METADATA_CSV))
    metadata = metadata.rename(columns={"file_id": "participant_id"})

    print("\n=== Aggregate existing LLM perplexities ===")
    qwen_agg = aggregate_perplexity(sentence_qwen, model_name="qwen").rename(columns={"file_id": "participant_id"})
    leo_agg = aggregate_perplexity(sentence_leo, model_name="leo").rename(columns={"file_id": "participant_id"})

    narrative_df = _build_narrative_table(sentence_qwen)

    print("\n=== Feature: average words per sentence ===")
    avg_words = compute_avg_words_per_sentence(sentence_qwen)
    _print_validation(avg_words, ["avg_words_per_sentence"], "avg_words")

    print("\n=== Feature: MATTR ===")
    mattr_df = compute_mattr(narrative_df)
    _print_validation(mattr_df, ["mattr_50"], "mattr")

    print("\n=== Feature: mean log frequency ===")
    freq_df, freq_source = compute_log_frequency_features(narrative_df)
    _print_validation(freq_df, ["mean_log_word_freq_all", "mean_log_word_freq_content"], "log_frequency")

    print("\n=== Feature: content/function ratio ===")
    cfr_df = compute_content_function_ratio(narrative_df)
    _print_validation(cfr_df, ["content_function_ratio"], "content_function_ratio")

    print("\n=== Feature: GPT-2 German perplexity ===")
    #_, gpt2_df = compute_gpt2_sentence_and_participant(sentence_qwen)
    #_print_validation(gpt2_df, ["gpt2_de_ppl_median"], "gpt2_perplexity")

    print("\n=== Feature: semantic coherence ===")
    sem_df, sem_model = compute_semantic_features(sentence_qwen)
    _print_validation(sem_df, ["semantic_coherence_mean", "tangentiality_first_last"], "semantic")

    out = metadata.merge(qwen_agg, on="participant_id", how="inner")
    out = out.merge(leo_agg, on="participant_id", how="left", suffixes=("", "_leo"))
    for feat in [avg_words, mattr_df, freq_df, cfr_df, sem_df]:
        out = out.merge(feat, on="participant_id", how="left")

    out.to_parquet(FEATURES_PARQUET, index=False)
    print(f"\nSaved features table: {FEATURES_PARQUET}")

    manifest = {
        "kenlm_feature": "skipped_by_user_request",
        "frequency_source": freq_source,
        "semantic_model": "none",
        "n_participants": int(len(out)),
    }
    manifest_path = Path(__file__).resolve().parents[0] / "ngram_model" / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved manifest: {manifest_path}")

    return out


if __name__ == "__main__":
    build_features()
