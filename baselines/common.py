import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
PERPLEXITY_DIR = DATA_DIR / "02_perplexities" / "data" / "02_perplexities_vllm"
METADATA_CSV = DATA_DIR / "00_raw_data" / "speech_metadata.csv"
QWEN_NO_PROMPT_CSV = (
    PERPLEXITY_DIR
    / "Qwen2.5-32B-Instruct"
    / "prompt_no_prompt"
    / "TAT_german_utterances_with_perplexities_no_prompt.csv"
)
LEO_NO_PROMPT_CSV = (
    PERPLEXITY_DIR
    / "leo-mistral-hessianai-7b-chat"
    / "prompt_no_prompt"
    / "TAT_german_utterances_with_perplexities_no_prompt.csv"
)
BASELINES_DIR = BASE_DIR / "baselines"
RESULTS_DIR = BASELINES_DIR / "results"
FEATURES_PARQUET = BASELINES_DIR / "features_per_participant.parquet"
DROPPED_LOG = BASELINES_DIR / "dropped_participants.log"


def set_global_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dirs() -> None:
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def init_logger() -> None:
    ensure_dirs()
    DROPPED_LOG.write_text("participant_id\tfeature\terror\n", encoding="utf-8")


def log_drop(participant_id: str, feature: str, error: Exception | str) -> None:
    with DROPPED_LOG.open("a", encoding="utf-8") as f:
        f.write(f"{participant_id}\t{feature}\t{str(error)}\n")


def median_by_participant(sentence_df: pd.DataFrame, value_col: str, out_col: str) -> pd.DataFrame:
    return (
        sentence_df.groupby("file_id", as_index=False)[value_col]
        .median()
        .rename(columns={"file_id": "participant_id", value_col: out_col})
    )


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, alpha: float = 0.05) -> tuple[float, float]:
    if len(values) == 0:
        return np.nan, np.nan
    idx = np.random.randint(0, len(values), size=(n_boot, len(values)))
    boots = values[idx]
    means = boots.mean(axis=1)
    lo = np.quantile(means, alpha / 2)
    hi = np.quantile(means, 1 - alpha / 2)
    return float(lo), float(hi)


def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or len(y) < 2:
        return np.nan
    sx = np.std(x, ddof=1)
    sy = np.std(y, ddof=1)
    pooled = np.sqrt(((len(x) - 1) * sx * sx + (len(y) - 1) * sy * sy) / (len(x) + len(y) - 2))
    if pooled == 0:
        return np.nan
    return float((np.mean(x) - np.mean(y)) / pooled)


def bootstrap_cohen_d(x: np.ndarray, y: np.ndarray, n_boot: int = 1000, alpha: float = 0.05) -> tuple[float, float]:
    if len(x) < 2 or len(y) < 2:
        return np.nan, np.nan
    ds = np.empty(n_boot)
    for i in range(n_boot):
        xb = x[np.random.randint(0, len(x), len(x))]
        yb = y[np.random.randint(0, len(y), len(y))]
        ds[i] = cohen_d(xb, yb)
    return float(np.nanquantile(ds, alpha / 2)), float(np.nanquantile(ds, 1 - alpha / 2))


def parse_subtlex(path: Path) -> Counter:
    sep_candidates = ["\t", ",", ";"]
    best_df = None
    for sep in sep_candidates:
        try:
            tmp = pd.read_csv(path, sep=sep)
            if tmp.shape[1] >= 2:
                best_df = tmp
                break
        except Exception:
            continue
    if best_df is None:
        raise ValueError("Could not parse SUBTLEX file")

    cols = {c.lower(): c for c in best_df.columns}
    word_col = None
    freq_col = None
    for cand in ["word", "lemma", "w", "ortho"]:
        if cand in cols:
            word_col = cols[cand]
            break
    for cand in ["freqcount", "frequency", "f", "count", "frq"]:
        if cand in cols:
            freq_col = cols[cand]
            break
    if word_col is None or freq_col is None:
        word_col = best_df.columns[0]
        freq_col = best_df.columns[1]

    counts = Counter()
    for _, row in best_df[[word_col, freq_col]].dropna().iterrows():
        token = str(row[word_col]).strip().lower()
        if not token:
            continue
        try:
            val = float(row[freq_col])
        except Exception:
            continue
        if val > 0:
            counts[token] += val
    return counts
