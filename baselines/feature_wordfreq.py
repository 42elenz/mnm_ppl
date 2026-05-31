import os
from collections import Counter

import numpy as np
import pandas as pd
import spacy

from baselines.common import log_drop, parse_subtlex


CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV"}


def _load_nlp():
    for model in ["de_core_news_lg", "de_core_news_sm"]:
        try:
            return spacy.load(model, disable=["ner", "parser"])
        except Exception:
            continue
    raise RuntimeError("No spaCy German model found. Install de_core_news_sm or de_core_news_lg.")


def _build_frequency_lexicon(narrative_df: pd.DataFrame) -> tuple[Counter, str]:
    subtlex_path = os.getenv("SUBTLEX_DE_PATH", "").strip()
    if subtlex_path and os.path.exists(subtlex_path):
        return parse_subtlex(pd.Path(subtlex_path) if hasattr(pd, "Path") else __import__("pathlib").Path(subtlex_path)), f"SUBTLEX-DE ({subtlex_path})"

    counts = Counter()
    for text in narrative_df["narrative"].astype(str).tolist():
        for tok in text.lower().split():
            t = "".join(ch for ch in tok if ch.isalpha())
            if t:
                counts[t] += 1
    return counts, "fallback_internal_narrative_counts"


def compute_log_frequency_features(narrative_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    nlp = _load_nlp()
    freq_counts, source = _build_frequency_lexicon(narrative_df)
    total = sum(freq_counts.values())
    eps = 1.0 / max(total, 1)

    rows = []
    for _, row in narrative_df.iterrows():
        pid = row["participant_id"]
        try:
            doc = nlp(str(row["narrative"]))
            all_vals = []
            content_vals = []
            for t in doc:
                if not t.is_alpha:
                    continue
                key = t.text.lower()
                p = freq_counts.get(key, 0) / max(total, 1)
                logf = np.log(p + eps)
                all_vals.append(logf)
                if t.pos_ in CONTENT_POS:
                    content_vals.append(logf)

            rows.append(
                {
                    "participant_id": pid,
                    "mean_log_word_freq_all": float(np.mean(all_vals)) if all_vals else np.nan,
                    "mean_log_word_freq_content": float(np.mean(content_vals)) if content_vals else np.nan,
                }
            )
        except Exception as e:
            log_drop(pid, "mean_log_word_freq", e)
            rows.append(
                {
                    "participant_id": pid,
                    "mean_log_word_freq_all": np.nan,
                    "mean_log_word_freq_content": np.nan,
                }
            )

    return pd.DataFrame(rows), source
