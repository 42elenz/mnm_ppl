import numpy as np
import pandas as pd
import spacy

from baselines.common import log_drop


WINDOW = 50


def _load_nlp():
    for model in ["de_core_news_lg", "de_core_news_sm"]:
        try:
            return spacy.load(model, disable=["ner", "parser"])
        except Exception:
            continue
    raise RuntimeError("No spaCy German model found. Install de_core_news_sm or de_core_news_lg.")


def _mattr(tokens: list[str], window: int = WINDOW) -> float:
    if len(tokens) < 2:
        return np.nan
    if len(tokens) < window:
        return len(set(tokens)) / len(tokens)
    vals = []
    for i in range(len(tokens) - window + 1):
        span = tokens[i : i + window]
        vals.append(len(set(span)) / window)
    return float(np.mean(vals))


def compute_mattr(narrative_df: pd.DataFrame) -> pd.DataFrame:
    nlp = _load_nlp()
    rows = []
    for _, row in narrative_df.iterrows():
        pid = row["participant_id"]
        try:
            doc = nlp(str(row["narrative"]))
            tokens = [t.text.lower() for t in doc if t.is_alpha]
            rows.append({"participant_id": pid, "mattr_50": _mattr(tokens, WINDOW)})
        except Exception as e:
            log_drop(pid, "mattr_50", e)
            rows.append({"participant_id": pid, "mattr_50": np.nan})
    return pd.DataFrame(rows)
