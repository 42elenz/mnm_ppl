import numpy as np
import pandas as pd
import spacy

from baselines.common import log_drop


CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV"}


def _load_nlp():
    for model in ["de_core_news_lg", "de_core_news_sm"]:
        try:
            return spacy.load(model, disable=["ner", "parser"])
        except Exception:
            continue
    raise RuntimeError("No spaCy German model found. Install de_core_news_sm or de_core_news_lg.")


def compute_content_function_ratio(narrative_df: pd.DataFrame) -> pd.DataFrame:
    nlp = _load_nlp()
    rows = []
    for _, row in narrative_df.iterrows():
        pid = row["participant_id"]
        try:
            doc = nlp(str(row["narrative"]))
            content = 0
            function = 0
            for t in doc:
                if not t.is_alpha:
                    continue
                if t.pos_ in CONTENT_POS:
                    content += 1
                else:
                    function += 1
            ratio = content / function if function > 0 else np.nan
            rows.append({"participant_id": pid, "content_function_ratio": ratio})
        except Exception as e:
            log_drop(pid, "content_function_ratio", e)
            rows.append({"participant_id": pid, "content_function_ratio": np.nan})
    return pd.DataFrame(rows)
