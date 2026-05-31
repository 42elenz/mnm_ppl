import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from baselines.common import log_drop


MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
FALLBACK_MODEL_ID = "sentence-transformers/distiluse-base-multilingual-cased-v1"


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den == 0:
        return np.nan
    return float(np.dot(a, b) / den)


def _load_embedder():
    for mid in [MODEL_ID, FALLBACK_MODEL_ID]:
        try:
            return SentenceTransformer(mid), mid
        except Exception:
            continue
    raise RuntimeError("Could not load sentence embedding model.")


def compute_semantic_features(sentence_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    model, model_id = _load_embedder()
    rows = []
    for pid, grp in sentence_df.groupby("file_id"):
        sents = grp["utterance"].astype(str).tolist()
        try:
            if len(sents) < 2:
                rows.append(
                    {
                        "participant_id": pid,
                        "semantic_coherence_mean": np.nan,
                        "tangentiality_first_last": np.nan,
                    }
                )
                continue
            emb = model.encode(sents, convert_to_numpy=True)
            cons = []
            for i in range(len(emb) - 1):
                cons.append(_cos(emb[i], emb[i + 1]))
            tang = _cos(emb[0], emb[-1])
            rows.append(
                {
                    "participant_id": pid,
                    "semantic_coherence_mean": float(np.nanmean(cons)) if cons else np.nan,
                    "tangentiality_first_last": tang,
                }
            )
        except Exception as e:
            log_drop(pid, "semantic_coherence", e)
            rows.append(
                {
                    "participant_id": pid,
                    "semantic_coherence_mean": np.nan,
                    "tangentiality_first_last": np.nan,
                }
            )

    return pd.DataFrame(rows), model_id
