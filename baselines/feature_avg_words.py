import pandas as pd


def compute_avg_words_per_sentence(sentence_df: pd.DataFrame) -> pd.DataFrame:
    out = (
        sentence_df.groupby("file_id", as_index=False)["word_count"]
        .mean()
        .rename(columns={"file_id": "participant_id", "word_count": "avg_words_per_sentence"})
    )
    return out
