"""Core function to compute perplexities for TAT German utterances."""

import os
import re
import pandas as pd
from functions.ppl.helper_perplexity import compute_perplexity_metrics_vllm


# German system prompts for the experimental conditions
PROMPTS = {
    "healthy": (
        "Eine psychisch gesunde Person beschreibt ein TAT-Bild. "
        "Die Beschreibung ist überwiegend strukturiert und weist keine "
        "typischen psychopathologischen Auffälligkeiten auf."
    ),
    "sick": (
        "Eine Person mit einer psychischen Erkrankung beschreibt ein TAT-Bild. "
        "Die Beschreibung kann Auffälligkeiten wie formale Denkstörungen, "
        "Inkohärenz oder ungewöhnliche Assoziationen zeigen."
    ),
    "neutral": "Eine Person beschreibt ein TAT-Bild.",
    "no_prompt": "",
}


def split_into_sentences(text: str) -> list[str]:
    """Split German text into sentences at .!? followed by uppercase letter."""
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-ZÄÖÜ])", text)
    return [s.strip() for s in sentences if s.strip()]


def explode_to_sentences(df: pd.DataFrame) -> pd.DataFrame:
    """Expand an utterance-level DataFrame to sentence-level rows."""
    rows = []
    for _, row in df.iterrows():
        for sent in split_into_sentences(row["utterance"]):
            rows.append(
                {
                    "file_id": row["file_id"],
                    "task": row["task"],
                    "utterance": sent,
                    "word_count": len(sent.split()),
                }
            )
    result = pd.DataFrame(rows)
    print(f"Sentence split: {len(df)} utterances → {len(result)} sentences")
    print(f"Average words per sentence: {result['word_count'].mean():.1f}")
    return result


def compute_perplexities(
    llm,
    sentence_df: pd.DataFrame,
    prompt_condition: str,
    output_folder: str,
    model_name: str = "Qwen2.5-32B-Instruct",
    overwrite: bool = False,
):
    """Compute perplexities for a single prompt condition.

    Args:
        llm: loaded vLLM model instance
        sentence_df: DataFrame with columns [file_id, task, utterance, word_count]
        prompt_condition: one of 'healthy', 'sick', 'neutral'
        output_folder: base folder; results saved under {output_folder}/{model}/{prompt_{condition}}/
        model_name: short model name for folder structure
        overwrite: if False, skip if output exists

    Returns:
        (data_with_ppl, token_data) — the augmented DataFrames
    """
    prompt_text = PROMPTS[prompt_condition]
    print(f"\n{'='*60}")
    print(f"Condition: {prompt_condition}")
    print(f"Prompt: {prompt_text}")
    print(f"{'='*60}")

    data_prepared = sentence_df.copy()
    data_prepared["prompt"] = prompt_text
    data_prepared["prompt_condition"] = prompt_condition
    data_prepared["completion"] = data_prepared["utterance"]

    save_dir = os.path.join(output_folder, model_name, f"prompt_{prompt_condition}")
    os.makedirs(save_dir, exist_ok=True)

    suffix = "" if prompt_condition == "healthy" else f"_{prompt_condition}"
    file_name = f"TAT_german_utterances_with_perplexities{suffix}.csv"

    data_with_ppl, token_data = compute_perplexity_metrics_vllm(
        llm,
        data_prepared,
        prompt_col="prompt",
        completion_col="completion",
        save_to_folder=save_dir,
        overwrite=overwrite,
        file_name=file_name,
        return_token_df=True,
    )

    print(f"Shape: {data_with_ppl.shape}")
    print(f"Saved to: {save_dir}")
    return data_with_ppl, token_data
