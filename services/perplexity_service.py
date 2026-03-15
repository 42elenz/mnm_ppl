"""Service to compute perplexities for all prompt conditions."""

import torch
import pandas as pd
from functions.ppl.compute_perplexities import (
    PROMPTS,
    explode_to_sentences,
    compute_perplexities,
)


def _load_llm(model_name: str, gpu_memory_utilization: float = 0.80):
    """Load a vLLM model with automatic tensor-parallel sizing."""
    from vllm import LLM

    n_gpus = torch.cuda.device_count()
    valid_tp_sizes = [8, 4, 2, 1]
    tp_size = next(tp for tp in valid_tp_sizes if tp <= n_gpus)
    print(f"Loading {model_name} on {tp_size} GPU(s)...")
    return LLM(
        model=model_name,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=gpu_memory_utilization,
    )


def perplexity_service(
    utterance_df: pd.DataFrame,
    model_name: str = "Qwen/Qwen2.5-32B-Instruct",
    output_folder: str = "../../00_data/02_perplexities",
    conditions: list[str] | None = None,
    overwrite: bool = False,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Compute perplexities for all prompt conditions.

    Args:
        utterance_df: utterance-level DataFrame from preprocessing
        model_name: HuggingFace model identifier
        output_folder: base output directory
        conditions: list of prompt conditions to run (default: all three)
        overwrite: re-compute even if outputs exist

    Returns:
        dict mapping condition name → (data_with_ppl, token_data)
    """
    if conditions is None:
        conditions = list(PROMPTS.keys())

    sentence_df = explode_to_sentences(utterance_df)
    model_short = model_name.split("/")[-1]

    llm = _load_llm(model_name)

    results = {}
    for condition in conditions:
        data_ppl, token_data = compute_perplexities(
            llm=llm,
            sentence_df=sentence_df,
            prompt_condition=condition,
            output_folder=output_folder,
            model_name=model_short,
            overwrite=overwrite,
        )
        results[condition] = (data_ppl, token_data)

    print(f"\nAll {len(conditions)} conditions computed.")
    return results
