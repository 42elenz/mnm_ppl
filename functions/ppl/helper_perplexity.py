import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple
from vllm import SamplingParams


def compute_perplexity_metrics_vllm(
    llm,
    df: pd.DataFrame,
    prompt_col: str = "prompt",
    completion_col: str = "completion",
    save_to_folder: str = None,
    file_name: str = "df_with_perplexities.csv",
    overwrite: bool = False,
    batch_size: int = 4,
    return_token_df: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute perplexity metrics using vLLM.

    Returns the original df augmented with perplexity columns and,
    optionally, a token-level DataFrame.
    """
    if save_to_folder is not None:
        os.makedirs(save_to_folder, exist_ok=True)
        save_path = os.path.join(save_to_folder, file_name)
        token_save_path = os.path.join(
            save_to_folder, file_name.replace(".csv", "_tokens.csv")
        )
    else:
        save_path = file_name
        token_save_path = file_name.replace(".csv", "_tokens.csv")

    if os.path.exists(save_path) and not overwrite:
        print(f"File {save_path} already exists. Loading existing file.")
        result_df = pd.read_csv(save_path)
        if return_token_df and os.path.exists(token_save_path):
            token_df = pd.read_csv(token_save_path)
            return result_df, token_df
        return (result_df, None) if return_token_df else result_df

    full_texts = [
        f"{row[prompt_col]}{row[completion_col]}" for _, row in df.iterrows()
    ]
    prompts = df[prompt_col].tolist()
    completions = df[completion_col].tolist()
    row_ids = df["id"].tolist() if "id" in df.columns else list(range(len(df)))

    sampling_params = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=1)

    # Pre-tokenize all prompts to get their token counts (avoids a second
    # llm.generate() call per batch — roughly halves inference time).
    tokenizer = llm.get_tokenizer()
    prompt_token_counts = [len(tokenizer.encode(p)) for p in prompts]

    perplexities_full = []
    perplexities_completion = []
    avg_losses_full = []
    avg_losses_completion = []
    token_records = []

    print("Computing perplexities with vLLM...")

    for i in tqdm(range(0, len(full_texts), batch_size)):
        batch_full = full_texts[i : i + batch_size]
        batch_ids = row_ids[i : i + batch_size]
        batch_prompt_token_counts = prompt_token_counts[i : i + batch_size]

        outputs_full = llm.generate(batch_full, sampling_params)

        for j, out_full in enumerate(outputs_full):
            row_id = batch_ids[j]
            prompt_token_count = batch_prompt_token_counts[j]

            if out_full.prompt_logprobs and out_full.prompt_token_ids:
                token_ids = out_full.prompt_token_ids
                logprobs_list = out_full.prompt_logprobs

                char_pos = 0
                word_idx = 0
                current_word = ""

                for token_idx, (token_id, token_logprob_dict) in enumerate(
                    zip(token_ids, logprobs_list)
                ):
                    if token_idx == 0 or token_logprob_dict is None:
                        continue

                    logprob_info = list(token_logprob_dict.values())[0]
                    logprob = logprob_info.logprob
                    token_str = (
                        logprob_info.decoded_token
                        if hasattr(logprob_info, "decoded_token")
                        else ""
                    )

                    is_completion = token_idx >= prompt_token_count
                    token_loss = -logprob
                    token_ppl = np.exp(token_loss)

                    token_stripped = token_str.strip()
                    starts_new_word = token_str.startswith(
                        " "
                    ) or token_str.startswith("\n")

                    if starts_new_word and current_word:
                        word_idx += 1
                        current_word = token_stripped
                    else:
                        current_word += token_stripped

                    token_records.append(
                        {
                            "row_id": row_id,
                            "token_idx": token_idx,
                            "token_id": token_id,
                            "token_str": token_str,
                            "logprob": logprob,
                            "loss": token_loss,
                            "perplexity": token_ppl,
                            "is_completion": is_completion,
                            "word_idx": word_idx,
                            "word_partial": current_word,
                        }
                    )

                logprobs_full = [
                    r["logprob"]
                    for r in token_records
                    if r["row_id"] == row_id and r["token_idx"] > 0
                ]
                logprobs_completion = [
                    r["logprob"]
                    for r in token_records
                    if r["row_id"] == row_id and r["is_completion"]
                ]

                if logprobs_full:
                    avg_loss_full = -np.mean(logprobs_full)
                    ppl_full = np.exp(avg_loss_full)
                else:
                    avg_loss_full, ppl_full = float("nan"), float("nan")

                if logprobs_completion:
                    avg_loss_completion = -np.mean(logprobs_completion)
                    ppl_completion = np.exp(avg_loss_completion)
                else:
                    avg_loss_completion, ppl_completion = float("nan"), float("nan")
            else:
                avg_loss_full, ppl_full = float("nan"), float("nan")
                avg_loss_completion, ppl_completion = float("nan"), float("nan")

            perplexities_full.append(ppl_full)
            perplexities_completion.append(ppl_completion)
            avg_losses_full.append(avg_loss_full)
            avg_losses_completion.append(avg_loss_completion)

    df = df.copy()
    df["full_texts"] = full_texts
    df["prompts"] = prompts
    df["completions"] = completions
    df["perplexity_full"] = perplexities_full
    df["perplexity_completion"] = perplexities_completion
    df["avg_loss_full"] = avg_losses_full
    df["avg_loss_completion"] = avg_losses_completion
    df["word_count_full"] = [len(t.split()) for t in full_texts]
    df["word_count_completion"] = [len(c.split()) for c in completions]

    token_df = pd.DataFrame(token_records)

    df.to_csv(save_path, index=False)
    token_df.to_csv(token_save_path, index=False)
    print(f"Saved to {save_path} and {token_save_path}")

    if return_token_df:
        return df, token_df
    return df
