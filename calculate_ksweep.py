"""
Strict Sliding Window K-Sweep Inference Pipeline.

This script restricts the history fed to the LLM to exactly `k` tokens 
(e.g., k=1, 5, 10, 20, 50) to evaluate whether elevated perplexity in 
psychosis is due to local syntactic failures or global context maintenance.
"""

import os
import gc
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams
from functions.ppl.compute_perplexities import PROMPTS

# ── Paths & Config ─────────────────────────────────────────────────────────
UTTERANCE_CSV = "data/01_processed/TAT_german_utterances.csv"
OUTPUT_DIR = "data/02_perplexities/data/02_perplexities_vllm/Qwen2.5-32B-Instruct/k_sweep"
MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct" 
PROMPT_CONDITION = "healthy"
K_VALUES = [1, 5, 10, 20, 50]
BATCH_SIZE_WINDOWS = 50000  # Number of windows to send to vLLM at once to avoid OOM

def main():
    print("="*60)
    print("Starting Strict Sliding Window K-Sweep")
    print(f"Model: {MODEL_NAME}")
    print(f"K Values: {K_VALUES}")
    print("="*60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Data
    df = pd.read_csv(UTTERANCE_CSV)
    print(f"Loaded {len(df)} utterances.")

    # 2. Initialize Model
    print("\nLoading vLLM model...")
    # Note: Adjust tensor_parallel_size if running on multi-GPU
    llm = LLM(model=MODEL_NAME, tensor_parallel_size=1, max_model_len=4096, trust_remote_code=True)
    tokenizer = llm.get_tokenizer()
    
    prompt_text = PROMPTS[PROMPT_CONDITION]
    prompt_ids = tokenizer.encode(prompt_text)
    
    sampling_params = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=1)

    # 3. Pre-tokenize all utterances to save time
    print("\nPre-tokenizing all utterances...")
    utterance_tokens = []
    for text in tqdm(df["utterance"]):
        utterance_tokens.append(tokenizer.encode(text))

    # 4. Sweep across K values
    for k in K_VALUES:
        print(f"\n{'-'*40}")
        print(f"Processing k = {k}")
        print(f"{'-'*40}")
        
        output_file = os.path.join(OUTPUT_DIR, f"ksweep_k{k}_tokens.csv")
        if os.path.exists(output_file):
            print(f"File {output_file} already exists. Skipping.")
            continue
            
        windows = []
        metadata = []
        
        # Build windows for this k
        for row_idx, utt_ids in enumerate(utterance_tokens):
            # We skip the very first token (index 0) because it has no utterance history
            # (Its history is just the prompt, which is technically k=0). 
            # We start evaluating from the 2nd token (index 1).
            for i in range(1, len(utt_ids)):
                context_start = max(0, i - k)
                # window = Prompt + previous k tokens + target token
                window_ids = prompt_ids + utt_ids[context_start : i + 1]
                
                windows.append(window_ids)
                metadata.append({
                    "row_id": row_idx,
                    "file_id": df.iloc[row_idx]["file_id"],
                    "task": df.iloc[row_idx]["task"],
                    "token_idx": i,
                    "target_token_id": utt_ids[i],
                    "target_token_str": tokenizer.decode([utt_ids[i]]),
                    "k": k
                })
        
        print(f"Built {len(windows)} sliding windows for k={k}.")
        
        # Batch process through vLLM
        results = []
        for batch_start in tqdm(range(0, len(windows), BATCH_SIZE_WINDOWS), desc=f"Inference k={k}"):
            batch_end = min(batch_start + BATCH_SIZE_WINDOWS, len(windows))
            batch_windows = windows[batch_start:batch_end]
            batch_meta = metadata[batch_start:batch_end]
            
            outputs = llm.generate(prompt_token_ids=batch_windows, sampling_params=sampling_params, use_tqdm=False)
            
            for out, meta in zip(outputs, batch_meta):
                # The logprob of the target token is in the last position of the prompt
                try:
                    last_logprob_dict = out.prompt_logprobs[-1]
                    # Since prompt_logprobs=1, the dict has 1 entry: the actual token's logprob
                    logprob_info = list(last_logprob_dict.values())[0]
                    
                    meta["logprob"] = logprob_info.logprob
                    meta["loss"] = -logprob_info.logprob
                    meta["perplexity"] = np.exp(-logprob_info.logprob)
                    results.append(meta)
                except Exception as e:
                    meta["logprob"] = np.nan
                    meta["loss"] = np.nan
                    meta["perplexity"] = np.nan
                    results.append(meta)
                    
        # Save token-level results for this k
        k_df = pd.DataFrame(results)
        k_df.to_csv(output_file, index=False)
        print(f"Saved {len(k_df)} token records to {output_file}")
        
        # Free memory before next K
        del windows, metadata, results, k_df
        gc.collect()

    print("\n" + "="*60)
    print("K-Sweep Inference Complete!")
    print("="*60)

if __name__ == "__main__":
    main()
