import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import statsmodels.formula.api as smf

# Make sure project root is on the path
sys.path.insert(0, os.path.abspath('.'))

from services.analysis_service import load_metadata, _normalize_subject_id

# Configure paths
MODEL_NAME = "Qwen2.5-32B-Instruct"
PROMPT_CONDITION = "healthy"
# Adjusting to the discovered nested path structure
DATA_DIR = f"data/02_perplexities/data/02_perplexities_vllm/{MODEL_NAME}/prompt_{PROMPT_CONDITION}"
TOKEN_CSV = os.path.join(DATA_DIR, f"TAT_german_utterances_with_perplexities_tokens_{PROMPT_CONDITION}.csv")
SENTENCE_CSV = os.path.join(DATA_DIR, f"TAT_german_utterances_with_perplexities_{PROMPT_CONDITION}.csv")
METADATA_CSV = "data/00_raw_data/speech_metadata.csv"
FIGURE_DIR = "figures/positional_loss"

def main():
    print(f"Loading data for {MODEL_NAME} ({PROMPT_CONDITION})...")
    
    # Check if files exist
    for f in [TOKEN_CSV, SENTENCE_CSV, METADATA_CSV]:
        if not os.path.exists(f):
            print(f"Error: File not found: {f}")
            return

    # 1. Load metadata
    meta = load_metadata(METADATA_CSV)
    
    # 2. Load sentence data to get file_id mapping
    sentence_df = pd.read_csv(SENTENCE_CSV)
    sentence_df["file_id"] = _normalize_subject_id(sentence_df["file_id"])
    
    # 3. Load token data
    # Note: Token CSV can be large, loading in chunks or being careful with memory
    print("Reading token-level data (this may take a moment)...")
    token_df = pd.read_csv(TOKEN_CSV)
    
    # 4. Filter for completion tokens only
    token_df = token_df[token_df["is_completion"] == True].copy()
    
    # 5. Map file_id, task, and diagnosis to tokens
    # row_id in token_df corresponds to the index of sentence_df
    token_df = token_df.merge(
        sentence_df[["file_id", "task"]], 
        left_on="row_id", 
        right_index=True
    )
    
    token_df = token_df.merge(
        meta[["file_id", "diagnosis"]], 
        on="file_id"
    )
    
    print(f"Total tokens after merge: {len(token_df)}")
    
    # 6. Recalculate relative token positions
    # Position within sentence
    token_df["sent_token_idx"] = token_df.groupby("row_id").cumcount()
    
    # Position within entire discourse (Task)
    # To do this correctly, we need to preserve the order of sentences within a task
    # Assuming sentence_df is already ordered by task and appearance.
    token_df["discourse_token_idx"] = token_df.groupby(["file_id", "task"]).cumcount()
    
    # 7. Grouping for broader comparison
    # SSD (Schizophrenia Spectrum Disorder) and BD (Bipolar Disorder)
    token_df["group"] = token_df["diagnosis"].replace({"SSD": "Psychosis", "BD": "Psychosis"})
    
    # 8. Plotting
    os.makedirs(FIGURE_DIR, exist_ok=True)
    sns.set_style("whitegrid")
    
    # For discourse-level, we can look further out, e.g., 150 tokens
    MAX_DISCOURSE_TOKENS = 150
    plot_df = token_df[token_df["discourse_token_idx"] < MAX_DISCOURSE_TOKENS].copy()

    # Plot 1: Detailed Diagnostic Groups (Discourse Level)
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=plot_df,
        x="discourse_token_idx",
        y="loss",
        hue="diagnosis",
        hue_order=["HC", "SSD", "BD", "MDD"],
        palette="viridis",
        errorbar=("ci", 95)
    )
    plt.title(f"Global Context Analysis ({MODEL_NAME})\nLoss by Discourse Position (Prompt: {PROMPT_CONDITION})")
    plt.xlabel("Token Position in Task (TAT Description)")
    plt.ylabel("Cross-Entropy Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "loss_by_diagnosis_discourse.png"), dpi=300)
    plt.close()

    # Plot 2: Psychosis Spectrum vs. HC (Discourse Level)
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=plot_df,
        x="discourse_token_idx",
        y="loss",
        hue="group",
        hue_order=["HC", "Psychosis", "MDD"],
        palette="Set2",
        errorbar=("ci", 95)
    )
    plt.title(f"Global Context Analysis ({MODEL_NAME})\nPsychosis Spectrum vs. Controls")
    plt.xlabel("Token Position in Task (TAT Description)")
    plt.ylabel("Cross-Entropy Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "loss_by_psychosis_group_discourse.png"), dpi=300)
    plt.close()
    
    print(f"Plots saved to {FIGURE_DIR}")
    
    # 9. Statistical Analysis: Linear Mixed Effects Model
    # Hypothesis: Loss increases more steeply (or starts higher and diverges) in Psychosis group
    # as global discourse context length increases.
    
    print("\nRunning Linear Mixed Effects Model (loss ~ discourse_token_idx * group)...")
    stats_df = token_df[token_df["group"].isin(["HC", "Psychosis"])].copy()
    stats_df["group"] = pd.Categorical(stats_df["group"], categories=["HC", "Psychosis"])
    
    try:
        model = smf.mixedlm("loss ~ discourse_token_idx * group", stats_df, groups=stats_df["file_id"])
        result = model.fit()
        print(result.summary())
        
        with open(os.path.join(FIGURE_DIR, "stats_summary_discourse.txt"), "w") as f:
            f.write("=== Global Context (Discourse) Interaction Analysis ===\n")
            f.write(f"Model: {MODEL_NAME}, Prompt: {PROMPT_CONDITION}\n")
            f.write("Hypothesis: Psychosis group shows higher loss as discourse context length increases.\n\n")
            f.write(result.summary().as_text())
    except Exception as e:
        print(f"Statistical model failed: {e}")


if __name__ == "__main__":
    main()
