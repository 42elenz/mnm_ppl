import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import sys

# Make sure project root is on the path
sys.path.insert(0, os.path.abspath('.'))

from services.analysis_service import load_metadata, _normalize_subject_id

# Configure paths
MODEL_NAME = "Qwen2.5-32B-Instruct"
PROMPT_CONDITION = "healthy"
DATA_DIR = f"data/02_perplexities/data/02_perplexities_vllm/{MODEL_NAME}/prompt_{PROMPT_CONDITION}"
TOKEN_CSV = os.path.join(DATA_DIR, f"TAT_german_utterances_with_perplexities_tokens_{PROMPT_CONDITION}.csv")
SENTENCE_CSV = os.path.join(DATA_DIR, f"TAT_german_utterances_with_perplexities_{PROMPT_CONDITION}.csv")
METADATA_CSV = "data/00_raw_data/speech_metadata.csv"
OUTPUT_DIR = "figures/surprisal_visualizations"

def get_html_template(title, body_content):
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                padding: 40px;
                background-color: #f5f5f5;
                color: #333;
            }}
            .container {{
                max-width: 900px;
                margin: auto;
                background: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
            .metadata {{
                background: #f8f9fa;
                padding: 15px;
                border-left: 5px solid #3498db;
                margin-bottom: 20px;
                font-size: 0.9em;
            }}
            .transcript {{
                font-size: 1.2em;
                white-space: pre-wrap;
                word-wrap: break-word;
            }}
            .token {{
                padding: 2px 1px;
                border-radius: 2px;
                cursor: help;
                transition: transform 0.1s;
                display: inline;
            }}
            .token:hover {{
                outline: 2px solid #333;
                z-index: 10;
            }}
            .tooltip {{
                position: relative;
                display: inline-block;
            }}
            .legend {{
                margin-top: 30px;
                padding: 10px;
                border-top: 1px solid #eee;
                font-size: 0.8em;
                color: #666;
            }}
            .color-box {{
                display: inline-block;
                width: 20px;
                height: 10px;
                margin-right: 5px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{title}</h1>
            {body_content}
            <div class="legend">
                <p><strong>Legend:</strong> Darker red indicates higher surprisal (Cross-Entropy Loss). 
                Hover over words to see exact values.</p>
                <div><span class="color-box" style="background-color: #fff;"></span> Low Surprisal</div>
                <div><span class="color-box" style="background-color: #fc9272;"></span> Medium Surprisal</div>
                <div><span class="color-box" style="background-color: #de2d26;"></span> High Surprisal</div>
            </div>
        </div>
    </body>
    </html>
    """

def create_visualization(file_id, diagnosis, tli_score, tokens_df):
    """Generates an HTML visualization for a specific subject's transcript."""
    
    # Filter tokens for this subject
    subj_tokens = tokens_df[tokens_df["file_id"] == file_id].copy()
    
    if subj_tokens.empty:
        print(f"No tokens found for {file_id}")
        return

    # Normalize loss for color mapping (using 95th percentile across all tokens as max)
    # We use a fixed range 0-10 for consistency across subjects if possible, 
    # or dynamic based on this subject. Let's use 0-8 as a reasonable range for Reds.
    norm = mcolors.Normalize(vmin=0, vmax=8)
    cmap = plt.cm.Reds

    body_html = f"""
    <div class="metadata">
        <strong>Subject ID:</strong> {file_id}<br>
        <strong>Diagnosis:</strong> {diagnosis}<br>
        <strong>TLI Disorganization Score:</strong> {tli_score:.2f}
    </div>
    <div class="transcript">"""

    for _, row in subj_tokens.iterrows():
        loss = row["loss"]
        # Skip NaNs
        if pd.isna(loss):
            color_hex = "#fff"
        else:
            rgba = cmap(norm(loss))
            color_hex = mcolors.to_hex(rgba)
        
        token_str = row["token_str"].replace("\n", "<br>")
        # Escape HTML if necessary, though these are likely clean
        
        body_html += f'<span class="token" style="background-color: {color_hex};" title="Loss: {loss:.3f} | Token: \'{row["token_str"]}\' | Pos: {row["token_idx"]}">{token_str}</span>'

    body_html += "</div>"
    
    title = f"Surprisal Attribution: {diagnosis} ({file_id})"
    html_content = get_html_template(title, body_html)
    
    output_path = os.path.join(OUTPUT_DIR, f"surprisal_{diagnosis}_{file_id}.html")
    with open(output_path, "w") as f:
        f.write(html_content)
    
    print(f"Generated: {output_path}")

def main():
    print("Selecting representative subjects for visualization...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Metadata
    meta = load_metadata(METADATA_CSV)
    
    # 2. Select Candidates
    # HC: Just pick the first one or one with average length
    hc_id = meta[meta["diagnosis"] == "HC"]["file_id"].iloc[0]
    
    # SSD: Highest TLI Disorganization
    ssd_candidate = meta[meta["diagnosis"] == "SSD"].sort_values("TLI_Disorganization", ascending=False).iloc[0]
    ssd_id = ssd_candidate["file_id"]
    ssd_tli = ssd_candidate["TLI_Disorganization"]
    
    # BD: Highest TLI Disorganization
    bd_candidate = meta[meta["diagnosis"] == "BD"].sort_values("TLI_Disorganization", ascending=False).iloc[0]
    bd_id = bd_candidate["file_id"]
    bd_tli = bd_candidate["TLI_Disorganization"]
    
    candidates = [
        (hc_id, "HC", 0.0),
        (ssd_id, "SSD", ssd_tli),
        (bd_id, "BD", bd_tli)
    ]
    
    print(f"Selected HC: {hc_id}")
    print(f"Selected SSD: {ssd_id} (TLI: {ssd_tli})")
    print(f"Selected BD: {bd_id} (TLI: {bd_tli})")
    
    # 3. Load Token Data
    print("Loading token-level data...")
    token_df = pd.read_csv(TOKEN_CSV)
    sentence_df = pd.read_csv(SENTENCE_CSV)
    sentence_df["file_id"] = _normalize_subject_id(sentence_df["file_id"])
    
    # Filter for completion tokens
    token_df = token_df[token_df["is_completion"] == True].copy()
    
    # Map file_id to tokens
    token_df = token_df.merge(
        sentence_df[["file_id"]], 
        left_on="row_id", 
        right_index=True
    )
    
    # 4. Generate Visualizations
    for file_id, diag, tli in candidates:
        create_visualization(file_id, diag, tli, token_df)

    print("\nVisualizations complete! Check the 'figures/surprisal_visualizations' folder.")

if __name__ == "__main__":
    main()
