"""
mnm_ppl — Perplexity-based normative modelling pipeline.

Steps:
  1. Preprocessing: parse raw TAT text files → sentence CSV
  2. Perplexity computation: run vLLM inference for all prompt conditions
"""

from services.preprocessing_service import preprocssing_service
from services.perplexity_service import perplexity_service


# ── Paths ──────────────────────────────────────────────────────────────────
INPUT_TEXTS_DIR = "/zi/home/esra.lenz/Documents/00_HITKIP/11_normative/mnm_ppl/data/00_raw_data/00_texts/"#"data/00_raw_data/00_texts/"
UTTERANCE_CSV = "/zi/home/esra.lenz/Documents/00_HITKIP/11_normative/mnm_ppl/data/01_processed/TAT_german_utterances.csv"#data/01_processed/TAT_german_utterances.csv"
PERPLEXITY_OUTPUT = "data/02_perplexities_vllm"
MODEL_NAME = "LeoLM/leo-mistral-hessianai-7b-chat" #"Qwen/Qwen2.5-32B-Instruct" # "LeoLM/leo-mistral-hessianai-7b-chat"

def main():
    # Step 1: Preprocessing — parse TAT .txt files into utterance CSV
    print("\n" + "=" * 60)
    print("STEP 1: Preprocessing")
    print("=" * 60)
    utterance_df = preprocssing_service(
        input_directory=INPUT_TEXTS_DIR,
        output_file=UTTERANCE_CSV,
        overwrite=True,
        filter_filler_words=True
    )

    # Step 2: Compute perplexities for all prompt conditions
    print("\n" + "=" * 60)
    print("STEP 2: Perplexity computation")
    print("=" * 60)
    perplexity_service(
        utterance_df=utterance_df,
        model_name=MODEL_NAME,
        output_folder=PERPLEXITY_OUTPUT,
        conditions=["healthy"],
        overwrite=False,
    )

    print("\n" + "=" * 60)
    print("Pipeline complete. Run the analysis notebook for results.")
    print("=" * 60)


if __name__ == "__main__":
    main()
