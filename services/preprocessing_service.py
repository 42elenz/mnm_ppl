import os
import pandas as pd
from pathlib import Path
from functions.preprocessing.transkript_parser import process_all_files

def preprocssing_service(input_directory="../../01_data_german/00_texts", 
                         output_file="../../00_data/01_processed/TAT_german_utterances.csv",
                         overwrite=True,
                         filter_filler_words=False
                         ):

    if not overwrite:
        if os.path.exists(output_file):
            print(f"Output file '{output_file}' already exists. Skipping processing.")
            return pd.read_csv(output_file) 
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df = process_all_files(input_directory, output_file, filter_filler_words=filter_filler_words)
    print("\n" + "="*50)
    print("Sample output:")
    print("="*50)
    print(df.head(10).to_string())
    return df