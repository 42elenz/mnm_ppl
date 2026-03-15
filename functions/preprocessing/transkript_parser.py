"""
Script to parse TAT text files and convert them to a structured DataFrame format.

Output format:
- file_id: ID from first line of file (e.g., "0006_3")
- task: Bild number (e.g., 1, 2, 4, 6)
- utterance: Text content for that Bild
- word_count: Number of words in the utterance
"""

import os
import re
import pandas as pd
from pathlib import Path


def parse_tat_file(filepath: str) -> list[dict]:
    """
    Parse a single TAT text file and extract all Bild sections.
    
    Args:
        filepath: Path to the text file
        
    Returns:
        List of dictionaries with file_id, task, utterance, word_count
    """
    # Try different encodings
    content = None
    for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                content = f.read()
            break
        except UnicodeDecodeError:
            continue
    
    if content is None:
        raise ValueError(f"Could not decode file with any supported encoding")
    
    lines = content.strip().split('\n')
    
    # First non-empty line is the file_id
    file_id = None
    for line in lines:
        line = line.strip()
        if line:
            file_id = line
            break
    
    if not file_id:
        print(f"Warning: Could not find file_id in {filepath}")
        return []
    
    # Find all "Bild X" sections using regex
    # Pattern matches "Bild" followed by whitespace and a number, possibly with a period
    bild_pattern = re.compile(r'^Bild\s+(\d+)\.?\s*$', re.IGNORECASE | re.MULTILINE)
    
    # Find all matches and their positions
    matches = list(bild_pattern.finditer(content))
    
    if not matches:
        print(f"Warning: No 'Bild' sections found in {filepath}")
        return []
    
    results = []
    
    for i, match in enumerate(matches):
        task_num = int(match.group(1))
        
        # Get start position (after the "Bild X" line)
        start_pos = match.end()
        
        # Get end position (start of next "Bild X" or end of file)
        if i < len(matches) - 1:
            end_pos = matches[i + 1].start()
        else:
            end_pos = len(content)
        
        # Extract the utterance text
        utterance = content[start_pos:end_pos].strip()
        
        # Clean up the utterance - remove extra whitespace but keep single spaces
        utterance = ' '.join(utterance.split())
        
        # Calculate word count
        word_count = len(utterance.split())
        
        results.append({
            'file_id': file_id,
            'task': task_num,
            'utterance': utterance,
            'word_count': word_count
        })
    
    return results


def process_all_files(input_dir: str, output_path: str = None) -> pd.DataFrame:
    """
    Process all TAT text files in a directory.
    
    Args:
        input_dir: Directory containing the text files
        output_path: Optional path to save the CSV output
        
    Returns:
        DataFrame with all parsed data
    """
    input_path = Path(input_dir)
    
    all_records = []
    processed_files = 0
    skipped_files = 0
    
    # Get all .txt files
    txt_files = sorted(input_path.glob('*.txt'))
    
    print(f"Found {len(txt_files)} text files to process...")
    
    for txt_file in txt_files:
        try:
            records = parse_tat_file(str(txt_file))
            all_records.extend(records)
            processed_files += 1
        except Exception as e:
            print(f"Error processing {txt_file.name}: {e}")
            skipped_files += 1
    
    print(f"\nProcessed {processed_files} files successfully")
    if skipped_files > 0:
        print(f"Skipped {skipped_files} files due to errors")
    
    # Create DataFrame
    df = pd.DataFrame(all_records)
    
    if len(df) > 0:
        # Sort by file_id and task
        df = df.sort_values(['file_id', 'task']).reset_index(drop=True)
        
        print(f"\nTotal records: {len(df)}")
        print(f"Unique file IDs: {df['file_id'].nunique()}")
        print(f"Task distribution:\n{df['task'].value_counts().sort_index()}")
        print(f"\nWord count statistics:")
        print(df['word_count'].describe())
    
    # Save if output path provided
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"\nSaved to: {output_path}")
    
    return df



