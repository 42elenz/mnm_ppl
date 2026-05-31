import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from baselines.common import log_drop, median_by_participant


MODEL_ID = "dbmdz/german-gpt2"


def _load_model(model_id: str = MODEL_ID):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device


def _sentence_perplexity(sentence: str, tokenizer, model, device: str) -> float:
    enc = tokenizer(sentence, return_tensors="pt", truncation=True)
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    if input_ids.shape[1] < 2:
        return np.nan
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn)
        logits = out.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            reduction="mean",
        )
    return float(torch.exp(loss).item())


def compute_gpt2_sentence_and_participant(sentence_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tokenizer, model, device = _load_model()
    out_rows = []
    for _, row in sentence_df.iterrows():
        pid = row["file_id"]
        sent = str(row["utterance"])
        try:
            ppl = _sentence_perplexity(sent, tokenizer, model, device)
            out_rows.append({"file_id": pid, "gpt2_de_sentence_ppl": ppl})
        except Exception as e:
            log_drop(pid, "gpt2_de_sentence_ppl", e)
            out_rows.append({"file_id": pid, "gpt2_de_sentence_ppl": np.nan})
    sent_out = pd.DataFrame(out_rows)
    part_out = median_by_participant(sent_out.rename(columns={"file_id": "participant_id"}), "gpt2_de_sentence_ppl", "gpt2_de_ppl_median")
    return sent_out, part_out
