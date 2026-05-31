# Baselines Summary

## Headline Numbers
- Qwen raw d (HC vs patients): 0.210
- Best classical baseline raw d: mean_log_word_freq_content = -0.000
- Qwen partial correlation survives for YMRS after all classical controls: True
- Qwen partial correlation survives for SAPS after all classical controls: True

## Conclusion
Qwen shows additional signal beyond classical baselines and smaller LMs in this no-KenLM comparison pipeline.
The nested models indicate incremental discrimination gains when Qwen is added after confounders, classical features, GPT-2, and Leo.
Clinical associations are partially robust after controlling for classical language features.