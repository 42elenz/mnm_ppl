import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

def check_confounder_neutrality(df: pd.DataFrame, raw_col: str, resid_col: str, confounders: list[str]) -> pd.DataFrame:
    """
    Check if residualization successfully removed the correlation with confounders.
    
    Returns a DataFrame with Spearman correlations between each confounder and 
    both the raw and residualized variables.
    """
    rows = []
    for c in confounders:
        if c not in df.columns:
            continue
            
        # Raw Correlation
        sub_raw = df[[raw_col, c]].dropna()
        if len(sub_raw) > 2:
            r_raw, p_raw = spearmanr(sub_raw[raw_col], sub_raw[c])
        else:
            r_raw, p_raw = np.nan, np.nan
            
        # Residualized Correlation
        sub_res = df[[resid_col, c]].dropna()
        if len(sub_res) > 2:
            r_res, p_res = spearmanr(sub_res[resid_col], sub_res[c])
        else:
            r_res, p_res = np.nan, np.nan
        
        rows.append({
            "Confounder": c,
            "r (Raw)": r_raw, "p (Raw)": p_raw,
            "r (Resid)": r_res, "p (Resid)": p_res
        })
    return pd.DataFrame(rows)

def plot_neutrality_check(df: pd.DataFrame, raw_col: str, resid_col: str, confounder: str, save_path: str = None):
    """
    Visual check of deconfounding: Scatter + Regression line before and after.
    The 'after' plot should ideally show a horizontal line (zero slope).
    """
    # Mapping for cleaner labels in plots
    name_map = {
        "qwen_loss_median": "Raw Loss",
        "qwen_loss_median_resid": "Residualized Loss",
        "log_avg_words": "Avg words (log)",
        "qwen_avg_words": "Avg words",
        "sex_num": "Sex",
        "age": "Age",
        "education": "Education"
    }
    
    def clean(name):
        if name in name_map: return name_map[name]
        return name.replace("_", " ").title()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=False)
    
    # Panel A: Raw Relationship
    sns.regplot(
        data=df, x=confounder, y=raw_col, ax=axes[0], 
        scatter_kws={'alpha':0.4, 's':20}, line_kws={'color':'red'}
    )
    axes[0].set_title(f"Unadjusted Relationship")
    axes[0].set_xlabel(clean(confounder))
    axes[0].set_ylabel(clean(raw_col))
    axes[0].grid(alpha=0.3)
    
    # Panel B: Residualized Relationship
    sns.regplot(
        data=df, x=confounder, y=resid_col, ax=axes[1], 
        scatter_kws={'alpha':0.4, 's':20}, line_kws={'color':'green'}
    )
    axes[1].set_title(f"Residualized Relationship")
    axes[1].set_xlabel(clean(confounder))
    axes[1].set_ylabel(clean(resid_col))
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved neutrality check plot: {save_path}")
    plt.show()
    return fig
