"""Reusable statistical functions for perplexity analysis."""

import numpy as np
import pandas as pd
from scipy.stats import (
    ttest_ind,
    mannwhitneyu,
    spearmanr,
    kruskal,
)
from scipy.stats.mstats import winsorize as scipy_winsorize
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_perplexity(sentence_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Aggregate sentence-level perplexity to person-level metrics."""
    agg = (
        sentence_df.groupby("file_id")
        .agg(
            {
                "perplexity_completion": [
                    "mean",
                    "median",
                    "std",
                    lambda x: x.quantile(0.90),
                    lambda x: x.quantile(0.75),
                    lambda x: (x.std() / x.mean()) if x.mean() > 0 else np.nan,
                ],
                "word_count": ["sum", "mean"],
                "utterance": "count",
            }
        )
        .reset_index()
    )
    agg.columns = [
        "file_id",
        f"{model_name}_ppl_mean",
        f"{model_name}_ppl_median",
        f"{model_name}_ppl_std",
        f"{model_name}_ppl_p90",
        f"{model_name}_ppl_p75",
        f"{model_name}_ppl_cv",
        f"{model_name}_total_words",
        f"{model_name}_avg_words",
        f"{model_name}_n_sentences",
    ]
    return agg


def compute_high_ppl_proportion(
    sentence_df: pd.DataFrame, threshold: float, model_name: str
) -> pd.DataFrame:
    """Compute proportion of sentences with perplexity above *threshold* per person."""
    tmp = sentence_df.copy()
    tmp["is_high_ppl"] = (tmp["perplexity_completion"] > threshold).astype(int)
    prop = (
        tmp.groupby("file_id")
        .agg({"is_high_ppl": ["sum", "mean"]})
        .reset_index()
    )
    prop.columns = [
        "file_id",
        f"{model_name}_n_high_ppl",
        f"{model_name}_prop_high_ppl",
    ]
    return prop


# ---------------------------------------------------------------------------
# Group comparison
# ---------------------------------------------------------------------------

def group_comparison(
    df: pd.DataFrame, var: str, group_var: str = "is_patient"
) -> dict:
    """Two-group comparison: t-test, Mann-Whitney U, Cohen's d, rank-biserial r."""
    g0 = df[df[group_var] == 0][var].dropna()
    g1 = df[df[group_var] == 1][var].dropna()
    t_stat, p_t = ttest_ind(g1, g0)
    u_stat, p_mw = mannwhitneyu(g1, g0, alternative="two-sided")
    pooled_std = np.sqrt(
        ((len(g1) - 1) * g1.std() ** 2 + (len(g0) - 1) * g0.std() ** 2)
        / (len(g1) + len(g0) - 2)
    )
    cohens_d = (g1.mean() - g0.mean()) / pooled_std if pooled_std > 0 else np.nan
    r_rb = 1 - (2 * u_stat) / (len(g0) * len(g1))
    return {
        "Variable": var,
        "Control (M±SD)": f"{g0.mean():.2f}±{g0.std():.2f}",
        "Patient (M±SD)": f"{g1.mean():.2f}±{g1.std():.2f}",
        "Control Median": g0.median(),
        "Patient Median": g1.median(),
        "t": t_stat,
        "p(t)": p_t,
        "U": u_stat,
        "p(MW)": p_mw,
        "Cohens d": cohens_d,
        "r": r_rb,
    }


def pairwise_vs_hc(
    df: pd.DataFrame, var: str, diagnosis_col: str = "diagnosis"
) -> list[dict]:
    """Pairwise Mann-Whitney U of each diagnosis vs HC with Cohen's d."""
    hc = df[df[diagnosis_col] == "HC"][var].dropna()
    results = []
    for dx in sorted(df[diagnosis_col].unique()):
        if dx == "HC":
            continue
        grp = df[df[diagnosis_col] == dx][var].dropna()
        u, p = mannwhitneyu(grp, hc, alternative="two-sided")
        pooled = np.sqrt(
            ((len(grp) - 1) * grp.std() ** 2 + (len(hc) - 1) * hc.std() ** 2)
            / (len(grp) + len(hc) - 2)
        )
        d = (grp.mean() - hc.mean()) / pooled if pooled > 0 else np.nan
        results.append({"Diagnosis": dx, "U": u, "p": p, "Cohen's d": d, "n": len(grp)})
    return results


def kruskal_wallis_test(df: pd.DataFrame, var: str, group_col: str = "diagnosis"):
    """Kruskal-Wallis across all groups."""
    groups = [g[var].dropna().values for _, g in df.groupby(group_col)]
    stat, p = kruskal(*groups)
    return stat, p


# ---------------------------------------------------------------------------
# Clinical correlations
# ---------------------------------------------------------------------------

CLINICAL_VARS = ["GAF", "TLI_Impoverished", "TLI_Disorganization", "TLI_Non_specific"]


def clinical_correlations(
    df: pd.DataFrame,
    ppl_var: str,
    clinical_vars: list[str] | None = None,
) -> pd.DataFrame:
    """Spearman correlations of *ppl_var* with each clinical variable."""
    if clinical_vars is None:
        clinical_vars = CLINICAL_VARS
    rows = []
    for cvar in clinical_vars:
        subset = df[[ppl_var, cvar]].dropna()
        if len(subset) < 5:
            rows.append({"Clinical Variable": cvar, "r": np.nan, "p": np.nan, "n": len(subset)})
            continue
        r, p = spearmanr(subset[ppl_var], subset[cvar])
        rows.append({"Clinical Variable": cvar, "r": r, "p": p, "n": len(subset)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Matched sample
# ---------------------------------------------------------------------------

def create_matched_sample(
    df: pd.DataFrame,
    match_vars: list[str] | None = None,
    caliper: float = 0.5,
) -> pd.DataFrame:
    """1:1 optimal matching of patients to controls on *match_vars*."""
    if match_vars is None:
        match_vars = ["age", "sex", "education"]

    patients = df[df["is_patient"] == 1].dropna(subset=match_vars).copy()
    controls = df[df["is_patient"] == 0].dropna(subset=match_vars).copy()

    for d in [patients, controls]:
        d["sex_num"] = (
            d["sex"]
            .map({"M": 0, "F": 1, "m": 0, "f": 1, "male": 0, "female": 1, "Male": 0, "Female": 1})
            .fillna(0.5)
        )

    scaler = StandardScaler()
    match_cols = ["age", "sex_num", "education"]
    X_pat = scaler.fit_transform(patients[match_cols])
    X_ctrl = scaler.transform(controls[match_cols])

    dist_matrix = cdist(X_pat, X_ctrl, metric="euclidean")
    dist_matrix[dist_matrix > caliper * len(match_cols)] = 1e10
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    valid = dist_matrix[row_ind, col_ind] < 1e10

    matched_patients = patients.iloc[row_ind[valid]].copy()
    matched_controls = controls.iloc[col_ind[valid]].copy()
    result = pd.concat([matched_patients, matched_controls])
    print(f"Matched sample: {len(matched_patients)} patients + {len(matched_controls)} controls")
    return result


# ---------------------------------------------------------------------------
# Outlier handling
# ---------------------------------------------------------------------------

def winsorize_and_log(
    df: pd.DataFrame,
    cols: list[str],
    limits: tuple[float, float] = (0.05, 0.05),
) -> pd.DataFrame:
    """Add winsorized and log-transformed columns for each col."""
    df = df.copy()
    for col in cols:
        df[f"{col}_wins"] = scipy_winsorize(
            df[col].fillna(df[col].median()), limits=limits
        )
        df[f"{col}_log"] = np.log1p(df[col])
    return df


# ---------------------------------------------------------------------------
# Confounder / residualization
# ---------------------------------------------------------------------------

def residualize_ppl(
    df: pd.DataFrame,
    ppl_var: str,
    confounders: list[str],
) -> pd.Series:
    """OLS residuals of *ppl_var* after regressing out *confounders*."""
    tmp = df[[ppl_var] + confounders].dropna()
    X = sm.add_constant(tmp[confounders])
    model = sm.OLS(tmp[ppl_var], X).fit()
    resid = pd.Series(model.resid, index=tmp.index, name=f"{ppl_var}_resid")
    return resid


def nested_regression_models(
    df: pd.DataFrame, ppl_var: str
) -> pd.DataFrame:
    """Run four nested OLS models and return a summary DataFrame."""
    tmp = df[[ppl_var, "is_patient", "log_words", "education"]].dropna()
    models = {
        "PPL ~ patient": ["is_patient"],
        "PPL ~ patient + words": ["is_patient", "log_words"],
        "PPL ~ patient + edu": ["is_patient", "education"],
        "PPL ~ patient + words + edu": ["is_patient", "log_words", "education"],
    }
    rows = []
    for label, predictors in models.items():
        X = sm.add_constant(tmp[predictors])
        res = sm.OLS(tmp[ppl_var], X).fit()
        rows.append(
            {
                "Model": label,
                "R2": res.rsquared,
                "R2_adj": res.rsquared_adj,
                "patient_coef": res.params.get("is_patient", np.nan),
                "patient_p": res.pvalues.get("is_patient", np.nan),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Steiger's Z (dependent correlations)
# ---------------------------------------------------------------------------

def steigers_z(r1: float, r2: float, r12: float, n: int) -> tuple[float, float]:
    """Compare two dependent correlations (r1 vs r2, with r12 = correlation between the two predictors).

    Returns (z_stat, p_value).
    """
    from scipy import stats as _stats

    z1 = 0.5 * np.log((1 + r1) / (1 - r1))
    z2 = 0.5 * np.log((1 + r2) / (1 - r2))
    det = 1 - r1**2 - r2**2 - r12**2 + 2 * r1 * r2 * r12
    z_stat = (z1 - z2) * np.sqrt(
        (n - 3) / (2 * (1 - r12) + (r1 + r2) ** 2 * (1 - r12) ** 3 / (4 * det))
    )
    p = 2 * (1 - _stats.norm.cdf(abs(z_stat)))
    return z_stat, p


# ---------------------------------------------------------------------------
# ROC helpers
# ---------------------------------------------------------------------------

def compute_roc(
    df: pd.DataFrame,
    metrics: list[str],
    target: str,
) -> list[dict]:
    """AUC + ROC curve data for each metric predicting binary *target*."""
    from sklearn.metrics import roc_curve, roc_auc_score

    results = []
    for m in metrics:
        sub = df[[m, target]].dropna()
        if sub[target].nunique() < 2 or len(sub) < 10:
            continue
        fpr, tpr, _ = roc_curve(sub[target], sub[m])
        auc_val = roc_auc_score(sub[target], sub[m])
        results.append({"metric": m, "fpr": fpr, "tpr": tpr, "auc": auc_val})
    return results


def logistic_cv_auc(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    cv: int = 5,
) -> float:
    """5-fold cross-validated AUC using logistic regression."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    sub = df[features + [target]].dropna()
    if sub[target].nunique() < 2 or len(sub) < 10:
        return np.nan
    X = sub[features].values
    y = sub[target].values
    model = LogisticRegression(max_iter=1000, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    return scores.mean()
