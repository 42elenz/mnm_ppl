"""Reusable statistical functions for perplexity analysis."""

import numpy as np
import pandas as pd
from scipy.stats import (
    ttest_ind,
    mannwhitneyu,
    spearmanr,
    kruskal,
)
from statsmodels.stats.multitest import multipletests
from scipy.stats.mstats import winsorize as scipy_winsorize
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate_loss_completion(sentence_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Aggregate sentence-level language-model loss to person-level metrics."""
    agg = (
        sentence_df.groupby("file_id")
        .agg(
            avg_loss_mean=("avg_loss_completion", "mean"),
            avg_loss_median=("avg_loss_completion", "median"),
            avg_loss_std=("avg_loss_completion", "std"),
            avg_loss_p90=("avg_loss_completion", lambda x: x.quantile(0.90)),
            avg_loss_p75=("avg_loss_completion", lambda x: x.quantile(0.75)),
            avg_loss_cv=(
                "avg_loss_completion",
                lambda x: (x.std() / x.mean()) if x.mean() > 0 else np.nan,
            ),
            total_words=("word_count_completion", "sum"),
            avg_words=("word_count_completion", "mean"),
            n_sentences=("utterance", "count"),
        )
        .reset_index()
    )
    return agg.rename(
        columns={
            "avg_loss_mean": f"{model_name}_loss_mean",
            "avg_loss_median": f"{model_name}_loss_median",
            "avg_loss_std": f"{model_name}_loss_std",
            "avg_loss_p90": f"{model_name}_loss_p90",
            "avg_loss_p75": f"{model_name}_loss_p75",
            "avg_loss_cv": f"{model_name}_loss_cv",
            "total_words": f"{model_name}_total_words",
            "avg_words": f"{model_name}_avg_words",
            "n_sentences": f"{model_name}_n_sentences",
        }
    )


def add_length_residualized_loss(sentence_df: pd.DataFrame) -> tuple[pd.DataFrame, sm.regression.linear_model.RegressionResultsWrapper | None]:
    """Residualize avg_loss_completion by log word_count_completion at sentence level."""
    df = sentence_df.copy()
    df["avg_loss_completion"] = pd.to_numeric(df["avg_loss_completion"], errors="coerce")
    wc = pd.to_numeric(df["word_count_completion"], errors="coerce")
    df["log_word_count_completion"] = np.log1p(wc.clip(lower=0))

    valid = (
        df["avg_loss_completion"].notna()
        & df["log_word_count_completion"].notna()
        & np.isfinite(df["log_word_count_completion"])
    )
    df["avg_loss_length_resid"] = np.nan

    if valid.sum() < 5:
        return df, None

    X = sm.add_constant(df.loc[valid, "log_word_count_completion"].astype(float))
    y = df.loc[valid, "avg_loss_completion"].astype(float)
    model = sm.OLS(y, X).fit()
    df.loc[valid, "avg_loss_length_resid"] = model.resid
    return df, model


def aggregate_residualized_loss(sentence_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Aggregate sentence-level residualized loss to person-level metrics."""
    agg = (
        sentence_df.groupby("file_id")
        .agg(
            loss_resid_mean=("avg_loss_length_resid", "mean"),
            loss_resid_median=("avg_loss_length_resid", "median"),
            loss_resid_std=("avg_loss_length_resid", "std"),
            loss_resid_p90=("avg_loss_length_resid", lambda x: x.quantile(0.90)),
            total_words=("word_count_completion", "sum"),
            avg_words=("word_count_completion", "mean"),
            n_sentences=("utterance", "count"),
        )
        .reset_index()
    )
    return agg.rename(
        columns={
            "loss_resid_mean": f"{model_name}_loss_resid_mean",
            "loss_resid_median": f"{model_name}_loss_resid_median",
            "loss_resid_std": f"{model_name}_loss_resid_std",
            "loss_resid_p90": f"{model_name}_loss_resid_p90",
            "total_words": f"{model_name}_total_words",
            "avg_words": f"{model_name}_avg_words",
            "n_sentences": f"{model_name}_n_sentences",
        }
    )

def add_length_residualized_perplexity(
    sentence_df: pd.DataFrame,
    ppl_col: str = "perplexity_completion",
    length_col: str = "word_count_completion",
    resid_col: str = "ppl_length_resid",
    log_length_col: str = "log_word_count_completion",
):
    """
    Residualize sentence-level perplexity by log word count.

    Use mainly as sensitivity analysis. For primary analyses,
    residualized loss is usually preferable.
    """
    df = sentence_df.copy()

    df[ppl_col] = pd.to_numeric(df[ppl_col], errors="coerce")
    wc = pd.to_numeric(df[length_col], errors="coerce")
    df[log_length_col] = np.log1p(wc.clip(lower=0))

    valid = (
        df[ppl_col].notna()
        & df[log_length_col].notna()
        & np.isfinite(df[ppl_col])
        & np.isfinite(df[log_length_col])
    )

    df[resid_col] = np.nan

    if valid.sum() < 5:
        return df, None

    X = sm.add_constant(df.loc[valid, log_length_col].astype(float))
    y = df.loc[valid, ppl_col].astype(float)

    model = sm.OLS(y, X).fit()
    df.loc[valid, resid_col] = model.resid

    return df, model

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

def aggregate_residualized_perplexity(sentence_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Aggregate sentence-level length-residualized perplexity to person-level metrics."""
    agg = (
        sentence_df.groupby("file_id")
        .agg(
            ppl_resid_mean=("ppl_length_resid", "mean"),
            ppl_resid_median=("ppl_length_resid", "median"),
            ppl_resid_std=("ppl_length_resid", "std"),
            ppl_resid_p90=("ppl_length_resid", lambda x: x.quantile(0.90)),
            ppl_resid_p75=("ppl_length_resid", lambda x: x.quantile(0.75)),
            total_words=("word_count_completion", "sum"),
            avg_words=("word_count_completion", "mean"),
            n_sentences=("utterance", "count"),
        )
        .reset_index()
    )

    return agg.rename(
        columns={
            "ppl_resid_mean": f"{model_name}_ppl_resid_mean",
            "ppl_resid_median": f"{model_name}_ppl_resid_median",
            "ppl_resid_std": f"{model_name}_ppl_resid_std",
            "ppl_resid_p90": f"{model_name}_ppl_resid_p90",
            "ppl_resid_p75": f"{model_name}_ppl_resid_p75",
            "total_words": f"{model_name}_total_words",
            "avg_words": f"{model_name}_avg_words",
            "n_sentences": f"{model_name}_n_sentences",
        }
    )
def add_winsorized_columns(
    df: pd.DataFrame,
    cols: list[str],
    lower_q: float = 0.01,
    upper_q: float = 0.99,
    suffix: str = "_wins",
) -> pd.DataFrame:
    """
    Add winsorized versions of selected columns.

    Example:
        qwen_loss_resid_median -> qwen_loss_resid_median_wins
    """
    out = df.copy()

    for col in cols:
        if col not in out.columns:
            continue

        x = pd.to_numeric(out[col], errors="coerce")
        lo = x.quantile(lower_q)
        hi = x.quantile(upper_q)
        out[f"{col}{suffix}"] = x.clip(lower=lo, upper=hi)

    return out

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
    r_rb = (2 * u_stat) / (len(g0) * len(g1)) - 1
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


def pairwise_vs_matched_hc(
    df: pd.DataFrame, var: str, diagnosis_col: str = "diagnosis"
) -> list[dict]:
    """Like pairwise_vs_hc but uses only the matched HC counterparts per diagnosis.

    Requires a 'match_id' column from create_matched_sample.
    For each diagnosis, selects only the HCs whose match_id corresponds
    to a patient of that diagnosis.
    """
    if "match_id" not in df.columns:
        return pairwise_vs_hc(df, var, diagnosis_col)

    patients = df[df["is_patient"] == 1]
    controls = df[df["is_patient"] == 0]
    results = []
    for dx in sorted(patients[diagnosis_col].unique()):
        dx_patients = patients[patients[diagnosis_col] == dx]
        dx_match_ids = dx_patients["match_id"].values
        dx_controls = controls[controls["match_id"].isin(dx_match_ids)]

        grp = dx_patients[var].dropna()
        hc = dx_controls[var].dropna()
        if len(grp) < 2 or len(hc) < 2:
            continue
        u, p = mannwhitneyu(grp, hc, alternative="two-sided")
        pooled = np.sqrt(
            ((len(grp) - 1) * grp.std() ** 2 + (len(hc) - 1) * hc.std() ** 2)
            / (len(grp) + len(hc) - 2)
        )
        d = (grp.mean() - hc.mean()) / pooled if pooled > 0 else np.nan
        results.append({
            "Diagnosis": dx, "U": u, "p": p, "Cohen's d": d,
            "n": len(grp), "n(HC)": len(hc),
        })
    return results


def kruskal_wallis_test(df: pd.DataFrame, var: str, group_col: str = "diagnosis"):
    """Kruskal-Wallis across all groups."""
    groups = [g[var].dropna().values for _, g in df.groupby(group_col)]
    stat, p = kruskal(*groups)
    return stat, p


# ---------------------------------------------------------------------------
# Clinical correlations
# ---------------------------------------------------------------------------

CLINICAL_VARS = ["GAF", "TLI_Impoverished", "TLI_Disorganization",
         'GAFscore', 'verbal_IQ',
       'TLI_Disorganization', 'TLI_Non_specific',
       'SANS_AlogieParalogie_ENIGMA',
       'SAPS_PositiveFormaleDenkstoerung_ENIGMA',
     'SAPS_Sum_ENIGMA', 'YMRS_Sum',
       'HAMD_Sum17', 'SANS_Anhedonie_ENIGMA',
       'SANS_Affektverflachung_ENIGMA', 'SANS_AbulieApathie_ENIGMA',
       'SAPS_Halluzinationen_ENIGMA', 'SAPS_Wahn_ENIGMA',
       'SAPS_BizarresVerhalten_ENIGMA',
        'SemanticVF', 'LexicalVF']


def clinical_correlations(
    df: pd.DataFrame,
    ppl_var: str,
    clinical_vars: list[str] | None = None,
    correction_type="fdr_bh" #fdr_tsbky
) -> pd.DataFrame:
    """Spearman correlations of *ppl_var* with each clinical variable.

    Adds FDR-corrected p-values (Benjamini-Hochberg) in column 'p_fdr'
    and a boolean 'sig_fdr' (True if p_fdr < 0.05).
    """
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

    result = pd.DataFrame(rows)

    # FDR correction (two-stage Benjamini-Krieger-Yekutieli) on non-NaN p-values
    valid_mask = result["p"].notna()
    if valid_mask.sum() > 0:
        result["p_fdr"] = np.nan
        result["sig_fdr"] = False
        _, p_corrected, _, _ = multipletests(
            result.loc[valid_mask, "p"], alpha=0.05, method=correction_type
        )
        result.loc[valid_mask, "p_fdr"] = p_corrected
        result.loc[valid_mask, "sig_fdr"] = p_corrected < 0.05

    return result


# ---------------------------------------------------------------------------
# Matched sample
# ---------------------------------------------------------------------------

def create_matched_sample(
    df: pd.DataFrame,
    match_vars: list[str] | None = None,
    caliper: float = 0.5,
) -> pd.DataFrame:
    """1:1 optimal matching of patients to HC on *match_vars*.

    Uses standardised Euclidean distance + Hungarian algorithm.
    """
    if match_vars is None:
        match_vars = ["age", "sex", "education"]

    # Build numeric matching columns (encode sex as 0/1)
    work = df.copy()
    if "sex" in match_vars:
        work["sex_num"] = (
            work["sex"]
            .map({"M": 0, "F": 1, "m": 0, "f": 1,
                   "male": 0, "female": 1, "Male": 0, "Female": 1})
            .fillna(0.5)
        )
    match_cols = ["sex_num" if v == "sex" else v for v in match_vars]

    required = [c for c in match_vars if c != "sex"] + (["sex"] if "sex" in match_vars else [])
    patients = work[work["is_patient"] == 1].dropna(subset=required)
    controls = work[work["is_patient"] == 0].dropna(subset=required)

    scaler = StandardScaler()
    X_pat = scaler.fit_transform(patients[match_cols])
    X_ctrl = scaler.transform(controls[match_cols])

    dist_matrix = cdist(X_pat, X_ctrl, metric="euclidean")
    dist_matrix[dist_matrix > caliper * len(match_cols)] = 1e10
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    valid = dist_matrix[row_ind, col_ind] < 1e10

    matched_patients = patients.iloc[row_ind[valid]].copy()
    matched_controls = controls.iloc[col_ind[valid]].copy()

    # Assign match pair IDs so we can trace which HC belongs to which patient
    pair_ids = np.arange(valid.sum())
    matched_patients["match_id"] = pair_ids
    matched_controls["match_id"] = pair_ids

    matched = pd.concat([matched_patients, matched_controls])
    print(f"  Matched sample: {valid.sum()} patients + {valid.sum()} HC"
          f"  (match_vars={match_vars})")
    return matched


def create_matched_sample_per_diagnosis(
    df: pd.DataFrame,
    match_vars: list[str] | None = None,
    k: int = 2,
    caliper: float = 0.5,
) -> pd.DataFrame:
    """1:k nearest-neighbor matching of each diagnosis group vs all HCs.

    For each diagnosis (BD, MDD, SSD, …), independently find the *k* closest
    HC subjects on *match_vars* (without replacement within a diagnosis).
    The same HC may appear in multiple diagnosis comparisons — that is fine
    because the per-diagnosis tests are independent.

    The returned DataFrame contains all patients and their matched HCs,
    with columns ``match_id`` (pair link) and ``match_diagnosis`` (which
    diagnosis group the HC was matched against).
    """
    if match_vars is None:
        match_vars = ["age", "sex", "education"]

    work = df.copy()
    if "sex" in match_vars:
        work["sex_num"] = (
            work["sex"]
            .map({"M": 0, "F": 1, "m": 0, "f": 1,
                   "male": 0, "female": 1, "Male": 0, "Female": 1})
            .fillna(0.5)
        )
    match_cols = ["sex_num" if v == "sex" else v for v in match_vars]

    required = [c for c in match_vars if c != "sex"] + (["sex"] if "sex" in match_vars else [])
    all_controls = work[work["is_patient"] == 0].dropna(subset=required)
    all_patients = work[work["is_patient"] == 1].dropna(subset=required)

    # Fit scaler on all data so distances are comparable across diagnoses
    scaler = StandardScaler()
    scaler.fit(work.dropna(subset=required)[match_cols])

    X_ctrl = scaler.transform(all_controls[match_cols])

    frames = []
    pair_counter = 0
    for dx in sorted(all_patients["diagnosis"].unique()):
        dx_patients = all_patients[all_patients["diagnosis"] == dx]
        X_pat = scaler.transform(dx_patients[match_cols])

        dist_matrix = cdist(X_pat, X_ctrl, metric="euclidean")
        caliper_val = caliper * len(match_cols)

        dx_matched_pat_idx = []
        dx_matched_ctrl_idx = []
        dx_pair_ids_pat = []
        dx_pair_ids_ctrl = []

        used_ctrl_dx: set[int] = set()
        for i in range(len(dx_patients)):
            dists = dist_matrix[i].copy()
            for used_j in used_ctrl_dx:
                dists[used_j] = 1e10
            chosen = []
            for _ in range(k):
                best = np.argmin(dists)
                if dists[best] > caliper_val or dists[best] >= 1e10:
                    break
                chosen.append(best)
                dists[best] = 1e10  # mark used (within this patient)
            if not chosen:
                continue
            used_ctrl_dx.update(chosen)  
            dx_matched_pat_idx.append(i)
            for c_idx in chosen:
                dx_matched_ctrl_idx.append(c_idx)
                dx_pair_ids_ctrl.append(pair_counter)
            dx_pair_ids_pat.append(pair_counter)
            pair_counter += 1

        if not dx_matched_pat_idx:
            continue

        mp = dx_patients.iloc[dx_matched_pat_idx].copy()
        mc = all_controls.iloc[dx_matched_ctrl_idx].copy()
        mp["match_id"] = dx_pair_ids_pat
        mc["match_id"] = dx_pair_ids_ctrl
        mp["match_diagnosis"] = dx
        mc["match_diagnosis"] = dx

        n_hc = mc["file_id"].nunique() if "file_id" in mc.columns else len(mc)
        print(f"  {dx}: {len(mp)} patients × up to {k} HCs → "
              f"{len(mc)} matched HC rows ({n_hc} unique)")
        frames.extend([mp, mc])

    matched = pd.concat(frames)
    total_pat = matched[matched["is_patient"] == 1]
    total_hc = matched[matched["is_patient"] == 0]
    print(f"  Per-diagnosis matched total: {len(total_pat)} patients + "
          f"{len(total_hc)} HC rows  (match_vars={match_vars}, k={k})")
    return matched


def matching_balance_table(
    df: pd.DataFrame,
    vars: list[str],
    group_col: str = "is_patient",
) -> pd.DataFrame:
    """Compare confounder distributions between groups.

    For continuous variables: mean ± SD per group, t-test p, standardised mean
    difference (SMD).  For binary/categorical 'sex': proportion female per group,
    chi² p, SMD.
    """
    g0 = df[df[group_col] == 0]
    g1 = df[df[group_col] == 1]
    rows = []
    for v in vars:
        if v == "sex":
            # proportion female
            p0 = (g0["sex"].str.lower() == "female").mean()
            p1 = (g1["sex"].str.lower() == "female").mean()
            from scipy.stats import chi2_contingency
            ct = pd.crosstab(df["sex"].str.lower(), df[group_col])
            _, pval, _, _ = chi2_contingency(ct)
            pooled = (p0 + p1) / 2
            smd = abs(p1 - p0) / np.sqrt(pooled * (1 - pooled)) if pooled not in (0, 1) else 0
            rows.append({
                "Variable": v,
                "HC": f"{p0:.1%} female",
                "Patient": f"{p1:.1%} female",
                "p-value": pval,
                "SMD": smd,
            })
        else:
            s0 = g0[v].dropna()
            s1 = g1[v].dropna()
            pooled_sd = np.sqrt((s0.var() + s1.var()) / 2)
            smd = abs(s1.mean() - s0.mean()) / pooled_sd if pooled_sd > 0 else 0
            _, pval = ttest_ind(s0, s1, equal_var=False)
            rows.append({
                "Variable": v,
                "HC": f"{s0.mean():.2f} ± {s0.std():.2f}",
                "Patient": f"{s1.mean():.2f} ± {s1.std():.2f}",
                "p-value": pval,
                "SMD": smd,
            })
    return pd.DataFrame(rows)


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
        # Guard against values <= -1, which produce invalid log1p warnings.
        safe = pd.to_numeric(df[col], errors="coerce").clip(lower=-0.999999)
        df[f"{col}_log"] = np.log1p(safe)
    return df


# ---------------------------------------------------------------------------
# Confounder / residualization
# ---------------------------------------------------------------------------

def residualize_ppl(
    df: pd.DataFrame,
    ppl_var: str,
    confounders: list[str],
    group_col: str = "is_patient",
    fit_group: int = 0,  # 0 = HC
    fit_on_all: bool = False,
) -> pd.Series:
    """OLS residuals of *ppl_var*, confounder model fit on reference group only."""
    all_tmp = df[[ppl_var, group_col] + confounders].dropna()
    if all_tmp.empty:
        return pd.Series(dtype=float, name=f"{ppl_var}_resid")

    if fit_on_all:
        ref = all_tmp
    else:
    # fit on HC only
        ref = all_tmp[all_tmp[group_col] == fit_group]
    if ref.empty:
        return pd.Series(dtype=float, name=f"{ppl_var}_resid")

    X_ref_base = pd.get_dummies(ref[confounders], drop_first=True)
    X_all_base = pd.get_dummies(all_tmp[confounders], drop_first=True)

    # Predict design must have exactly the columns used during fitting.
    X_all_base = X_all_base.reindex(columns=X_ref_base.columns, fill_value=0)

    X_ref = sm.add_constant(X_ref_base, has_constant="add").astype(float)
    X_all = sm.add_constant(X_all_base, has_constant="add").astype(float)
    X_ref = X_ref.apply(pd.to_numeric, errors="coerce")
    X_all = X_all.apply(pd.to_numeric, errors="coerce")
    y_ref = pd.to_numeric(ref[ppl_var], errors="coerce")

    fit_ok = (~X_ref.isna().any(axis=1)) & y_ref.notna()
    if fit_ok.sum() < 3:
        return pd.Series(dtype=float, name=f"{ppl_var}_resid")

    X_ref = X_ref.loc[fit_ok]
    y_ref = y_ref.loc[fit_ok]
    model = sm.OLS(y_ref, X_ref).fit()

    # predict & residualize for everyone
    pred_ok = ~X_all.isna().any(axis=1)
    if pred_ok.sum() == 0:
        return pd.Series(dtype=float, name=f"{ppl_var}_resid")
    X_all_pred = X_all.loc[pred_ok]
    y_all = pd.to_numeric(all_tmp[ppl_var], errors="coerce").loc[pred_ok]
    predicted = model.predict(X_all_pred)
    resid = pd.Series(
        y_all.values - predicted.values,
        index=y_all.index,
        name=f"{ppl_var}_resid",
    )
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


def evaluate_incremental_value(
    df: pd.DataFrame,
    target: str,
    ppl_metric: str,
    length_metric: str,
    confounder_cols: list[str] | None = None,
    proximity_col: str | None = None,
    cv_type: str = "loo",
) -> dict:
    """
    Comprehensive multi-model evaluation:
    1. LLM Loss only
    2. Length only
    3. Length + LLM Loss
    4. Full Confounders (Age, Sex, Edu, Length)
    5. Full Confounders + LLM Loss
    6. (Optional) Full Confounders + Proximity
    7. (Optional) Full Confounders + Proximity + LLM Loss
    """
    from scipy.stats import chi2
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut, cross_val_predict, KFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    # 1. Prepare Data
    cols_to_use = [target, ppl_metric, length_metric]
    if confounder_cols:
        # Resolve 'sex' to 'sex_num' if necessary
        actual_confounders = ["sex_num" if c == "sex" else c for c in confounder_cols]
        cols_to_use.extend([c for c in actual_confounders if c not in cols_to_use])
    else:
        actual_confounders = []
        
    if proximity_col:
        cols_to_use.append(proximity_col)

    data = df[cols_to_use].dropna().copy()
    data["log_length"] = np.log1p(data[length_metric])
    y = data[target].astype(int)

    def fit_model(features):
        X = sm.add_constant(data[features], has_constant="add")
        return sm.Logit(y, X).fit(disp=False)

    def get_cv_probs(features):
        if cv_type == "loo":
            cv = LeaveOneOut()
        else:
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
        # Note: Do not scale binary target/features, but scaling other numeric features
        # can help with logistic regression convergence.
        # To maintain identical results for unscaled models (no scaler), we do not scale here,
        # but keep it identical to the main script.
        return cross_val_predict(
            LogisticRegression(max_iter=1000), data[features], y, cv=cv, method="predict_proba"
        )[:, 1]

    # Model Definitions
    m_ppl = [ppl_metric]
    m_len = ["log_length"]
    m_len_ppl = ["log_length", ppl_metric]

    # Ensure length is not double-counted (e.g. as both 'qwen_avg_words' and 'log_length')
    clean_confounders = [c for c in actual_confounders if c != length_metric and c != "log_length"]
    m_conf = list(dict.fromkeys(clean_confounders + ["log_length"]))
    m_conf_ppl = m_conf + [ppl_metric]

    # Fit models for LRT/AIC
    res_ppl = fit_model(m_ppl)
    res_len = fit_model(m_len)
    res_len_ppl = fit_model(m_len_ppl)
    res_conf = fit_model(m_conf)
    res_conf_ppl = fit_model(m_conf_ppl)

    # CV probabilities for AUC/ROC
    prob_ppl = get_cv_probs(m_ppl)
    prob_len = get_cv_probs(m_len)
    prob_len_ppl = get_cv_probs(m_len_ppl)
    prob_conf = get_cv_probs(m_conf)
    prob_conf_ppl = get_cv_probs(m_conf_ppl)

    # Likelihood Ratio Tests
    lrt_len_p = chi2.sf(-2 * (res_len.llf - res_len_ppl.llf), df=1)
    lrt_conf_p = chi2.sf(-2 * (res_conf.llf - res_conf_ppl.llf), df=1)

    result = {
        "y_true": y.values,
        "models": {
            "Median LLM Loss Only": {"prob": prob_ppl, "aic": res_ppl.aic, "auc": roc_auc_score(y, prob_ppl)},
            "Length Only": {"prob": prob_len, "aic": res_len.aic, "auc": roc_auc_score(y, prob_len)},
            "Length + Median LLM Loss": {"prob": prob_len_ppl, "aic": res_len_ppl.aic, "auc": roc_auc_score(y, prob_len_ppl)},
            "Confounders Only": {"prob": prob_conf, "aic": res_conf.aic, "auc": roc_auc_score(y, prob_conf)},
            "Confounders + Median LLM Loss": {"prob": prob_conf_ppl, "aic": res_conf_ppl.aic, "auc": roc_auc_score(y, prob_conf_ppl)},
        },
        "lrt_p_vs_length": lrt_len_p,
        "lrt_p_vs_confounders": lrt_conf_p,
    }
    
    if proximity_col:
        m_prox = m_conf + [proximity_col]
        m_prox_ppl = m_prox + [ppl_metric]
        
        res_prox = fit_model(m_prox)
        res_prox_ppl = fit_model(m_prox_ppl)
        
        prob_prox = get_cv_probs(m_prox)
        prob_prox_ppl = get_cv_probs(m_prox_ppl)
        
        lrt_prox_p = chi2.sf(-2 * (res_conf.llf - res_prox.llf), df=1)
        lrt_prox_ppl_p = chi2.sf(-2 * (res_prox.llf - res_prox_ppl.llf), df=1)
        
        result["models"]["Confounders + Proximity"] = {"prob": prob_prox, "aic": res_prox.aic, "auc": roc_auc_score(y, prob_prox)}
        result["models"]["Confounders + Proximity + Median LLM Loss"] = {"prob": prob_prox_ppl, "aic": res_prox_ppl.aic, "auc": roc_auc_score(y, prob_prox_ppl)}
        result["lrt_p_vs_proximity"] = lrt_prox_ppl_p

    return result


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
