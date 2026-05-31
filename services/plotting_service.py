"""Service that produces all standard plots for a given analysis setting."""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from functions.plotting.plots import (
    plot_group_boxplot,
    plot_diagnostic_groups,
    plot_clinical_scatter,
    plot_extreme_groups,
    plot_tli_subscales,
    plot_roc_curves,
    plot_confounder_analysis,
    plot_raw_vs_residualized,
    plot_raw_perplexity_basic,
    plot_metrics_heatmap,
    plot_correlation_robustness,
    plot_matching_balance
)
from functions.analysis.statistics import (
    CLINICAL_VARS,
    group_comparison,
    clinical_correlations,
    compute_roc,
)


def plot_setting(
    df,
    ppl_var: str,
    setting_name: str,
    figure_dir: str = "figures",
    clinical_vars: list[str] | None = None,
):
    """Generate the standard set of plots for one analysis setting.

    Args:
        df: person-level DataFrame (with is_patient, diagnosis, clinical cols)
        ppl_var: perplexity column to use (e.g. 'qwen_ppl_median')
        setting_name: label like 'with_outliers', used in filenames
        figure_dir: base output directory for figures
        clinical_vars: list of clinical variable names

    Returns:
        dict with comparison results and correlation DataFrame
    """
    if clinical_vars is None:
        clinical_vars = CLINICAL_VARS
    out = os.path.join(figure_dir, setting_name)

    patients = df[df["is_patient"] == 1].copy()

    # 1. Group boxplot
    plot_group_boxplot(df, ppl_var, title=f"Healthy Controls vs Patient",
                       save_path=os.path.join(out, "01_group_boxplot.png"))

    # 2. Group comparison table
    comp = group_comparison(df, ppl_var)
    print(f"\n{setting_name} — {ppl_var}:")
    for k, v in comp.items():
        print(f"  {k}: {v}")

    # 3. Clinical correlations
    corr_df = clinical_correlations(patients, ppl_var, clinical_vars)
    print(f"\nClinical correlations ({setting_name}):")
    print(corr_df.to_string(index=False))

    # 4. Clinical scatter
    plot_clinical_scatter(patients, ppl_var, clinical_vars,
                          title=f"{setting_name}: Clinical correlations",
                          save_path=os.path.join(out, "02_clinical_scatter.png"))

    # 5. Diagnostic groups
    plot_diagnostic_groups(df, ppl_var, title=f"{setting_name}: by diagnosis",
                           save_path=os.path.join(out, "03_diagnostic_groups.png"))

    # 6. Extreme groups
    plot_extreme_groups(patients, ppl_var, clinical_vars,
                        save_path=os.path.join(out, "04_extreme_groups.png"))

    # 7. TLI subscales
    plot_tli_subscales(patients, ppl_var,
                       save_path=os.path.join(out, "05_tli_subscales.png"))

    return {"comparison": comp, "correlations": corr_df}


def plot_cross_setting(
    corr_dfs: dict,
    figure_dir: str = "figures",
):
    """Produce robustness plots comparing correlations across settings."""
    plot_correlation_robustness(
        corr_dfs,
        save_path_scatter=os.path.join(figure_dir, "cross_setting", "robustness_scatter.png"),
        save_path_bar=os.path.join(figure_dir, "cross_setting", "robustness_bar.png"),
    )


def plot_advanced(
    df,
    patients,
    ppl_var: str,
    wc_var: str,
    figure_dir: str = "figures",
):
    """Additional plots: confounder analysis, raw vs residualized, basic overview."""
    plot_confounder_analysis(
        df, ppl_var, wc_var,
        save_path=os.path.join(figure_dir, "advanced", "confounder_analysis.png"),
    )
    plot_raw_vs_residualized(
        df, ppl_var, wc_var,
        save_path=os.path.join(figure_dir, "advanced", "raw_vs_residualized.png"),
    )
    plot_raw_perplexity_basic(
        df, ppl_var,
        save_path=os.path.join(figure_dir, "advanced", "raw_perplexity_basic.png"),
    )

def plot_matched_sample_balance(
    df_full,
    df_matched,
    match_vars: list[str],
    figure_dir: str = "figures",
    run_tests_only: bool = False,
    run_matching_tests: bool = True,
    distance_cutoffs: tuple[float, ...] = (0.3, 0.5, 0.75),
    user_distance_cutoff: float | None = None,
):
    """Plot matching balance and run optional matching diagnostics.

    Diagnostics include:
    - duplicate ID checks within each diagnosis,
    - match_id integrity checks,
    - per-diagnosis patient-vs-HC distance summary on match variables.
    - per-diagnosis distance exceedance rates for cutoff thresholds.

    Returns a dict with computed test tables. This keeps the function usable
    in notebooks where users want tests only (without plotting).
    """
    results = {}

    if not run_tests_only:
        results["balance"] = plot_matching_balance(
            df_full,
            df_matched,
            match_vars,
            figure_dir=os.path.join(figure_dir, "matched"),
        )

    if not run_matching_tests:
        return results

    m = df_matched.copy()

    # Duplicate checks within diagnosis (if available).
    if "match_diagnosis" in m.columns and "file_id" in m.columns:
        dup_hc_dx = (
            m[m["is_patient"] == 0]
            .groupby(["match_diagnosis", "file_id"])
            .size()
            .reset_index(name="n")
            .query("n > 1")
            .sort_values(["match_diagnosis", "n"], ascending=[True, False])
        )
        dup_pat_dx = (
            m[m["is_patient"] == 1]
            .groupby(["match_diagnosis", "file_id"])
            .size()
            .reset_index(name="n")
            .query("n > 1")
            .sort_values(["match_diagnosis", "n"], ascending=[True, False])
        )
        results["dup_hc_within_diagnosis"] = dup_hc_dx
        results["dup_patient_within_diagnosis"] = dup_pat_dx
        print("\nDuplicate checks (within diagnosis):")
        print(f"  HC duplicates found: {len(dup_hc_dx)}")
        print(f"  Patient duplicates found: {len(dup_pat_dx)}")

    # Integrity checks per match_id.
    if "match_id" in m.columns:
        integrity = (
            m.groupby("match_id")
            .apply(
                lambda g: pd.Series(
                    {
                        "n_pat": int((g["is_patient"] == 1).sum()),
                        "n_hc": int((g["is_patient"] == 0).sum()),
                        "n_dx": int(g["match_diagnosis"].nunique())
                        if "match_diagnosis" in g.columns
                        else np.nan,
                    }
                )
            )
            .reset_index()
        )
        max_hc_per_patient = int(
            np.nanmax(integrity["n_hc"].values) if len(integrity) else 0
        )
        bad = integrity[(integrity["n_pat"] != 1) | (integrity["n_hc"] < 1)]
        if "n_dx" in integrity.columns:
            bad = pd.concat([bad, integrity[integrity["n_dx"] > 1]], ignore_index=True)
            bad = bad.drop_duplicates()
        results["match_integrity"] = integrity
        results["match_integrity_issues"] = bad
        print("\nMatch integrity:")
        print(f"  Max HC per patient-pair observed: {max_hc_per_patient}")
        print(f"  Problematic match_id rows: {len(bad)}")

    # Per-diagnosis distances on matching variables.
    if (
        "match_id" in m.columns
        and "match_diagnosis" in m.columns
        and len(match_vars) > 0
    ):
        work = df_full.copy()

        # Recreate numeric sex coding if needed to mirror matching behavior.
        if "sex" in match_vars:
            if "sex_num" not in work.columns and "sex" in work.columns:
                work["sex_num"] = (
                    work["sex"]
                    .map(
                        {
                            "M": 0,
                            "F": 1,
                            "m": 0,
                            "f": 1,
                            "male": 0,
                            "female": 1,
                            "Male": 0,
                            "Female": 1,
                        }
                    )
                    .fillna(0.5)
                )
            if "sex_num" not in m.columns and "sex" in m.columns:
                m["sex_num"] = (
                    m["sex"]
                    .map(
                        {
                            "M": 0,
                            "F": 1,
                            "m": 0,
                            "f": 1,
                            "male": 0,
                            "female": 1,
                            "Male": 0,
                            "Female": 1,
                        }
                    )
                    .fillna(0.5)
                )

        match_cols = ["sex_num" if v == "sex" else v for v in match_vars]
        required = [
            c for c in match_vars if c != "sex"
        ] + (["sex"] if "sex" in match_vars else [])

        base = work.dropna(subset=required).copy()
        dist_rows = []

        if len(base) > 0 and all(col in base.columns for col in match_cols):
            scaler = StandardScaler().fit(base[match_cols])

            for dx, gdx in m.groupby("match_diagnosis"):
                for mid, g in gdx.groupby("match_id"):
                    pat = g[g["is_patient"] == 1]
                    hcs = g[g["is_patient"] == 0]
                    if len(pat) != 1 or len(hcs) == 0:
                        continue
                    if not all(col in g.columns for col in match_cols):
                        continue

                    pmat = pat[match_cols].dropna()
                    hmat = hcs[match_cols].dropna()
                    if len(pmat) != 1 or len(hmat) == 0:
                        continue

                    pvec = scaler.transform(pmat)[0]
                    hcvecs = scaler.transform(hmat)
                    for i in range(len(hmat)):
                        dist_rows.append(
                            {
                                "diagnosis": dx,
                                "match_id": mid,
                                "distance_z_euclid": float(np.linalg.norm(pvec - hcvecs[i])),
                            }
                        )

        dist_df = pd.DataFrame(dist_rows)
        if len(dist_df):
            dist_summary = (
                dist_df.groupby("diagnosis")["distance_z_euclid"]
                .agg(["count", "mean", "median", "std", "max"])
                .reset_index()
                .sort_values("median")
            )
        else:
            dist_summary = pd.DataFrame(
                columns=["diagnosis", "count", "mean", "median", "std", "max"]
            )

        results["distance_pairs"] = dist_df
        results["distance_summary_by_diagnosis"] = dist_summary

        # Cutoff diagnostics: share of matches above threshold per diagnosis.
        cutoff_values = list(distance_cutoffs)
        if user_distance_cutoff is not None:
            cutoff_values.append(float(user_distance_cutoff))
        cutoff_values = sorted(set(cutoff_values))

        if len(dist_df) and len(cutoff_values):
            cutoff_rows = []
            for dx, gdx in dist_df.groupby("diagnosis"):
                n = len(gdx)
                row = {"diagnosis": dx, "n_pairs": n}
                for c in cutoff_values:
                    n_above = int((gdx["distance_z_euclid"] > c).sum())
                    row[f"n_gt_{c:g}"] = n_above
                    row[f"pct_gt_{c:g}"] = n_above / n if n > 0 else np.nan
                cutoff_rows.append(row)
            cutoff_df = pd.DataFrame(cutoff_rows).sort_values("diagnosis")

            if user_distance_cutoff is not None:
                flagged_pairs = dist_df[
                    dist_df["distance_z_euclid"] > float(user_distance_cutoff)
                ].copy()
            else:
                flagged_pairs = pd.DataFrame(columns=dist_df.columns)
        else:
            cutoff_df = pd.DataFrame()
            flagged_pairs = pd.DataFrame(columns=dist_df.columns if len(dist_df) else ["diagnosis", "match_id", "distance_z_euclid"])

        results["distance_cutoff_summary_by_diagnosis"] = cutoff_df
        results["distance_pairs_above_user_cutoff"] = flagged_pairs

        print("\nPer-diagnosis distance summary available in return dict:")
        print("  results['distance_summary_by_diagnosis']")
        print("  results['distance_cutoff_summary_by_diagnosis']")
        if user_distance_cutoff is not None:
            print(f"  results['distance_pairs_above_user_cutoff'] (cutoff={user_distance_cutoff})")

    return results
