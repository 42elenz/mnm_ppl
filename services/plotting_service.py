"""Service that produces all standard plots for a given analysis setting."""

import os
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
    plot_group_boxplot(df, ppl_var, title=f"{setting_name}: HC vs Patient",
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
