"""Reusable plotting functions for perplexity analysis."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import gaussian_kde, kruskal, mannwhitneyu, spearmanr, chi2_contingency
from matplotlib.patches import Patch
import statsmodels.api as sm

def fmt_p(p):
    if p < 0.001:
        return "< 0.001"
    elif p < 0.01:
        return f"{p:.3f}"
    elif round(p, 2) == 0.05:
        return f"{p:.3f}"  # Grenzfälle immer 3 Stellen
    else:
        return f"{p:.2f}"

# Consistent styling
DIAG_ORDER = ["HC", "MDD", "BD", "SSD"]
DIAG_COLORS = {"HC": "#3498DB", "SSD": "#E74C3C", "MDD": "#F39C12", "BD": "#9B59B6"}
GROUP_COLORS = {"control": "#3498db", "patient": "#e74c3c", "high_ppl": "#c0392b", "normal_ppl": "#f39c12"}


def _savefig(fig, path):
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# 1. Group comparison boxplot (HC vs Patient)
# ---------------------------------------------------------------------------

def plot_group_boxplot(
    df: pd.DataFrame,
    var: str,
    title: str = "",
    save_path: str | None = None,
    ylabel: str | None = "Mean Loss",
    plot_style: str = "box",
):
    """Plot *var* split by is_patient using box or raincloud style.

    Parameters
    ----------
    plot_style : str
        "box" (default) for boxplot + stripplot, or "raincloud" for
        half-violin + jittered points + narrow boxplot.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    plot_df = df.copy()
    plot_df["Group"] = plot_df["is_patient"].map({0: "HC", 1: "Patient"})

    if plot_style == "box":
        sns.boxplot(
            data=plot_df, x="Group", y=var, hue="Group",
            order=["HC", "Patient"],
            palette={"HC": GROUP_COLORS["control"], "Patient": GROUP_COLORS["patient"]},
            width=0.5, ax=ax, legend=False,
        )
        sns.stripplot(
            data=plot_df, x="Group", y=var, hue="Group",
            order=["HC", "Patient"],
            palette={"HC": "black", "Patient": "black"},
            alpha=0.4, size=3, jitter=True, ax=ax, legend=False,
        )
    elif plot_style == "raincloud":
        order = ["HC", "Patient"]
        colors = {"HC": GROUP_COLORS["control"], "Patient": GROUP_COLORS["patient"]}

        # Half-violin (cloud) on the right side of each group position
        for i, grp in enumerate(order):
            vals = plot_df.loc[plot_df["Group"] == grp, var].dropna().values
            if len(vals) < 3:
                continue
            kde = gaussian_kde(vals)
            ys = np.linspace(vals.min(), vals.max(), 200)
            density = kde(ys)
            density = density / density.max() * 0.35
            ax.fill_betweenx(ys, i, i + density, color=colors[grp], alpha=0.4, linewidth=0)
            ax.plot(i + density, ys, color=colors[grp], linewidth=1)

        # Rain (jittered points) on the left side
        for i, grp in enumerate(order):
            vals = plot_df.loc[plot_df["Group"] == grp, var].dropna().values
            if len(vals) == 0:
                continue
            jitter = np.random.uniform(-0.18, -0.03, size=len(vals))
            ax.scatter(i + jitter, vals, color=colors[grp], alpha=0.45, s=12, edgecolors="none", zorder=2)

        # Narrow boxplot in the center
        bp_data = [plot_df.loc[plot_df["Group"] == grp, var].dropna().values for grp in order]
        bp = ax.boxplot(
            bp_data,
            positions=range(len(order)),
            widths=0.12,
            patch_artist=True,
            showfliers=False,
            zorder=3,
        )
        for patch, grp in zip(bp["boxes"], order):
            patch.set_facecolor(colors[grp])
            patch.set_alpha(0.85)
        for element in ["whiskers", "caps", "medians"]:
            for line in bp[element]:
                line.set_color("black")
                line.set_linewidth(1)

        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order)
    else:
        raise ValueError("plot_style must be 'box' or 'raincloud'")

    ax.set_ylabel(ylabel if ylabel is not None else var)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _savefig(fig, save_path)
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# 2. Diagnostic group boxplot (per diagnosis)
# ---------------------------------------------------------------------------

def plot_diagnostic_groups(
    df: pd.DataFrame,
    var: str,
    title: str = "Perplexity by diagnosis",
    save_path: str | None = None,
):
    """Two-panel plot: boxplot by diagnosis + bar chart of % high-PPL."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    diags_present = [dx for dx in DIAG_ORDER if dx in df["diagnosis"].values]
    palette = {dx: DIAG_COLORS.get(dx, "#888888") for dx in diags_present}

    # Panel A: boxplot with hue
    sns.boxplot(
        data=df, x="diagnosis", y=var, hue="diagnosis",
        order=diags_present, palette=palette,
        width=0.5, ax=ax1, legend=False,
    )
    sns.stripplot(
        data=df, x="diagnosis", y=var, hue="diagnosis",
        order=diags_present, palette={dx: "black" for dx in diags_present},
        alpha=0.4, size=3, jitter=True, ax=ax1, legend=False,
    )
    data_groups = [df[df["diagnosis"] == dx][var].dropna().values for dx in diags_present if len(df[df["diagnosis"] == dx][var].dropna()) > 0]
    stat, p_kw = kruskal(*data_groups)
    ax1.set_title(f"{title} (KW p={p_kw:.4f})")
    ax1.set_ylabel(var)
    ax1.grid(axis="y", alpha=0.3)

    # Panel B: % with high PPL (if column exists)
    if var.endswith("_ppl_median"):
        high_col = var.replace("_ppl_median", "_high")
    elif var.endswith("_div_median"):
        high_col = var.replace("_div_median", "_high")
    else:
        high_col = f"{var}_high"
    if high_col not in df.columns:
        hc_vals = df[df["diagnosis"] == "HC"][var].dropna()
        if len(hc_vals) > 0:
            thresh = hc_vals.quantile(0.90)
            df = df.copy()
            df[high_col] = (df[var] > thresh).astype(int)

    if high_col in df.columns:
        pct = df.groupby("diagnosis")[high_col].mean().reindex(diags_present) * 100
        ax2.bar(diags_present, pct, color=[palette.get(d, "#888") for d in diags_present])
        ax2.set_ylabel("% with high PPL")
        ax2.set_title("Prevalence of high perplexity")
        ax2.grid(axis="y", alpha=0.3)
    else:
        ax2.set_visible(False)

    fig.tight_layout()
    _savefig(fig, save_path)
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# 3. Clinical correlation scatter
# ---------------------------------------------------------------------------

def plot_clinical_scatter(
    df: pd.DataFrame,
    ppl_var: str,
    clinical_vars: list[str],
    title: str = "Clinical correlations",
    save_path: str | None = None,
):
    """Scatter plots of *ppl_var* vs each clinical variable (patients only), colored by diagnosis."""
    n = len(clinical_vars)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    diags_present = [dx for dx in DIAG_ORDER if dx in df["diagnosis"].values]
    palette = {dx: DIAG_COLORS.get(dx, "#888888") for dx in diags_present}

    for i, cvar in enumerate(clinical_vars):
        ax = axes_flat[i]
        sub = df[[ppl_var, cvar, "diagnosis"]].dropna()
        if len(sub) < 5:
            ax.set_visible(False)
            continue
        sns.scatterplot(
            data=sub, x=ppl_var, y=cvar, hue="diagnosis",
            hue_order=diags_present, palette=palette,
            alpha=0.6, s=25, ax=ax,
        )
        r, p = spearmanr(sub[ppl_var], sub[cvar])
        z = np.polyfit(sub[ppl_var], sub[cvar], 1)
        xs = np.linspace(sub[ppl_var].min(), sub[ppl_var].max(), 100)
        ax.plot(xs, np.polyval(z, xs), "k--", alpha=0.6)
        ax.set_xlabel(ppl_var)
        ax.set_ylabel(cvar)
        ax.set_title(f"r={r:.3f}, p={p:.4f}")
        ax.grid(alpha=0.2)
        if i > 0:
            ax.get_legend().remove()

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(title, y=1.02, fontsize=13)
    fig.tight_layout()
    _savefig(fig, save_path)
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# 4. Robustness: correlation across 3 settings
# ---------------------------------------------------------------------------

def plot_correlation_robustness(
    corr_dfs: dict[str, pd.DataFrame],
    save_path_scatter: str | None = None,
    save_path_bar: str | None = None,
):
    """Scatter + bar chart comparing clinical correlations across analysis settings.

    *corr_dfs*: dict mapping setting name → DataFrame with columns
    ['Clinical Variable', 'r', 'p'].
    """
    settings = list(corr_dfs.keys())
    clinical_vars = corr_dfs[settings[0]]["Clinical Variable"].tolist()

    # --- Bar chart ---
    fig_bar, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(clinical_vars))
    width = 0.8 / len(settings)
    for i, sname in enumerate(settings):
        vals = corr_dfs[sname].set_index("Clinical Variable").reindex(clinical_vars)["r"]
        ax.bar(x + i * width, vals, width, label=sname, alpha=0.8)
    ax.set_xticks(x + width * (len(settings) - 1) / 2)
    ax.set_xticklabels(clinical_vars, rotation=30, ha="right")
    ax.set_ylabel("Spearman r")
    ax.set_title("Correlation robustness across analysis settings")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5)
    fig_bar.tight_layout()
    _savefig(fig_bar, save_path_bar)
    plt.show()

    # --- Scatter grid ---
    ncols = len(settings)
    nrows = len(clinical_vars)
    fig_sc, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    for col_i, sname in enumerate(settings):
        cdf = corr_dfs[sname]
        for row_i, cvar in enumerate(clinical_vars):
            ax = axes[row_i][col_i]
            row = cdf[cdf["Clinical Variable"] == cvar]
            if row.empty:
                ax.set_visible(False)
                continue
            r_val = row["r"].values[0]
            p_val = row["p"].values[0]
            ax.text(0.5, 0.5, f"r={r_val:.3f}\np={p_val:.4f}",
                    ha="center", va="center", fontsize=14, transform=ax.transAxes)
            ax.set_title(f"{sname} | {cvar}" if row_i == 0 else cvar)
            if col_i == 0:
                ax.set_ylabel(cvar)
    fig_sc.suptitle("Correlation robustness (scatter grid)", y=1.01)
    fig_sc.tight_layout()
    _savefig(fig_sc, save_path_scatter)
    plt.show()

    return fig_bar, fig_sc


# ---------------------------------------------------------------------------
# 5. Alternative metrics heatmap
# ---------------------------------------------------------------------------

def plot_metrics_heatmap(
    corr_matrix: pd.DataFrame,
    pval_matrix: pd.DataFrame | None = None,
    title: str = "Alternative metrics × Clinical variables",
    save_path: str | None = None,
):
    """Heatmap of Spearman r-values with optional significance markers."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        corr_matrix, annot=True, fmt=".3f", cmap="RdBu_r",
        center=0, ax=ax, linewidths=0.5,
    )
    if pval_matrix is not None:
        for i in range(corr_matrix.shape[0]):
            for j in range(corr_matrix.shape[1]):
                p = pval_matrix.iloc[i, j]
                if p < 0.001:
                    marker = "***"
                elif p < 0.01:
                    marker = "**"
                elif p < 0.05:
                    marker = "*"
                else:
                    marker = ""
                if marker:
                    ax.text(
                        j + 0.5, i + 0.75, marker,
                        ha="center", va="center", fontsize=8, color="black",
                    )
    ax.set_title(title)
    fig.tight_layout()
    _savefig(fig, save_path)
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# 6. Extreme groups
# ---------------------------------------------------------------------------

def plot_extreme_groups(
    patients: pd.DataFrame,
    ppl_var: str,
    clinical_vars: list[str],
    save_path: str | None = None,
):
    """Boxplot + strip comparing top vs bottom quartile patients on clinical vars."""
    q75 = patients[ppl_var].quantile(0.75)
    q25 = patients[ppl_var].quantile(0.25)
    high_q = patients[patients[ppl_var] >= q75]
    low_q = patients[patients[ppl_var] <= q25]

    n = len(clinical_vars)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    if n == 1:
        axes = [axes]

    for i, cvar in enumerate(clinical_vars):
        ax = axes[i]
        data_h = high_q[cvar].dropna()
        data_l = low_q[cvar].dropna()
        bp = ax.boxplot([data_l, data_h], labels=["Low PPL", "High PPL"], patch_artist=True, widths=0.5)
        bp["boxes"][0].set_facecolor("#3498db")
        bp["boxes"][1].set_facecolor("#e74c3c")
        for k, d in enumerate([data_l, data_h]):
            ax.scatter(np.random.normal(k + 1, 0.04, size=len(d)), d, alpha=0.4, s=12, color="black")
        if len(data_h) > 1 and len(data_l) > 1:
            u, p = mannwhitneyu(data_h, data_l, alternative="two-sided")
            ax.set_title(f"{cvar}\np={p:.4f}")
        else:
            ax.set_title(cvar)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Extreme groups: top vs bottom PPL quartile", y=1.02)
    fig.tight_layout()
    _savefig(fig, save_path)
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# 7. TLI subscale scatter
# ---------------------------------------------------------------------------

def plot_tli_subscales(
    patients: pd.DataFrame,
    ppl_var: str,
    save_path: str | None = None,
):
    """Two-panel scatter: PPL vs TLI Impoverished and TLI Disorganization, colored by diagnosis."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    diags_present = [dx for dx in DIAG_ORDER if dx in patients["diagnosis"].values]
    palette = {dx: DIAG_COLORS.get(dx, "#888888") for dx in diags_present}

    for ax, cvar in zip([ax1, ax2], ["TLI_Impoverished", "TLI_Disorganization"]):
        sub = patients[[ppl_var, cvar, "diagnosis"]].dropna()
        if len(sub) < 5:
            ax.set_visible(False)
            continue
        sns.scatterplot(
            data=sub, x=ppl_var, y=cvar, hue="diagnosis",
            hue_order=diags_present, palette=palette,
            alpha=0.6, s=25, ax=ax,
        )
        r, p = spearmanr(sub[ppl_var], sub[cvar])
        z = np.polyfit(sub[ppl_var], sub[cvar], 1)
        xs = np.linspace(sub[ppl_var].min(), sub[ppl_var].max(), 100)
        ax.plot(xs, np.polyval(z, xs), "k--", alpha=0.6)
        ax.set_xlabel(ppl_var)
        ax.set_ylabel(cvar)
        ax.set_title(f"{cvar}: r={r:.3f}, p={p:.4f}")
        ax.grid(alpha=0.2)

    fig.suptitle("TLI subscale analysis", y=1.02)
    fig.tight_layout()
    _savefig(fig, save_path)
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# 8. ROC curves
# ---------------------------------------------------------------------------

def plot_roc_curves(
    roc_results: list[dict],
    targets: dict[str, list[dict]],
    save_path: str | None = None,
):
    """Plot ROC curves per target. *targets* maps target_name → list of roc dicts."""
    n = len(targets)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (tname, rocs) in zip(axes, targets.items()):
        for roc in rocs:
            ax.plot(roc["fpr"], roc["tpr"], label=f'{roc["metric"]} (AUC={roc["auc"]:.3f})')
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title(f"ROC: {tname}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

    fig.tight_layout()
    _savefig(fig, save_path)
    plt.show()
    return fig


def plot_incremental_value_roc(
    y_true: np.ndarray,
    models_dict: dict,
    save_path: str | None = None,
    show_title: bool = True,
    show_legend: bool = True,
):
    """
    Plot ROC curves for models in the comprehensive comparison.
    """
    from sklearn.metrics import roc_curve
    fig, ax = plt.subplots(figsize=(7, 6))

    # Styling configuration
    styles = {
        "PPL Only": {"color": "#E67E22", "ls": "-", "lw": 1.5},
        "Median LLM Loss Only": {"color": "#E67E22", "ls": "-", "lw": 1.5},
        "Length Only": {"color": "gray", "ls": "--", "lw": 1.5},
        "Length + PPL": {"color": "#E74C3C", "ls": "-", "lw": 2.5},
        "Length + Median LLM Loss": {"color": "#E74C3C", "ls": "-", "lw": 2.5},
        "Confounders Only": {"color": "#34495E", "ls": "--", "lw": 1.5},
        "Confounders + PPL": {"color": "#27AE60", "ls": "-", "lw": 2.5},
        "Confounders + Median LLM Loss": {"color": "#27AE60", "ls": "-", "lw": 2.5},
        "Confounders + Proximity": {"color": "#8E44AD", "ls": "-.", "lw": 2.0},
        "Confounders + Proximity + Median LLM Loss": {"color": "#2980B9", "ls": "-", "lw": 2.5},
    }

    for name, result in models_dict.items():
        fpr, tpr, _ = roc_curve(y_true, result["prob"])
        style = styles.get(name, {"color": "black", "ls": "-", "lw": 1})
        ax.plot(fpr, tpr, label=f"{name} (AUC={result['auc']:.3f})", **style)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    if show_title:
        ax.set_title("Incremental Predictive Value (Multi-Model Comparison)")
    if show_legend:
        ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    _savefig(fig, save_path)
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# 9. Confounder analysis (6-panel)
# ---------------------------------------------------------------------------

def plot_confounder_analysis(
    df: pd.DataFrame,
    ppl_var: str,
    wc_var: str,
    save_path: str | None = None,
):
    """Six-panel confounder visualization."""
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig)

    df = df.copy()
    df["log_words"] = np.log1p(df[wc_var])

    # Panel 1: Word count by group
    ax1 = fig.add_subplot(gs[0, 0])
    for is_pat, lbl, clr in [(0, "HC", GROUP_COLORS["control"]), (1, "Patient", GROUP_COLORS["patient"])]:
        vals = df[df["is_patient"] == is_pat][wc_var].dropna()
        parts = ax1.violinplot([vals], positions=[is_pat], showmeans=True, showmedians=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(clr)
            pc.set_alpha(0.6)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["HC", "Patient"])
    ax1.set_ylabel("Word count")
    ax1.set_title("Word count by group")

    # Panel 2: Education by group
    ax2 = fig.add_subplot(gs[0, 1])
    if "education" in df.columns:
        for is_pat, lbl, clr in [(0, "HC", GROUP_COLORS["control"]), (1, "Patient", GROUP_COLORS["patient"])]:
            vals = df[df["is_patient"] == is_pat]["education"].dropna()
            parts = ax2.violinplot([vals], positions=[is_pat], showmeans=True, showmedians=True)
            for pc in parts["bodies"]:
                pc.set_facecolor(clr)
                pc.set_alpha(0.6)
        ax2.set_xticks([0, 1])
        ax2.set_xticklabels(["HC", "Patient"])
        ax2.set_ylabel("Education (years)")
        ax2.set_title("Education by group")

    # Panel 3: PPL vs word count
    ax3 = fig.add_subplot(gs[0, 2])
    for is_pat, clr, lbl in [(0, GROUP_COLORS["control"], "HC"), (1, GROUP_COLORS["patient"], "Patient")]:
        sub = df[df["is_patient"] == is_pat]
        ax3.scatter(sub[wc_var], sub[ppl_var], alpha=0.5, s=15, color=clr, label=lbl)
    ax3.set_xlabel("Word count")
    ax3.set_ylabel(ppl_var)
    ax3.set_title("PPL vs Word count")
    ax3.legend()
    ax3.grid(alpha=0.2)

    # Panel 4: PPL vs education
    ax4 = fig.add_subplot(gs[1, 0])
    if "education" in df.columns:
        for is_pat, clr, lbl in [(0, GROUP_COLORS["control"], "HC"), (1, GROUP_COLORS["patient"], "Patient")]:
            sub = df[df["is_patient"] == is_pat]
            ax4.scatter(sub["education"], sub[ppl_var], alpha=0.5, s=15, color=clr, label=lbl)
        ax4.set_xlabel("Education")
        ax4.set_ylabel(ppl_var)
        ax4.set_title("PPL vs Education")
        ax4.legend()
        ax4.grid(alpha=0.2)

    # Panel 5: Residualized PPL (word count only)
    ax5 = fig.add_subplot(gs[1, 1])
    tmp = df[[ppl_var, "log_words", "is_patient"]].dropna()
    X_w = sm.add_constant(tmp[["log_words"]])
    resid_w = sm.OLS(tmp[ppl_var], X_w).fit().resid
    for is_pat, clr, lbl in [(0, GROUP_COLORS["control"], "HC"), (1, GROUP_COLORS["patient"], "Patient")]:
        vals = resid_w[tmp["is_patient"] == is_pat]
        parts = ax5.violinplot([vals], positions=[is_pat], showmeans=True, showmedians=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(clr)
            pc.set_alpha(0.6)
    ax5.set_xticks([0, 1])
    ax5.set_xticklabels(["HC", "Patient"])
    ax5.set_ylabel("Residual PPL")
    ax5.set_title("Adj. for word count")

    # Panel 6: Residualized PPL (word count + education)
    ax6 = fig.add_subplot(gs[1, 2])
    if "education" in df.columns:
        tmp2 = df[[ppl_var, "log_words", "education", "is_patient"]].dropna()
        X_we = sm.add_constant(tmp2[["log_words", "education"]])
        resid_we = sm.OLS(tmp2[ppl_var], X_we).fit().resid
        for is_pat, clr, lbl in [(0, GROUP_COLORS["control"], "HC"), (1, GROUP_COLORS["patient"], "Patient")]:
            vals = resid_we[tmp2["is_patient"] == is_pat]
            parts = ax6.violinplot([vals], positions=[is_pat], showmeans=True, showmedians=True)
            for pc in parts["bodies"]:
                pc.set_facecolor(clr)
                pc.set_alpha(0.6)
        ax6.set_xticks([0, 1])
        ax6.set_xticklabels(["HC", "Patient"])
        ax6.set_ylabel("Residual PPL")
        ax6.set_title("Adj. for words + education")

    fig.tight_layout()
    _savefig(fig, save_path)
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# 10. Raw vs Residualized comparison (4-panel)
# ---------------------------------------------------------------------------

def plot_raw_vs_residualized(
    df: pd.DataFrame,
    ppl_var: str,
    wc_var: str,
    confounders: list[str] | None = None,
    save_path: str | None = None,
):
    """Four-panel figure: KDE raw, KDE residualized, Cohen's d bar, rain plot.

    Uses the same residualization as the main pipeline (OLS on raw median PPL,
    fit on HC only) when *confounders* is provided.  Falls back to log_words +
    education if *confounders* is None (legacy behaviour).
    """
    from functions.analysis.statistics import residualize_ppl as _residualize_ppl

    analysis_df = df.copy()

    if confounders is not None:
        resid = _residualize_ppl(analysis_df, ppl_var, confounders)
        analysis_df["ppl_residual"] = np.nan
        analysis_df.loc[resid.index, "ppl_residual"] = resid
    else:
        # Legacy fallback: log-scale, words + education only
        analysis_df["log_words"] = np.log1p(analysis_df[wc_var])
        analysis_df["ppl_log"] = np.log(analysis_df[ppl_var].clip(lower=1e-6))
        if "education" in analysis_df.columns:
            tmp = analysis_df[["ppl_log", "log_words", "education"]].dropna()
            X = sm.add_constant(tmp[["log_words", "education"]])
        else:
            tmp = analysis_df[["ppl_log", "log_words"]].dropna()
            X = sm.add_constant(tmp[["log_words"]])
        resid = sm.OLS(tmp["ppl_log"], X).fit().resid
        analysis_df.loc[tmp.index, "ppl_residual"] = resid

    # Column used for the raw panels
    raw_col = ppl_var

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: KDE raw (median PPL)
    ax = axes[0, 0]
    for dx in DIAG_ORDER:
        sub = analysis_df[analysis_df["diagnosis"] == dx][raw_col].dropna()
        if len(sub) > 2:
            kde = gaussian_kde(sub)
            xs = np.linspace(sub.min(), sub.max(), 200)
            ax.plot(xs, kde(xs), color=DIAG_COLORS[dx], label=dx)
            ax.axvline(sub.mean(), color=DIAG_COLORS[dx], linestyle="--", alpha=0.5)
    ax.set_title("a) Raw median perplexity by diagnosis")
    ax.set_xlabel("Median PPL")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.2)

    # Panel B: KDE residualized
    ax = axes[0, 1]
    for dx in DIAG_ORDER:
        sub = analysis_df[analysis_df["diagnosis"] == dx]["ppl_residual"].dropna()
        if len(sub) > 2:
            kde = gaussian_kde(sub)
            xs = np.linspace(sub.min(), sub.max(), 200)
            ax.plot(xs, kde(xs), color=DIAG_COLORS[dx], label=dx)
            ax.axvline(sub.mean(), color=DIAG_COLORS[dx], linestyle="--", alpha=0.5)
    ax.set_title("b) Residualized perplexity by diagnosis")
    ax.set_xlabel("Residualized Median PPL")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.2)

    # Panel C: Cohen's d comparison
    ax = axes[1, 0]
    hc_raw = analysis_df[analysis_df["diagnosis"] == "HC"][raw_col].dropna()
    hc_res = analysis_df[analysis_df["diagnosis"] == "HC"]["ppl_residual"].dropna()
    d_raw, d_res = [], []
    dx_labels = [dx for dx in DIAG_ORDER if dx != "HC"]
    for dx in dx_labels:
        grp_raw = analysis_df[analysis_df["diagnosis"] == dx][raw_col].dropna()
        grp_res = analysis_df[analysis_df["diagnosis"] == dx]["ppl_residual"].dropna()
        ps = np.sqrt(((len(grp_raw) - 1) * grp_raw.std() ** 2 + (len(hc_raw) - 1) * hc_raw.std() ** 2) / (len(grp_raw) + len(hc_raw) - 2))
        d_raw.append((grp_raw.mean() - hc_raw.mean()) / ps if ps > 0 else 0)
        ps2 = np.sqrt(((len(grp_res) - 1) * grp_res.std() ** 2 + (len(hc_res) - 1) * hc_res.std() ** 2) / (len(grp_res) + len(hc_res) - 2))
        d_res.append((grp_res.mean() - hc_res.mean()) / ps2 if ps2 > 0 else 0)
    x = np.arange(len(dx_labels))
    ax.bar(x - 0.2, d_raw, 0.35, label="Raw", color="#3498db", alpha=0.8)
    ax.bar(x + 0.2, d_res, 0.35, label="Residualized", color="#e74c3c", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(dx_labels)
    ax.set_ylabel("Cohen's d (vs HC)")
    ax.set_title("a) Effect sizes: raw vs residualized")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Panel D: rain plot of residualized perplexity by diagnosis
    ax = axes[1, 1]
    diags_present = [dx for dx in DIAG_ORDER if dx in analysis_df["diagnosis"].values]
    rain_df = analysis_df[analysis_df["diagnosis"].isin(diags_present)][["diagnosis", "ppl_residual"]].dropna()

    # Half-violin (cloud)
    for i, dx in enumerate(diags_present):
        vals = rain_df[rain_df["diagnosis"] == dx]["ppl_residual"].values
        if len(vals) < 3:
            continue
        kde = gaussian_kde(vals)
        ys = np.linspace(vals.min(), vals.max(), 200)
        density = kde(ys)
        # Scale density so the widest part is ~0.35
        density = density / density.max() * 0.35
        # Draw half-violin on the right side of position
        ax.fill_betweenx(ys, i, i + density, color=DIAG_COLORS.get(dx, "#888"),
                         alpha=0.4, linewidth=0)
        ax.plot(i + density, ys, color=DIAG_COLORS.get(dx, "#888"), linewidth=1)

    # Rain (individual points, jittered to the left)
    for i, dx in enumerate(diags_present):
        vals = rain_df[rain_df["diagnosis"] == dx]["ppl_residual"].values
        jitter = np.random.uniform(-0.15, -0.02, size=len(vals))
        ax.scatter(i + jitter, vals, color=DIAG_COLORS.get(dx, "#888"),
                   alpha=0.4, s=10, edgecolors="none", zorder=2)

    # Boxplot (narrow, on the center)
    bp_data = [rain_df[rain_df["diagnosis"] == dx]["ppl_residual"].values for dx in diags_present]
    bp = ax.boxplot(bp_data, positions=range(len(diags_present)), widths=0.1,
                    patch_artist=True, showfliers=False, zorder=3)
    for patch, dx in zip(bp["boxes"], diags_present):
        patch.set_facecolor(DIAG_COLORS.get(dx, "#888"))
        patch.set_alpha(0.8)
    for element in ["whiskers", "caps", "medians"]:
        for line in bp[element]:
            line.set_color("black")
            line.set_linewidth(1)

    ax.set_xticks(range(len(diags_present)))
    ax.set_xticklabels(diags_present)
    ax.set_ylabel("Residualized Median PPL")
    ax.set_title("b) Residualized perplexity by diagnosis")
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    _savefig(fig, save_path)
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# 11. Basic raw perplexity (2-panel: boxplot + violin)
# ---------------------------------------------------------------------------

def plot_raw_perplexity_basic(
    df: pd.DataFrame,
    ppl_var: str,
    save_path: str | None = None,
):
    """Two-panel overview: jittered boxplot by diagnosis + violin plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    diags_present = [d for d in DIAG_ORDER if d in df["diagnosis"].values]
    palette = {dx: DIAG_COLORS.get(dx, "#888") for dx in diags_present}

    # Panel A: Boxplot with hue
    sns.boxplot(
        data=df, x="diagnosis", y=ppl_var, hue="diagnosis",
        order=diags_present, palette=palette,
        width=0.5, ax=ax1, legend=False,
    )
    sns.stripplot(
        data=df, x="diagnosis", y=ppl_var, hue="diagnosis",
        order=diags_present, palette={dx: "black" for dx in diags_present},
        alpha=0.3, size=3, jitter=True, ax=ax1, legend=False,
    )
    data_groups = [df[df["diagnosis"] == dx][ppl_var].dropna().values for dx in diags_present if len(df[df["diagnosis"] == dx][ppl_var].dropna()) > 0]
    stat, p_kw = kruskal(*data_groups)
    ax1.set_title(f"A) Boxplot by diagnosis (KW p={p_kw:.4f})")
    ax1.set_ylabel(ppl_var)
    ax1.grid(axis="y", alpha=0.3)

    # Panel B: Violin with hue
    sns.violinplot(
        data=df, x="diagnosis", y=ppl_var, hue="diagnosis",
        order=diags_present, palette=palette,
        inner="quartile", ax=ax2, legend=False,
    )

    # Add HC vs Patient comparison in title
    hc = df[df["is_patient"] == 0][ppl_var].dropna()
    pat = df[df["is_patient"] == 1][ppl_var].dropna()
    u, p_mw = mannwhitneyu(pat, hc, alternative="two-sided")
    pooled = np.sqrt(((len(pat) - 1) * pat.std() ** 2 + (len(hc) - 1) * hc.std() ** 2) / (len(pat) + len(hc) - 2))
    d = (pat.mean() - hc.mean()) / pooled if pooled > 0 else 0
    ax2.set_title(f"B) Violin (HC vs Pat: p={p_mw:.4f}, d={d:.3f})")
    ax2.set_ylabel(ppl_var)
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _savefig(fig, save_path)
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Matching balance: table + Love plot
# ---------------------------------------------------------------------------

def plot_matching_balance(
    df_full: pd.DataFrame,
    df_matched: pd.DataFrame,
    match_vars: list[str],
    figure_dir: str = "figures",
) -> pd.DataFrame:
    """Display a balance table and Love plot comparing before/after matching.

    Returns the balance DataFrame for further use.
    """
    from functions.analysis.statistics import matching_balance_table

    bal_before = matching_balance_table(df_full, match_vars)
    bal_after = matching_balance_table(df_matched, match_vars)

    bal = bal_before[["Variable"]].copy()
    bal["HC (before)"] = bal_before["HC"]
    bal["Patient (before)"] = bal_before["Patient"]
    bal["SMD (before)"] = bal_before["SMD"]
    bal["HC (after)"] = bal_after["HC"]
    bal["Patient (after)"] = bal_after["Patient"]
    bal["SMD (after)"] = bal_after["SMD"]
    bal["p (after)"] = bal_after["p-value"]

    display(bal.style.format({
        "SMD (before)": "{:.3f}", "SMD (after)": "{:.3f}", "p (after)": "{:.4f}",
    }))

    # Love plot
    fig, ax = plt.subplots(figsize=(6, 3.5))
    y = np.arange(len(match_vars))
    ax.scatter(bal["SMD (before)"], y, marker="o", s=60, label="Before matching", zorder=3)
    ax.scatter(bal["SMD (after)"], y, marker="D", s=60, label="After matching", zorder=3)
    ax.axvline(0.1, color="grey", ls="--", lw=1, label="SMD = 0.1 threshold")
    ax.set_yticks(y)
    ax.set_yticklabels(bal["Variable"])
    ax.set_xlabel("Standardised Mean Difference (SMD)")
    ax.set_title("Covariate balance: HC vs Patients")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    _savefig(fig, os.path.join(figure_dir, "matching_balance_love_plot.png"))
    plt.show()

    print(f"\nN before: {len(df_full)} | N after matching: {len(df_matched)}")
    print(f"SMD < 0.1 on all vars after matching: {(bal['SMD (after)'] < 0.1).all()}")
    return bal


# ---------------------------------------------------------------------------
# Clinical correlation tables: format, clean, and save
# ---------------------------------------------------------------------------

def format_and_save_clinical_correlations(
    merged_corr: pd.DataFrame,
    setting_name: str,
    figure_dir: str,
    fmt_p_func,
) -> pd.DataFrame:
    """
    Format and save clinical correlation tables (raw vs residualized).
    
    Parameters
    ----------
    merged_corr : pd.DataFrame
        DataFrame with columns like p_raw, p_resid, p_fdr_raw, p_fdr_resid, r_raw, r_resid.
    setting_name : str
        Name of the setting (e.g. 'with_outliers', 'without_outliers').
    figure_dir : str
        Base figure directory.
    fmt_p_func : callable
        Function to format p-values (e.g., fmt_p).
    
    Returns
    -------
    pd.DataFrame
        Formatted dataframe, sorted by p_raw, with stars instead of True/False.
    """
    
    # Convert p-values to publication-style stars
    def _p_to_stars(p):
        if pd.isna(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    if "p_fdr_raw" in merged_corr.columns:
        merged_corr["sig_fdr_raw"] = merged_corr["p_fdr_raw"].apply(_p_to_stars)
    if "p_fdr_resid" in merged_corr.columns:
        merged_corr["sig_fdr_resid"] = merged_corr["p_fdr_resid"].apply(_p_to_stars)

    merged_corr = merged_corr.sort_values(by="p_raw", ascending=True, na_position="last")

    # Save one table per setting for publication use
    tables_dir = os.path.join(figure_dir, setting_name, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    corr_out_csv = os.path.join(tables_dir, "clinical_correlations_raw_vs_residualized.csv")
    corr_out_xlsx = os.path.join(tables_dir, "clinical_correlations_raw_vs_residualized.xlsx")
    

    # Clean column names and formatting for presentation
    df_clinical = merged_corr.copy()
    df_clinical = df_clinical.rename(columns={
        "Clinical Variable": "Variable",
        "r_raw": "ρ (raw)",
        "p_raw": "P (raw)",
        "n_raw": "n",
        "p_fdr_raw": "P FDR (raw)",
        "sig_fdr_raw": "Sig.",
        "r_resid": "ρ (residualized)",
        "p_resid": "P (residualized)",
        "n_resid": "n",
        "p_fdr_resid": "P FDR (residualized)",
        "sig_fdr_resid": "Sig.",
    })

    # Clean variable names for better readability
    name_map = {
        "YMRS_Sum": "YMRS total",
        "verbal_IQ": "Verbal IQ",
        "SAPS_Sum_ENIGMA": "SAPS total",
        "SANS_AbulieApathie_ENIGMA": "SANS abulia/apathy",
        "GAF": "GAF",
        "SAPS_PositiveFormaleDenkstoerung_ENIGMA": "SAPS positive FTD",
        "SANS_Affektverflachung_ENIGMA": "SANS affective flattening",
        "SANS_AlogieParalogie_ENIGMA": "SANS alogia",
        "LexicalVF": "Lexical verbal fluency",
        "TLI_Impoverished": "TLI impoverished",
        "SAPS_Wahn_ENIGMA": "SAPS delusions",
        "SemanticVF": "Semantic verbal fluency",
        "SANS_Anhedonie_ENIGMA": "SANS anhedonia",
        "SAPS_Halluzinationen_ENIGMA": "SAPS hallucinations",
        "HAMD_Sum17": "HAMD-17 total",
        "SAPS_BizarresVerhalten_ENIGMA": "SAPS bizarre behaviour",
        "TLI_Non_specific": "TLI non-specific",
        "TLI_Disorganization": "TLI disorganization",
    }
    if "Variable" in df_clinical.columns:
        df_clinical["Variable"] = df_clinical["Variable"].map(name_map).fillna(df_clinical["Variable"])

    # NPJ-style rounding
    if "ρ (raw)" in df_clinical.columns:
        df_clinical["ρ (raw)"] = df_clinical["ρ (raw)"].round(2)
    if "ρ (residualized)" in df_clinical.columns:
        df_clinical["ρ (residualized)"] = df_clinical["ρ (residualized)"].round(2)
    
    # Format p-values using provided function
    for col in ["P (raw)", "P FDR (raw)", "P (residualized)", "P FDR (residualized)"]:
        if col in df_clinical.columns:
            df_clinical[col] = df_clinical[col].apply(fmt_p_func)

    df_clinical.to_csv(corr_out_csv, index=False)
    df_clinical.to_excel(corr_out_xlsx, index=False)
    print(f"\n  Clinical correlations (raw vs residualized), sorted by p_raw:")
    print(merged_corr.to_string(index=False))
    print(f"  Saved tables: {corr_out_csv} | {corr_out_xlsx}")
    
    return merged_corr



 
# ── Helper: Mean ± SD, rounded to 1 decimal ──────
def mean_sd(series):
    s = series.dropna()
    if len(s) == 0:
        return "—"
    return f"{s.mean():.1f} ± {s.std():.1f}"
 
# ── Helper: Median [IQR] ──────
def median_iqr(series):
    s = series.dropna()
    if len(s) == 0:
        return "—"
    return f"{s.median():.1f} [{s.quantile(0.25):.1f}–{s.quantile(0.75):.1f}]"
 
# ── Helper: n (%) for binary/categorical ──────
def n_pct(series, value):
    s = series.dropna()
    n = (s == value).sum()
    pct = n / len(s) * 100
    return f"{n} ({pct:.1f}%)"
 
 
# ══════════════════════════════════════════════════════════════════
# TABLE 1: Sample Characteristics (Main Table)
# ══════════════════════════════════════════════════════════════════
 
def create_table_demo(df, model_name="qwen"):
    """Create Table 1: Demographic and clinical characteristics."""
 
    groups = {"HC": df[df["is_patient"] == 0],
              "BD": df[df["diagnosis"] == "BD"],
              "MDD": df[df["diagnosis"] == "MDD"],
              "SSD": df[df["diagnosis"] == "SSD"]}
 
    patients = df[df["is_patient"] == 1]
    rows = []
 
    # ── n ──
    rows.append({
        "Variable": "n",
        "HC": str(len(groups["HC"])),
        "BD": str(len(groups["BD"])),
        "MDD": str(len(groups["MDD"])),
        "SSD": str(len(groups["SSD"])),
        "P": ""
    })
 
    # ── Demographic: continuous variables ──
    demo_vars = [
        ("Age, years", "age"),
        ("Education, years", "education"),
        ("Verbal IQ", "verbal_IQ"),
    ]
 
    for label, col in demo_vars:
        vals = []
        group_data = []
        for name in ["HC", "BD", "MDD", "SSD"]:
            g = groups[name][col].dropna()
            vals.append(mean_sd(groups[name][col]))
            group_data.append(g)
 
        # Kruskal-Wallis across all 4 groups
        clean = [g.values for g in group_data if len(g) > 0]
        if len(clean) >= 2:
            _, p = kruskal(*clean)
        else:
            p = np.nan
 
        rows.append({
            "Variable": label,
            "HC": vals[0], "BD": vals[1], "MDD": vals[2], "SSD": vals[3],
            "P": fmt_p(p) if not np.isnan(p) else "—"
        })
 
    # ── Sex: n (%) female ──
    sex_mapped = df["sex"].map({"M": 0, "F": 1, "m": 0, "f": 1,
                                 "male": 0, "female": 1, "Male": 0, "Female": 1})
    counts = pd.crosstab(df["diagnosis"].fillna("HC"), sex_mapped.fillna(0))
    try:
        _, p_sex, _, _ = chi2_contingency(counts)
    except:
        p_sex = np.nan
 
    sex_row = {"Variable": "Female, n (%)", "P": fmt_p(p_sex) if not np.isnan(p_sex) else "—"}
    for name in ["HC", "BD", "MDD", "SSD"]:
        g = groups[name]
        s = g["sex"].map({"M": 0, "F": 1, "m": 0, "f": 1,
                          "male": 0, "female": 1, "Male": 0, "Female": 1})
        n_f = s.sum()
        pct = n_f / len(s) * 100
        sex_row[name] = f"{int(n_f)} ({pct:.1f}%)"
    rows.append(sex_row)
 
    # ── Linguistic variables ──
    ling_vars = [
        ("Words per sentence", f"{model_name}_avg_words"),
        ("Number of sentences", f"{model_name}_n_sentences"),
    ]
 
    for label, col in ling_vars:
        vals = []
        group_data = []
        for name in ["HC", "BD", "MDD", "SSD"]:
            vals.append(mean_sd(groups[name][col]))
            group_data.append(groups[name][col].dropna())
 
        clean = [g.values for g in group_data if len(g) > 0]
        _, p = kruskal(*clean) if len(clean) >= 2 else (None, np.nan)
 
        rows.append({
            "Variable": label,
            "HC": vals[0], "BD": vals[1], "MDD": vals[2], "SSD": vals[3],
            "P": fmt_p(p) if not np.isnan(p) else "—"
        })
 
    # ── Separator row ──
    rows.append({
        "Variable": "Clinical measures",
        "HC": "", "BD": "", "MDD": "", "SSD": "", "P": ""
    })
 
    # ── Clinical variables (patients only) ──
    clinical_vars = [
        ("HAMD-17 total", "HAMD_Sum17"),
        ("YMRS total", "YMRS_Sum"),
        ("GAF", "GAF"),
        ("SAPS total", "SAPS_Sum_ENIGMA"),
        ("SANS alogia", "SANS_AlogieParalogie_ENIGMA"),
        ("SANS affective flattening", "SANS_Affektverflachung_ENIGMA"),
        ("SANS abulia/apathy", "SANS_AbulieApathie_ENIGMA"),
        ("SANS anhedonia", "SANS_Anhedonie_ENIGMA"),
        ("SAPS positive FTD", "SAPS_PositiveFormaleDenkstoerung_ENIGMA"),
        ("SAPS delusions", "SAPS_Wahn_ENIGMA"),
        ("SAPS hallucinations", "SAPS_Halluzinationen_ENIGMA"),
        ("SAPS bizarre behaviour", "SAPS_BizarresVerhalten_ENIGMA"),
        ("TLI impoverished", "TLI_Impoverished"),
        ("TLI disorganization", "TLI_Disorganization"),
        ("TLI non-specific", "TLI_Non_specific"),
        ("Lexical verbal fluency", "LexicalVF"),
        ("Semantic verbal fluency", "SemanticVF"),
    ]
 
    for label, col in clinical_vars:
        if col not in df.columns:
            continue
        vals_pat = []
        group_data = []
        for name in ["BD", "MDD", "SSD"]:
            vals_pat.append(mean_sd(groups[name][col]))
            group_data.append(groups[name][col].dropna())
 
        # Kruskal-Wallis across patient groups only
        clean = [g.values for g in group_data if len(g) > 0]
        if len(clean) >= 2 and all(len(g) > 0 for g in clean):
            _, p = kruskal(*clean)
        else:
            p = np.nan
 
        rows.append({
            "Variable": label,
            "HC": "—",
            "BD": vals_pat[0], "MDD": vals_pat[1], "SSD": vals_pat[2],
            "P": fmt_p(p) if not np.isnan(p) else "—"
        })
 
    table1 = pd.DataFrame(rows)
    # Reorder columns
    table1 = table1[["Variable", "HC", "BD", "MDD", "SSD", "P"]]
    return table1