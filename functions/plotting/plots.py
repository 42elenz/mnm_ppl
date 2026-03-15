"""Reusable plotting functions for perplexity analysis."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import gaussian_kde, kruskal, mannwhitneyu, spearmanr
from matplotlib.patches import Patch
import statsmodels.api as sm

# Consistent styling
DIAG_ORDER = ["HC", "BD", "MDD", "SSD"]
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
):
    """Boxplot + strip plot of *var* split by is_patient."""
    fig, ax = plt.subplots(figsize=(5, 5))
    plot_df = df.copy()
    plot_df["Group"] = plot_df["is_patient"].map({0: "HC", 1: "Patient"})
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
    ax.set_ylabel(var)
    ax.set_title(title or var)
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
    high_col = var.replace("_ppl_median", "_high")
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
    save_path: str | None = None,
):
    """Four-panel figure: KDE raw, KDE residualized, Cohen's d bar, summary text."""
    analysis_df = df.copy()
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

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: KDE raw
    ax = axes[0, 0]
    for dx in DIAG_ORDER:
        sub = analysis_df[analysis_df["diagnosis"] == dx]["ppl_log"].dropna()
        if len(sub) > 2:
            kde = gaussian_kde(sub)
            xs = np.linspace(sub.min(), sub.max(), 200)
            ax.plot(xs, kde(xs), color=DIAG_COLORS[dx], label=dx)
            ax.axvline(sub.mean(), color=DIAG_COLORS[dx], linestyle="--", alpha=0.5)
    ax.set_title("A) Raw log-perplexity by diagnosis")
    ax.set_xlabel("log(PPL)")
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
    ax.set_title("B) Residualized perplexity by diagnosis")
    ax.set_xlabel("Residual PPL")
    ax.legend()
    ax.grid(alpha=0.2)

    # Panel C: Cohen's d comparison
    ax = axes[1, 0]
    hc_raw = analysis_df[analysis_df["diagnosis"] == "HC"]["ppl_log"].dropna()
    hc_res = analysis_df[analysis_df["diagnosis"] == "HC"]["ppl_residual"].dropna()
    d_raw, d_res = [], []
    dx_labels = [dx for dx in DIAG_ORDER if dx != "HC"]
    for dx in dx_labels:
        grp_raw = analysis_df[analysis_df["diagnosis"] == dx]["ppl_log"].dropna()
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
    ax.set_title("C) Effect sizes: raw vs residualized")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Panel D: summary
    ax = axes[1, 1]
    ax.axis("off")
    text_lines = ["Summary:", ""]
    for dx, dr, dre in zip(dx_labels, d_raw, d_res):
        text_lines.append(f"{dx}: raw d={dr:.3f}, residual d={dre:.3f}")
    ax.text(0.1, 0.5, "\n".join(text_lines), fontsize=11, va="center", family="monospace")
    ax.set_title("D) Summary")

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
