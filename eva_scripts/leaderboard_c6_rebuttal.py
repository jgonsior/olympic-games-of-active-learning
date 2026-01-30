import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kendalltau
from tabulate import tabulate

# =========================
# 1) SET YOUR PATHS HERE
# =========================
METRIC_PATHS = {
    # reference (paper terminology: "full mean")
    "full_auc": "/home/thiele/exp_results/full_exp_jan/plots/final_leaderboard/rank_sparse_zero_full_auc_weighted_f1-score.parquet",
    # phase-focused
    "ramp_up": "/home/thiele/exp_results/full_exp_jan/plots/final_leaderboard/rank_sparse_zero_ramp_up_auc_weighted_f1-score.parquet",
    "plateau": "/home/thiele/exp_results/full_exp_jan/plots/final_leaderboard/rank_sparse_zero_plateau_auc_weighted_f1-score.parquet",
    # truncation-based
    "first_5": "/home/thiele/exp_results/full_exp_jan/plots/final_leaderboard/rank_sparse_zero_first_5_weighted_f1-score.parquet",
    "last_5": "/home/thiele/exp_results/full_exp_jan/plots/final_leaderboard/rank_sparse_zero_last_5_weighted_f1-score.parquet",
    "final_value": "/home/thiele/exp_results/full_exp_jan/plots/final_leaderboard/rank_sparse_zero_final_value_weighted_f1-score.parquet",
}

REFERENCE_METRIC = "full_auc"

B_BOOT = 5000
SEED = 0

# Facet plot axis range (as requested earlier). Note:
# final_value has a small negative tail; values < 0 will not be shown with this xlim.
TAU_XLIM = (0.0, 1.0)

BINS_PER_DATASET = 20
BINS_BOOTSTRAP = 30

# Output filenames
FACET_PD_PDF = "tau_facet_per_dataset_all.pdf"
FACET_BOOT_PDF = "tau_facet_bootstrap_all.pdf"

# Preferred ordering (reference excluded automatically)
PREFERRED_ORDER = ["ramp_up", "plateau", "first_5", "last_5", "final_value"]

# Paper-consistent display names
DISPLAY_NAME = {
    "full_auc": "full mean",
    "ramp_up": "ramp-up",
    "plateau": "plateau",
    "first_5": "first 5",
    "last_5": "last 5",
    "final_value": "last value",
}
DISPLAY_NAME_LATEX = {
    "full_auc": r"full\_mean",
    "ramp_up": r"ramp\_up",
    "plateau": r"plateau",
    "first_5": r"first\_5",
    "last_5": r"last\_5",
    "final_value": r"final\_value",
}


# =========================
# Helpers
# =========================
def load_rank_matrix(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "dataset" in df.columns:
        df = df.set_index("dataset")
    if "Total" in df.index:
        df = df.drop(index="Total")
    df = df.select_dtypes(include="number")
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    return df


def align_all(mats: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    common_rows = None
    common_cols = None
    for df in mats.values():
        common_rows = (
            df.index if common_rows is None else common_rows.intersection(df.index)
        )
        common_cols = (
            df.columns if common_cols is None else common_cols.intersection(df.columns)
        )
    return {k: df.loc[common_rows, common_cols].copy() for k, df in mats.items()}


def leaderboard_order_from_matrix(R: pd.DataFrame) -> list[str]:
    mean_rank = R.mean(axis=0)
    return mean_rank.sort_values(ascending=True).index.to_list()


def kendall_tau_b_from_orders(order_a: list[str], order_b: list[str]) -> float:
    pos_a = {s: i for i, s in enumerate(order_a)}
    pos_b = {s: i for i, s in enumerate(order_b)}
    common = [s for s in order_a if s in pos_b]
    va = [pos_a[s] for s in common]
    vb = [pos_b[s] for s in common]
    tau, _ = kendalltau(va, vb, variant="b")
    return float(tau)


def per_dataset_tau_values(R_ref: pd.DataFrame, R_alt: pd.DataFrame) -> np.ndarray:
    taus = []
    for d in R_ref.index:
        o_ref = R_ref.loc[d].sort_values(ascending=True).index.to_list()
        o_alt = R_alt.loc[d].sort_values(ascending=True).index.to_list()
        taus.append(kendall_tau_b_from_orders(o_ref, o_alt))
    return np.array(taus, dtype=float)


def bootstrap_tau_values(
    R_ref: pd.DataFrame, R_alt: pd.DataFrame, B: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    datasets = R_ref.index.to_numpy()
    n = len(datasets)

    taus = np.empty(B, dtype=float)
    for i in range(B):
        sample = rng.choice(datasets, size=n, replace=True)
        Rr = R_ref.loc[sample]
        Ra = R_alt.loc[sample]
        o_ref = leaderboard_order_from_matrix(Rr)
        o_alt = leaderboard_order_from_matrix(Ra)
        taus[i] = kendall_tau_b_from_orders(o_ref, o_alt)
    return taus


def summarize_percentiles(x: np.ndarray, p_lo=2.5, p_hi=97.5) -> dict:
    x = np.asarray(x, dtype=float)
    return {
        "median": float(np.nanmedian(x)),
        "p_lo": float(np.nanpercentile(x, p_lo)),
        "p_hi": float(np.nanpercentile(x, p_hi)),
        "n": int(np.sum(~np.isnan(x))),
    }


def metric_display(m: str) -> str:
    return DISPLAY_NAME.get(m, m)


def metric_display_latex(m: str) -> str:
    return DISPLAY_NAME_LATEX.get(m, m)


def ordered_alt_metrics(all_metrics: list[str], ref: str) -> list[str]:
    alts = [m for m in all_metrics if m != ref]
    out = []
    seen = set()
    for m in PREFERRED_ORDER:
        if m in alts and m not in seen:
            out.append(m)
            seen.add(m)
    for m in alts:
        if m not in seen:
            out.append(m)
            seen.add(m)
    return out


def facet_histplot(
    data: pd.DataFrame,
    metric_order: list[str],
    xlim: tuple[float, float],
    bins: int,
    title: str,
    out_pdf: str,
):
    """
    data: DataFrame with columns ['tau', 'metric'] where metric is an internal key.
    Produces a FacetGrid with one row per metric, no 'metric = ...', and annotates the median value (horizontal, non-orange).
    """
    data = data.copy()
    data["metric_disp"] = data["metric"].map(metric_display)
    metric_disp_order = [metric_display(m) for m in metric_order]

    stats = (
        data.groupby("metric_disp")["tau"]
        .agg(
            median="median",
            p_lo=lambda x: np.nanpercentile(x, 2.5),
            p_hi=lambda x: np.nanpercentile(x, 97.5),
        )
        .reindex(metric_disp_order)
    )

    g = sns.FacetGrid(
        data,
        row="metric_disp",
        row_order=metric_disp_order,
        height=1.30,
        aspect=3.8,
        sharex=True,
        sharey=False,
        margin_titles=False,
        despine=True,
    )

    def _plot_hist(x, **kwargs):
        ax = plt.gca()
        sns.histplot(
            x,
            bins=bins,
            binrange=xlim,
            stat="count",
            kde=False,
            ax=ax,
        )

    g.map(_plot_hist, "tau")

    xr = xlim[1] - xlim[0]
    x_pad = 0.02 * xr  # for median label offset

    for ax, m_disp in zip(g.axes.flat, g.row_names):
        ax.set_title("")  # remove any seaborn facet title remnants

        row = stats.loc[m_disp]
        med = float(row["median"])
        lo = float(row["p_lo"])
        hi = float(row["p_hi"])

        # Vertical lines (keep median line orange if you like; text will not be orange)
        ax.axvline(med, linewidth=1.5, ls="--", alpha=0.90, color="orange")
        ax.axvline(lo, linewidth=1.0, ls="--", alpha=0.85)
        ax.axvline(hi, linewidth=1.0, ls="--", alpha=0.85)

        ax.set_xlim(*xlim)

        # CHANGED: remove per-row y-label
        ax.set_ylabel("")

        # Metric label only
        ax.text(
            0.01,
            0.92,
            m_disp,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )

        # Median annotation: horizontal, non-orange, with a small white bbox for readability
        y_top = ax.get_ylim()[1]
        x_text = med + x_pad
        # keep inside axes
        x_text = min(max(x_text, xlim[0] + 0.01 * xr), xlim[1] - 0.01 * xr)

        ax.text(
            x_text,
            y_top * 0.92,
            f"median={med:.2f}",
            ha="left",
            va="top",
            fontsize=8,
            color="black",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.5),
        )

    g.set_xlabels(r"Kendall's $\tau_b$")

    # NEW: single shared y-label (Matplotlib 3.4+). Fallback if not available.
    # if hasattr(g.fig, "supylabel"):
    #    g.fig.supylabel("Count")
    # else:
    #    g.fig.text(0.02, 0.5, "Count", rotation=90, va="center", ha="left")

    g.fig.suptitle(title, fontsize=10, y=1.01)
    g.tight_layout(rect=(0.04, 0, 1, 1))
    g.savefig(out_pdf)
    plt.close(g.fig)


# =========================
# Main
# =========================
def main():
    mats = {name: load_rank_matrix(path) for name, path in METRIC_PATHS.items()}
    mats = align_all(mats)

    metrics = list(METRIC_PATHS.keys())
    ref = REFERENCE_METRIC
    if ref not in mats:
        raise ValueError(
            f"REFERENCE_METRIC '{ref}' not found in METRIC_PATHS keys: {list(mats.keys())}"
        )

    print("Aligned matrices (datasets x strategies):")
    for m in metrics:
        print(f"  {m:12s}: {mats[m].shape}")

    n_datasets = mats[ref].shape[0]
    n_strat = mats[ref].shape[1]
    print(f"\nCommon intersection -> datasets={n_datasets}, strategies={n_strat}")

    # Global leaderboard orders (for the "all-data" tau column)
    orders_global = {m: leaderboard_order_from_matrix(mats[m]) for m in metrics}

    # Faceted plots (ALL metrics vs reference)
    alt_metrics = ordered_alt_metrics(metrics, ref)

    records_pd = []
    records_boot = []
    for m in alt_metrics:
        taus_pd = per_dataset_tau_values(mats[ref], mats[m])
        taus_boot = bootstrap_tau_values(mats[ref], mats[m], B=B_BOOT, seed=SEED)
        records_pd.extend([{"tau": float(t), "metric": m} for t in taus_pd])
        records_boot.extend([{"tau": float(t), "metric": m} for t in taus_boot])

        # Optional sanity note about clipping due to TAU_XLIM
        n_clip_pd = int(np.sum((taus_pd < TAU_XLIM[0]) | (taus_pd > TAU_XLIM[1])))
        n_clip_bt = int(np.sum((taus_boot < TAU_XLIM[0]) | (taus_boot > TAU_XLIM[1])))
        if n_clip_pd > 0 or n_clip_bt > 0:
            print(
                f"Note: {m} has values outside xlim={TAU_XLIM} "
                f"(per-dataset clipped count={n_clip_pd}/{len(taus_pd)}, "
                f"bootstrap clipped count={n_clip_bt}/{len(taus_boot)})."
            )

    df_pd = pd.DataFrame.from_records(records_pd)
    df_boot = pd.DataFrame.from_records(records_boot)

    ref_disp = metric_display(ref)

    facet_histplot(
        df_pd,
        metric_order=alt_metrics,
        xlim=TAU_XLIM,
        bins=BINS_PER_DATASET,
        title=f"Per-dataset Kendall $\\tau_b$ vs {ref_disp} (n={n_datasets})",
        out_pdf=FACET_PD_PDF,
    )
    facet_histplot(
        df_boot,
        metric_order=alt_metrics,
        xlim=TAU_XLIM,
        bins=BINS_BOOTSTRAP,
        title=f"Dataset bootstrap Kendall $\\tau_b$ vs {ref_disp} (B={B_BOOT})",
        out_pdf=FACET_BOOT_PDF,
    )

    print("\nSaved faceted histogram PDFs:")
    print(f"  {FACET_PD_PDF}")
    print(f"  {FACET_BOOT_PDF}")

    # LaTeX table (robustness vs reference) + LaTeX text block
    rows = []
    for m in alt_metrics:
        tau_all = kendall_tau_b_from_orders(orders_global[ref], orders_global[m])

        taus_pd = per_dataset_tau_values(mats[ref], mats[m])
        pd_sum = summarize_percentiles(taus_pd, 2.5, 97.5)

        taus_boot = bootstrap_tau_values(mats[ref], mats[m], B=B_BOOT, seed=SEED)
        boot_sum = summarize_percentiles(taus_boot, 2.5, 97.5)

        rows.append(
            [
                metric_display(m),
                f"{tau_all:.3f}",
                f"{pd_sum['median']:.3f} [{pd_sum['p_lo']:.3f}, {pd_sum['p_hi']:.3f}]",
                f"{boot_sum['median']:.3f} [{boot_sum['p_lo']:.3f}, {boot_sum['p_hi']:.3f}]",
            ]
        )

    headers = [
        "Aggregation",
        r"All-data $\tau_b$",
        r"Across-task median $\tau_b$ [2.5,97.5]",
        rf"Dataset-mix median $\tau_b$ [2.5,97.5] (B={B_BOOT})",
    ]
    latex_table = tabulate(rows, headers=headers, tablefmt="latex_booktabs")

    print("\n=========================")
    print("LATEX TABLE (paste into paper / rebuttal)")
    print("=========================")
    print(latex_table)

    latex_text = rf"""
\paragraph{{Robustness of aggregation-based leaderboards.}}
To quantify sensitivity to the choice of aggregation, we compare the strategy orderings induced by different aggregation metrics using Kendall's $\tau_b$ rank correlation coefficient (ties-aware) on dataset-wise rank matrices ({n_datasets} datasets $\times$ {n_strat} strategies).
We report (i) \emph{{across-task robustness}} by computing $\tau_b$ between per-dataset strategy orderings and summarizing the distribution across datasets, and (ii) \emph{{dataset-mix robustness}} via a dataset bootstrap with $B={B_BOOT}$ resamples (sampling {n_datasets} datasets with replacement) where we recompute the global leaderboard (mean rank across sampled datasets) per resample.
The resulting agreement values (Table~\ref{{tab:agg_robustness}}) show that phase-only and truncation-based summaries are not necessary to obtain a stable leaderboard and can be more sensitive than the full-curve aggregation: ramp-up-only summaries exhibit the largest deviations, while plateau-based summaries remain largely consistent with the full-curve reference.
Overall, the full-curve aggregation (\texttt{{{metric_display_latex(ref)}}}) is the safest default scalar summary.
""".strip()

    print("\n=========================")
    print("LATEX TEXT BLOCK (paste into paper / rebuttal)")
    print("=========================")
    print(latex_text)

    print("\nDone.")


if __name__ == "__main__":
    main()
