"""Plot fragment-level velocity decomposition diagnostics."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from scipy.stats import mannwhitneyu
except ImportError:
    mannwhitneyu = None


FRAGMENT_ORDER = ["aromatic_ring", "ring", "rotatable_region", "other"]
SOURCE_LABELS = {"target": "target velocity", "pred": "predicted velocity"}
FILTERED_SUFFIX = "_filtered"
MIN_FRAGMENT_ATOMS = 3
MIN_FRAGMENT_TYPE_COUNT = 5


def _existing_order(values: Iterable[str]) -> List[str]:
    present = set(values)
    ordered = [value for value in FRAGMENT_ORDER if value in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def _figure_path(output_dir: Path, stem: str, suffix: str = "") -> Path:
    return output_dir / f"{stem}{suffix}.png"


def _apply_source_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["velocity_source_label"] = out["velocity_source"].map(SOURCE_LABELS).fillna(
        out["velocity_source"]
    )
    return out


def plot_residual_box(
    fragment_df: pd.DataFrame,
    output_dir: Path,
    suffix: str = "",
) -> None:
    data = _apply_source_labels(fragment_df)
    if data.empty:
        warnings.warn("No rows available for residual_ratio box plot.")
        return
    plt.figure(figsize=(8, 4.8))
    sns.boxplot(
        data=data,
        x="fragment_type",
        y="residual_ratio",
        hue="velocity_source_label",
        order=_existing_order(data["fragment_type"]),
        showfliers=False,
    )
    plt.xlabel("fragment_type")
    plt.ylabel("residual_ratio")
    plt.legend(title="")
    _savefig(_figure_path(output_dir, "fig_residual_ratio_by_fragment_type", suffix))


def plot_rigid_explain_box(
    fragment_df: pd.DataFrame,
    output_dir: Path,
    suffix: str = "",
) -> None:
    data = _apply_source_labels(fragment_df)
    if data.empty:
        warnings.warn("No rows available for rigid_explain_ratio box plot.")
        return
    plt.figure(figsize=(8, 4.8))
    sns.boxplot(
        data=data,
        x="fragment_type",
        y="rigid_explain_ratio",
        hue="velocity_source_label",
        order=_existing_order(data["fragment_type"]),
        showfliers=False,
    )
    plt.xlabel("fragment_type")
    plt.ylabel("rigid_explain_ratio")
    plt.legend(title="")
    _savefig(_figure_path(output_dir, "fig_rigid_explain_ratio_by_fragment_type", suffix))


def _molecule_long(fragment_df: pd.DataFrame) -> pd.DataFrame:
    return (
        fragment_df.groupby(
            ["molecule_id", "smiles", "num_rotatable_bonds", "velocity_source"],
            as_index=False,
        )["residual_ratio"]
        .mean()
        .rename(columns={"residual_ratio": "mean_residual_ratio"})
    )


def plot_rotatable_vs_residual(
    fragment_df: pd.DataFrame,
    output_dir: Path,
    suffix: str = "",
) -> None:
    data = _apply_source_labels(_molecule_long(fragment_df))
    if data.empty:
        warnings.warn("No rows available for rotatable-bond residual plot.")
        return
    plt.figure(figsize=(7, 4.8))
    sns.scatterplot(
        data=data,
        x="num_rotatable_bonds",
        y="mean_residual_ratio",
        hue="velocity_source_label",
        alpha=0.7,
    )
    for _, group in data.groupby("velocity_source_label"):
        if group["num_rotatable_bonds"].nunique() > 1:
            sns.regplot(
                data=group,
                x="num_rotatable_bonds",
                y="mean_residual_ratio",
                scatter=False,
                truncate=False,
            )
    plt.xlabel("num_rotatable_bonds")
    plt.ylabel("mean residual_ratio per molecule")
    plt.legend(title="")
    _savefig(_figure_path(output_dir, "fig_rotatable_bonds_vs_residual_ratio", suffix))


def plot_target_vs_pred(
    fragment_df: pd.DataFrame,
    output_dir: Path,
    suffix: str = "",
) -> None:
    if fragment_df.empty:
        warnings.warn("No rows available for target-vs-pred residual plot.")
        return
    pivot = fragment_df.pivot_table(
        index=["molecule_id", "fragment_id", "fragment_type"],
        columns="velocity_source",
        values="residual_ratio",
        aggfunc="mean",
    ).reset_index()
    if not {"target", "pred"}.issubset(pivot.columns):
        warnings.warn("Skipping target-vs-pred plot because target/pred rows are incomplete.")
        return

    plt.figure(figsize=(5.8, 5.2))
    sns.scatterplot(
        data=pivot,
        x="target",
        y="pred",
        hue="fragment_type",
        hue_order=_existing_order(pivot["fragment_type"]),
        alpha=0.75,
    )
    finite = pivot[["target", "pred"]].replace([np.inf, -np.inf], np.nan).dropna()
    if not finite.empty:
        low = float(finite.min().min())
        high = float(finite.max().max())
        plt.plot([low, high], [low, high], color="black", linewidth=1, linestyle="--")
    plt.xlabel("target residual_ratio")
    plt.ylabel("predicted residual_ratio")
    plt.legend(title="fragment_type")
    _savefig(_figure_path(output_dir, "fig_target_vs_pred_residual_ratio", suffix))


def plot_mean_residual_bar(
    fragment_df: pd.DataFrame,
    output_dir: Path,
    suffix: str = "",
) -> None:
    data = _apply_source_labels(fragment_df)
    if data.empty:
        warnings.warn("No rows available for mean residual_ratio bar plot.")
        return
    plt.figure(figsize=(8, 4.8))
    try:
        sns.barplot(
            data=data,
            x="fragment_type",
            y="residual_ratio",
            hue="velocity_source_label",
            order=_existing_order(data["fragment_type"]),
            errorbar="se",
        )
    except (AttributeError, TypeError):
        sns.barplot(
            data=data,
            x="fragment_type",
            y="residual_ratio",
            hue="velocity_source_label",
            order=_existing_order(data["fragment_type"]),
            ci=68,
        )
    plt.xlabel("fragment_type")
    plt.ylabel("mean residual_ratio")
    plt.legend(title="")
    _savefig(_figure_path(output_dir, "fig_mean_residual_ratio_bar", suffix))


def _mann_whitney_row(
    comparison: str,
    source: str,
    group_a_name: str,
    group_a: pd.Series,
    group_b_name: str,
    group_b: pd.Series,
) -> Dict[str, object]:
    group_a = group_a.dropna()
    group_b = group_b.dropna()
    p_value = np.nan
    if mannwhitneyu is not None and len(group_a) > 0 and len(group_b) > 0:
        p_value = float(mannwhitneyu(group_a, group_b, alternative="two-sided").pvalue)

    mean_a = float(group_a.mean()) if len(group_a) else np.nan
    mean_b = float(group_b.mean()) if len(group_b) else np.nan
    if np.isnan(mean_a) or np.isnan(mean_b):
        direction = "not_available"
    elif mean_a < mean_b:
        direction = f"{group_a_name}_lower"
    elif mean_a > mean_b:
        direction = f"{group_a_name}_higher"
    else:
        direction = "equal_means"

    return {
        "comparison": comparison,
        "velocity_source": source,
        "group_a": group_a_name,
        "group_b": group_b_name,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "median_a": float(group_a.median()) if len(group_a) else np.nan,
        "median_b": float(group_b.median()) if len(group_b) else np.nan,
        "p_value": p_value,
        "effect_direction": direction,
    }


def save_stat_tests(fragment_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    rows = []
    for source, source_df in fragment_df.groupby("velocity_source"):
        rows.append(
            _mann_whitney_row(
                "aromatic_ring vs rotatable_region",
                source,
                "aromatic_ring",
                source_df.loc[source_df["fragment_type"] == "aromatic_ring", "residual_ratio"],
                "rotatable_region",
                source_df.loc[
                    source_df["fragment_type"] == "rotatable_region", "residual_ratio"
                ],
            )
        )
        rows.append(
            _mann_whitney_row(
                "ring vs rotatable_region",
                source,
                "ring",
                source_df.loc[source_df["fragment_type"] == "ring", "residual_ratio"],
                "rotatable_region",
                source_df.loc[
                    source_df["fragment_type"] == "rotatable_region", "residual_ratio"
                ],
            )
        )
        rows.append(
            _mann_whitney_row(
                "rigid fragments vs flexible fragments",
                source,
                "rigid_fragments",
                source_df.loc[
                    source_df["fragment_type"].isin(["aromatic_ring", "ring"]),
                    "residual_ratio",
                ],
                "flexible_fragments",
                source_df.loc[
                    source_df["fragment_type"] == "rotatable_region", "residual_ratio"
                ],
            )
        )

    stat_df = pd.DataFrame(rows)
    stat_df.to_csv(output_path, index=False)
    return stat_df


def save_summary(fragment_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    summary_df = fragment_df.copy()
    if "fit_status" in summary_df.columns and "omega_norm" in summary_df.columns:
        summary_df.loc[summary_df["fit_status"] != "ok", "omega_norm"] = np.nan

    grouped = summary_df.groupby(["fragment_type", "velocity_source"], dropna=False)
    summary = grouped.agg(
        count=("residual_ratio", "size"),
        mean_residual_ratio=("residual_ratio", "mean"),
        std_residual_ratio=("residual_ratio", "std"),
        median_residual_ratio=("residual_ratio", "median"),
        mean_rigid_explain_ratio=("rigid_explain_ratio", "mean"),
        std_rigid_explain_ratio=("rigid_explain_ratio", "std"),
        median_rigid_explain_ratio=("rigid_explain_ratio", "median"),
        mean_omega_norm=("omega_norm", "mean"),
    ).reset_index()
    summary.to_csv(output_path, index=False)
    return summary


def _filter_ok_fragments(fragment_df: pd.DataFrame) -> pd.DataFrame:
    total_before = len(fragment_df)
    status = (
        fragment_df["fit_status"]
        if "fit_status" in fragment_df.columns
        else pd.Series("ok", index=fragment_df.index)
    )
    atom_counts = (
        fragment_df["num_fragment_atoms"]
        if "num_fragment_atoms" in fragment_df.columns
        else pd.Series(MIN_FRAGMENT_ATOMS, index=fragment_df.index)
    )

    filtered = fragment_df.loc[
        (status == "ok") & (atom_counts >= MIN_FRAGMENT_ATOMS)
    ].copy()

    print(f"total fragments before filtering: {total_before}")
    print(f"total fragments after filtering: {len(filtered)}")
    print(f"removed too_small fragments: {int((status == 'too_small').sum())}")
    print(f"removed rank_deficient fragments: {int((status == 'rank_deficient').sum())}")
    print(
        "fragments with num_fragment_atoms < "
        f"{MIN_FRAGMENT_ATOMS}: {int((atom_counts < MIN_FRAGMENT_ATOMS).sum())}"
    )
    print("fragment_type counts after filtering:")
    counts = filtered["fragment_type"].value_counts().sort_index()
    if counts.empty:
        print("(none)")
    else:
        print(counts.to_string())

    small_counts = counts[counts < MIN_FRAGMENT_TYPE_COUNT]
    for fragment_type, count in small_counts.items():
        warnings.warn(
            f"fragment_type '{fragment_type}' has only {int(count)} rows after filtering "
            f"(< {MIN_FRAGMENT_TYPE_COUNT})."
        )

    return filtered


def plot_all(input_dir: Path, no_filter: bool = False) -> None:
    fragment_path = input_dir / "decomp_by_fragment.csv"
    if not fragment_path.exists():
        raise FileNotFoundError(f"Missing fragment CSV: {fragment_path}")

    fragment_df = pd.read_csv(fragment_path)
    plot_df = fragment_df.copy() if no_filter else _filter_ok_fragments(fragment_df)
    figure_suffix = "" if no_filter else FILTERED_SUFFIX
    figures_dir = input_dir / "figures"

    sns.set_theme(style="whitegrid", context="paper")
    plot_residual_box(plot_df, figures_dir, suffix=figure_suffix)
    plot_rigid_explain_box(plot_df, figures_dir, suffix=figure_suffix)
    plot_rotatable_vs_residual(plot_df, figures_dir, suffix=figure_suffix)
    plot_target_vs_pred(plot_df, figures_dir, suffix=figure_suffix)
    plot_mean_residual_bar(plot_df, figures_dir, suffix=figure_suffix)

    stat_df = save_stat_tests(plot_df, input_dir / "stat_tests.csv")
    if not no_filter:
        stat_df.to_csv(input_dir / "stat_tests_filtered.csv", index=False)
        save_summary(plot_df, input_dir / "decomp_summary_filtered.csv")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        "-i",
        type=str,
        default="logs_velocity_decomp_first",
    )
    parser.add_argument(
        "--no_filter",
        action="store_true",
        help="Plot and test the raw unfiltered fragment rows.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_all(Path(args.input_dir), no_filter=args.no_filter)
