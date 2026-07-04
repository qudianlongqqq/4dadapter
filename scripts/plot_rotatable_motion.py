"""Plot rotatable-bond relative motion diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from scipy.stats import mannwhitneyu, spearmanr
except ImportError:
    mannwhitneyu = None
    spearmanr = None


SOURCE_LABELS = {"target": "target velocity", "pred": "predicted velocity"}
VALID_STATUS = "ok"
FLEXIBILITY_THRESHOLD = 5
MOLECULE_KEY_COLS = ["molecule_id", "smiles", "num_atoms"]


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def _apply_source_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["velocity_source_label"] = out["velocity_source"].map(SOURCE_LABELS).fillna(
        out["velocity_source"]
    )
    return out


def _valid_bond_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[
        (df["fit_status_a"] == VALID_STATUS) & (df["fit_status_b"] == VALID_STATUS)
    ].copy()


def _ensure_count_aliases(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "rdkit_num_rotatable_bonds" not in out.columns and "num_rotatable_bonds" in out.columns:
        out["rdkit_num_rotatable_bonds"] = out["num_rotatable_bonds"]
    if "num_rotatable_bonds" not in out.columns and "rdkit_num_rotatable_bonds" in out.columns:
        out["num_rotatable_bonds"] = out["rdkit_num_rotatable_bonds"]
    return out


def _candidate_counts(raw_bond_df: pd.DataFrame) -> pd.DataFrame:
    raw = _ensure_count_aliases(raw_bond_df)
    if raw.empty or "bond_index" not in raw.columns:
        return pd.DataFrame(columns=MOLECULE_KEY_COLS + ["candidate_rotatable_bonds"])

    keep_cols = [
        col
        for col in MOLECULE_KEY_COLS + ["num_rotatable_bonds", "rdkit_num_rotatable_bonds"]
        if col in raw.columns
    ]
    counts = (
        raw.dropna(subset=["bond_index"])
        .groupby(keep_cols, as_index=False)["bond_index"]
        .nunique()
        .rename(columns={"bond_index": "candidate_rotatable_bonds"})
    )
    return counts


def _add_candidate_counts(df: pd.DataFrame, raw_bond_df: pd.DataFrame) -> pd.DataFrame:
    out = _ensure_count_aliases(df)
    if "candidate_rotatable_bonds" in out.columns:
        return out

    counts = _candidate_counts(raw_bond_df)
    if counts.empty:
        out["candidate_rotatable_bonds"] = np.nan
        return out

    merge_cols = [col for col in MOLECULE_KEY_COLS if col in out.columns and col in counts.columns]
    out = out.merge(
        counts[merge_cols + ["candidate_rotatable_bonds"]],
        on=merge_cols,
        how="left",
    )
    return out


def _derive_molecule_error_df(errors_df: pd.DataFrame, raw_bond_df: pd.DataFrame) -> pd.DataFrame:
    errors = _add_candidate_counts(errors_df, raw_bond_df)
    columns = [
        "molecule_id",
        "smiles",
        "num_atoms",
        "num_rotatable_bonds",
        "rdkit_num_rotatable_bonds",
        "candidate_rotatable_bonds",
        "matched_rotatable_bonds",
        "mean_abs_torsion_error",
        "sum_abs_torsion_error",
        "max_abs_torsion_error",
        "mean_abs_relative_rotation_error",
        "sum_abs_relative_rotation_error",
    ]
    if errors.empty:
        return pd.DataFrame(columns=columns)

    group_cols = [
        col
        for col in [
            "molecule_id",
            "smiles",
            "num_atoms",
            "num_rotatable_bonds",
            "rdkit_num_rotatable_bonds",
            "candidate_rotatable_bonds",
        ]
        if col in errors.columns
    ]
    out = errors.groupby(group_cols, as_index=False).agg(
        matched_rotatable_bonds=("bond_index", "size"),
        mean_abs_torsion_error=("abs_error_torsion_velocity", "mean"),
        sum_abs_torsion_error=("abs_error_torsion_velocity", "sum"),
        max_abs_torsion_error=("abs_error_torsion_velocity", "max"),
        mean_abs_relative_rotation_error=("abs_error_relative_rotation_norm", "mean"),
        sum_abs_relative_rotation_error=("abs_error_relative_rotation_norm", "sum"),
    )
    return out


def plot_torsion_distribution(bond_df: pd.DataFrame, output_dir: Path) -> None:
    data = _apply_source_labels(bond_df.dropna(subset=["torsion_velocity"]))
    if data.empty:
        return
    plt.figure(figsize=(7.2, 4.8))
    sns.histplot(
        data=data,
        x="torsion_velocity",
        hue="velocity_source_label",
        bins=40,
        kde=True,
        element="step",
        stat="density",
        common_norm=False,
    )
    plt.xlabel("torsion_velocity")
    plt.ylabel("density")
    _savefig(output_dir / "fig_torsion_velocity_distribution.png")


def plot_abs_torsion_distribution(bond_df: pd.DataFrame, output_dir: Path) -> None:
    data = _apply_source_labels(bond_df.dropna(subset=["abs_torsion_velocity"]))
    if data.empty:
        return
    plt.figure(figsize=(7.2, 4.8))
    sns.histplot(
        data=data,
        x="abs_torsion_velocity",
        hue="velocity_source_label",
        bins=40,
        kde=True,
        element="step",
        stat="density",
        common_norm=False,
    )
    plt.xlabel("abs_torsion_velocity")
    plt.ylabel("density")
    _savefig(output_dir / "fig_abs_torsion_velocity_distribution.png")


def plot_target_vs_pred(errors_df: pd.DataFrame, output_dir: Path) -> None:
    data = errors_df.dropna(
        subset=["target_torsion_velocity", "pred_torsion_velocity"]
    ).copy()
    if data.empty:
        return
    plt.figure(figsize=(5.8, 5.2))
    sns.scatterplot(
        data=data,
        x="target_torsion_velocity",
        y="pred_torsion_velocity",
        alpha=0.75,
    )
    finite = data[["target_torsion_velocity", "pred_torsion_velocity"]].replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()
    if not finite.empty:
        low = float(finite.min().min())
        high = float(finite.max().max())
        plt.plot([low, high], [low, high], color="black", linewidth=1, linestyle="--")
    plt.xlabel("target_torsion_velocity")
    plt.ylabel("pred_torsion_velocity")
    _savefig(output_dir / "fig_target_vs_pred_torsion_velocity.png")


def plot_rotatable_count_vs_torsion(molecule_df: pd.DataFrame, output_dir: Path) -> None:
    plot_rotatable_count_vs_metric(
        molecule_df,
        output_dir,
        metric="mean_abs_torsion_velocity",
        output_name="fig_num_rotatable_bonds_vs_mean_abs_torsion_velocity.png",
    )


def plot_rotatable_count_vs_metric(
    molecule_df: pd.DataFrame,
    output_dir: Path,
    metric: str,
    output_name: str,
    x_metric: str = "num_rotatable_bonds",
) -> None:
    if metric not in molecule_df.columns or x_metric not in molecule_df.columns:
        return
    data = _apply_source_labels(molecule_df.dropna(subset=[x_metric, metric]))
    if data.empty:
        return
    plt.figure(figsize=(7.2, 4.8))
    sns.scatterplot(
        data=data,
        x=x_metric,
        y=metric,
        hue="velocity_source_label",
        alpha=0.75,
    )
    for _, group in data.groupby("velocity_source_label"):
        if group[x_metric].nunique() > 1:
            sns.regplot(
                data=group,
                x=x_metric,
                y=metric,
                scatter=False,
                truncate=False,
            )
    plt.xlabel(x_metric)
    plt.ylabel(metric)
    plt.legend(title="")
    _savefig(output_dir / output_name)


def plot_error_count_vs_metric(
    molecule_error_df: pd.DataFrame,
    output_dir: Path,
    metric: str,
    output_name: str,
    x_metric: str = "matched_rotatable_bonds",
) -> None:
    if metric not in molecule_error_df.columns or x_metric not in molecule_error_df.columns:
        return
    data = molecule_error_df.dropna(subset=[x_metric, metric]).copy()
    if data.empty:
        return
    plt.figure(figsize=(7.2, 4.8))
    sns.scatterplot(data=data, x=x_metric, y=metric, alpha=0.75)
    if data[x_metric].nunique() > 1:
        sns.regplot(data=data, x=x_metric, y=metric, scatter=False, truncate=False)
    plt.xlabel(x_metric)
    plt.ylabel(metric)
    _savefig(output_dir / output_name)


def plot_target_vs_pred_molecule_metric(
    molecule_df: pd.DataFrame,
    output_dir: Path,
    metric: str,
    output_name: str,
) -> None:
    if metric not in molecule_df.columns:
        return
    pivot = molecule_df.pivot_table(
        index=["molecule_id", "smiles", "num_atoms", "num_rotatable_bonds"],
        columns="velocity_source",
        values=metric,
        aggfunc="mean",
    ).reset_index()
    if not {"target", "pred"}.issubset(pivot.columns):
        return

    data = pivot.dropna(subset=["target", "pred"]).copy()
    if data.empty:
        return
    plt.figure(figsize=(5.8, 5.2))
    sns.scatterplot(data=data, x="target", y="pred", alpha=0.75)
    finite = data[["target", "pred"]].replace([np.inf, -np.inf], np.nan).dropna()
    if not finite.empty:
        low = float(finite.min().min())
        high = float(finite.max().max())
        plt.plot([low, high], [low, high], color="black", linewidth=1, linestyle="--")
    plt.xlabel(f"target_{metric}")
    plt.ylabel(f"pred_{metric}")
    _savefig(output_dir / output_name)


def plot_relative_rotation_distribution(bond_df: pd.DataFrame, output_dir: Path) -> None:
    data = _apply_source_labels(bond_df.dropna(subset=["relative_rotation_norm"]))
    if data.empty:
        return
    plt.figure(figsize=(7.2, 4.8))
    sns.histplot(
        data=data,
        x="relative_rotation_norm",
        hue="velocity_source_label",
        bins=40,
        kde=True,
        element="step",
        stat="density",
        common_norm=False,
    )
    plt.xlabel("relative_rotation_norm")
    plt.ylabel("density")
    _savefig(output_dir / "fig_relative_rotation_norm_distribution.png")


def _test_row(
    test_name: str,
    group_a: str,
    group_b: str,
    values_a: pd.Series,
    values_b: pd.Series,
) -> Dict[str, object]:
    values_a = values_a.dropna()
    values_b = values_b.dropna()
    statistic = np.nan
    p_value = np.nan
    if mannwhitneyu is not None and len(values_a) > 0 and len(values_b) > 0:
        result = mannwhitneyu(values_a, values_b, alternative="two-sided")
        statistic = float(result.statistic)
        p_value = float(result.pvalue)

    mean_a = float(values_a.mean()) if len(values_a) else np.nan
    mean_b = float(values_b.mean()) if len(values_b) else np.nan
    if np.isnan(mean_a) or np.isnan(mean_b):
        direction = "not_available"
    elif mean_a > mean_b:
        direction = f"{group_a}_higher"
    elif mean_a < mean_b:
        direction = f"{group_a}_lower"
    else:
        direction = "equal_means"

    return {
        "test_name": test_name,
        "group_a": group_a,
        "group_b": group_b,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "median_a": float(values_a.median()) if len(values_a) else np.nan,
        "median_b": float(values_b.median()) if len(values_b) else np.nan,
        "statistic": statistic,
        "p_value": p_value,
        "effect_direction": direction,
    }


def _spearman_row(
    test_name: str,
    group_a: str,
    group_b: str,
    x: pd.Series,
    y: pd.Series,
) -> Dict[str, object]:
    data = pd.concat([x, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    statistic = np.nan
    p_value = np.nan
    if spearmanr is not None and len(data) > 1 and data.iloc[:, 0].nunique() > 1:
        result = spearmanr(data.iloc[:, 0], data.iloc[:, 1])
        statistic = float(result.statistic)
        p_value = float(result.pvalue)

    if np.isnan(statistic):
        direction = "not_available"
    elif statistic > 0:
        direction = "positive_correlation"
    elif statistic < 0:
        direction = "negative_correlation"
    else:
        direction = "zero_correlation"

    return {
        "test_name": test_name,
        "group_a": group_a,
        "group_b": group_b,
        "mean_a": float(data.iloc[:, 0].mean()) if len(data) else np.nan,
        "mean_b": float(data.iloc[:, 1].mean()) if len(data) else np.nan,
        "median_a": float(data.iloc[:, 0].median()) if len(data) else np.nan,
        "median_b": float(data.iloc[:, 1].median()) if len(data) else np.nan,
        "statistic": statistic,
        "p_value": p_value,
        "effect_direction": direction,
    }


def save_stat_tests(
    bond_df: pd.DataFrame,
    molecule_df: pd.DataFrame,
    molecule_error_df: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    rows = []
    rows.append(
        _test_row(
            "target vs pred abs_torsion_velocity",
            "target",
            "pred",
            bond_df.loc[bond_df["velocity_source"] == "target", "abs_torsion_velocity"],
            bond_df.loc[bond_df["velocity_source"] == "pred", "abs_torsion_velocity"],
        )
    )
    for metric in ("sum_abs_torsion_velocity", "max_abs_torsion_velocity"):
        if metric not in molecule_df.columns:
            continue
        rows.append(
            _test_row(
                f"target vs pred {metric}",
                "target",
                "pred",
                molecule_df.loc[molecule_df["velocity_source"] == "target", metric],
                molecule_df.loc[molecule_df["velocity_source"] == "pred", metric],
            )
        )

    for source, source_df in molecule_df.groupby("velocity_source"):
        low = source_df.loc[
            source_df["num_rotatable_bonds"] < FLEXIBILITY_THRESHOLD,
            "mean_abs_torsion_velocity",
        ]
        high = source_df.loc[
            source_df["num_rotatable_bonds"] >= FLEXIBILITY_THRESHOLD,
            "mean_abs_torsion_velocity",
        ]
        rows.append(
            _test_row(
                f"low vs high flexibility mean_abs_torsion_velocity ({source})",
                f"{source}_low_flex",
                f"{source}_high_flex",
                low,
                high,
            )
        )
        rows.append(
            _spearman_row(
                f"legacy num_rotatable_bonds vs mean_abs_torsion_velocity ({source})",
                "num_rotatable_bonds",
                f"{source}_mean_abs_torsion_velocity",
                source_df["num_rotatable_bonds"],
                source_df["mean_abs_torsion_velocity"],
            )
        )
        for metric in ("sum_abs_torsion_velocity", "max_abs_torsion_velocity"):
            if metric not in source_df.columns:
                continue
            rows.append(
                _spearman_row(
                    f"legacy num_rotatable_bonds vs {metric} ({source})",
                    "num_rotatable_bonds",
                    f"{source}_{metric}",
                    source_df["num_rotatable_bonds"],
                    source_df[metric],
                )
            )
        if "valid_rotatable_bonds" not in source_df.columns:
            continue
        for metric in (
            "mean_abs_torsion_velocity",
            "sum_abs_torsion_velocity",
            "max_abs_torsion_velocity",
            "top3_mean_abs_torsion_velocity",
        ):
            if metric not in source_df.columns:
                continue
            rows.append(
                _spearman_row(
                    f"valid_rotatable_bonds vs {metric} ({source})",
                    "valid_rotatable_bonds",
                    f"{source}_{metric}",
                    source_df["valid_rotatable_bonds"],
                    source_df[metric],
                )
            )

    if not molecule_error_df.empty and "matched_rotatable_bonds" in molecule_error_df.columns:
        for metric in (
            "mean_abs_torsion_error",
            "sum_abs_torsion_error",
            "max_abs_torsion_error",
            "mean_abs_relative_rotation_error",
            "sum_abs_relative_rotation_error",
        ):
            if metric not in molecule_error_df.columns:
                continue
            rows.append(
                _spearman_row(
                    f"matched_rotatable_bonds vs {metric}",
                    "matched_rotatable_bonds",
                    metric,
                    molecule_error_df["matched_rotatable_bonds"],
                    molecule_error_df[metric],
                )
            )

    stat_df = pd.DataFrame(rows)
    stat_df.to_csv(output_path, index=False)
    return stat_df


def _spearman_value(df: pd.DataFrame, x_col: str, y_col: str) -> float:
    if x_col not in df.columns or y_col not in df.columns:
        return np.nan
    data = df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 2 or data[x_col].nunique() < 2:
        return np.nan
    return float(data[x_col].corr(data[y_col], method="spearman"))


def _print_correlation_logs(
    molecule_df: pd.DataFrame,
    molecule_error_df: pd.DataFrame,
) -> None:
    print("correlation using old num_rotatable_bonds:")
    for source, source_df in molecule_df.groupby("velocity_source"):
        for metric in ("sum_abs_torsion_velocity", "max_abs_torsion_velocity"):
            print(
                f"  {source} num_rotatable_bonds vs {metric}: "
                f"{_spearman_value(source_df, 'num_rotatable_bonds', metric)}"
            )

    print("correlation using valid_rotatable_bonds:")
    for source, source_df in molecule_df.groupby("velocity_source"):
        for metric in (
            "sum_abs_torsion_velocity",
            "max_abs_torsion_velocity",
            "top3_mean_abs_torsion_velocity",
        ):
            print(
                f"  {source} valid_rotatable_bonds vs {metric}: "
                f"{_spearman_value(source_df, 'valid_rotatable_bonds', metric)}"
            )

    print("correlation using matched_rotatable_bonds:")
    for metric in ("sum_abs_torsion_error", "max_abs_torsion_error"):
        print(
            f"  matched_rotatable_bonds vs {metric}: "
            f"{_spearman_value(molecule_error_df, 'matched_rotatable_bonds', metric)}"
        )


def plot_all(input_dir: Path) -> None:
    bond_path = input_dir / "rotatable_bond_relative_motion.csv"
    molecule_path = input_dir / "rotatable_motion_by_molecule.csv"
    errors_path = input_dir / "rotatable_motion_target_pred_errors.csv"
    molecule_errors_path = input_dir / "rotatable_motion_error_by_molecule.csv"
    if not bond_path.exists():
        raise FileNotFoundError(f"Missing rotatable bond CSV: {bond_path}")
    if not molecule_path.exists():
        raise FileNotFoundError(f"Missing molecule CSV: {molecule_path}")
    if not errors_path.exists():
        raise FileNotFoundError(f"Missing target/pred error CSV: {errors_path}")

    raw_bond_df = _ensure_count_aliases(pd.read_csv(bond_path))
    bond_df = _valid_bond_rows(raw_bond_df)
    molecule_df = _add_candidate_counts(pd.read_csv(molecule_path), raw_bond_df)
    errors_df = _add_candidate_counts(pd.read_csv(errors_path), raw_bond_df)
    if molecule_errors_path.exists():
        molecule_error_df = _add_candidate_counts(
            pd.read_csv(molecule_errors_path),
            raw_bond_df,
        )
    else:
        molecule_error_df = _derive_molecule_error_df(errors_df, raw_bond_df)
    figures_dir = input_dir / "figures"

    print(f"total rotatable bond/source rows before filtering: {len(raw_bond_df)}")
    print(f"valid rotatable bond/source rows after filtering: {len(bond_df)}")
    if not raw_bond_df.empty:
        print("fit_status_a counts:")
        print(raw_bond_df["fit_status_a"].value_counts(dropna=False).to_string())
        print("fit_status_b counts:")
        print(raw_bond_df["fit_status_b"].value_counts(dropna=False).to_string())
    if not molecule_error_df.empty and "matched_rotatable_bonds" in molecule_error_df.columns:
        old_count_col = (
            "num_rotatable_bonds"
            if "num_rotatable_bonds" in molecule_error_df.columns
            else "rdkit_num_rotatable_bonds"
        )
        bad_count = int(
            (
                molecule_error_df["matched_rotatable_bonds"]
                > molecule_error_df[old_count_col]
            ).sum()
        )
        print(f"num rows where matched_rotatable_bonds > num_rotatable_bonds: {bad_count}")
    _print_correlation_logs(molecule_df, molecule_error_df)

    sns.set_theme(style="whitegrid", context="paper")
    plot_torsion_distribution(bond_df, figures_dir)
    plot_abs_torsion_distribution(bond_df, figures_dir)
    plot_target_vs_pred(errors_df, figures_dir)
    plot_rotatable_count_vs_torsion(molecule_df, figures_dir)
    plot_rotatable_count_vs_metric(
        molecule_df,
        figures_dir,
        metric="sum_abs_torsion_velocity",
        output_name="fig_num_rotatable_bonds_vs_sum_abs_torsion_velocity.png",
    )
    plot_rotatable_count_vs_metric(
        molecule_df,
        figures_dir,
        metric="max_abs_torsion_velocity",
        output_name="fig_num_rotatable_bonds_vs_max_abs_torsion_velocity.png",
    )
    plot_rotatable_count_vs_metric(
        molecule_df,
        figures_dir,
        metric="sum_abs_torsion_velocity",
        output_name="fig_valid_rotatable_bonds_vs_sum_abs_torsion_velocity.png",
        x_metric="valid_rotatable_bonds",
    )
    plot_rotatable_count_vs_metric(
        molecule_df,
        figures_dir,
        metric="max_abs_torsion_velocity",
        output_name="fig_valid_rotatable_bonds_vs_max_abs_torsion_velocity.png",
        x_metric="valid_rotatable_bonds",
    )
    plot_rotatable_count_vs_metric(
        molecule_df,
        figures_dir,
        metric="top3_mean_abs_torsion_velocity",
        output_name="fig_valid_rotatable_bonds_vs_top3_mean_abs_torsion_velocity.png",
        x_metric="valid_rotatable_bonds",
    )
    plot_error_count_vs_metric(
        molecule_error_df,
        figures_dir,
        metric="sum_abs_torsion_error",
        output_name="fig_matched_rotatable_bonds_vs_sum_abs_torsion_error.png",
    )
    plot_error_count_vs_metric(
        molecule_error_df,
        figures_dir,
        metric="max_abs_torsion_error",
        output_name="fig_matched_rotatable_bonds_vs_max_abs_torsion_error.png",
    )
    plot_target_vs_pred_molecule_metric(
        molecule_df,
        figures_dir,
        metric="sum_abs_torsion_velocity",
        output_name="fig_target_vs_pred_sum_abs_torsion_velocity.png",
    )
    plot_target_vs_pred_molecule_metric(
        molecule_df,
        figures_dir,
        metric="max_abs_torsion_velocity",
        output_name="fig_target_vs_pred_max_abs_torsion_velocity.png",
    )
    plot_relative_rotation_distribution(bond_df, figures_dir)
    save_stat_tests(
        bond_df,
        molecule_df,
        molecule_error_df,
        input_dir / "rotatable_motion_stat_tests.csv",
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        "-i",
        type=str,
        default="logs_rotatable_motion_first",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_all(Path(args.input_dir))
