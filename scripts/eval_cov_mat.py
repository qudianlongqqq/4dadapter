import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from loguru import logger as log

import wandb
from etflow.commons import load_pkl
from etflow.commons.covmat import CovMatEvaluator, print_covmat_results


KEY_THRESHOLDS = (0.8, 1.0, 1.25, 1.5, 2.0)


def _json_value(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _save_metrics(path, cov_df, matching_metrics):
    output_dir = Path(path).expanduser().resolve().parent
    columns = list(cov_df.columns)
    data = [
        {column: _json_value(value) for column, value in row.items()}
        for row in cov_df.to_dict(orient="records")
    ]

    metrics_csv = output_dir / "eval_cov_mat_metrics.csv"
    metrics_json = output_dir / "eval_cov_mat_metrics.json"
    metrics_md = output_dir / "eval_cov_mat_metrics.md"
    cov_df.to_csv(metrics_csv, index=False)

    key_thresholds = {}
    for threshold in KEY_THRESHOLDS:
        matching_rows = [
            row
            for row in data
            if np.isclose(float(row["Threshold"]), threshold, atol=1e-8)
        ]
        if matching_rows:
            key_thresholds[str(threshold)] = matching_rows[0]

    with metrics_json.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "input_path": str(Path(path)),
                "num_rows": len(data),
                "columns": columns,
                "data": data,
                "key_thresholds": key_thresholds,
            },
            handle,
            indent=2,
            allow_nan=False,
        )

    with metrics_md.open("w", encoding="utf-8") as handle:
        handle.write("# Coverage Metrics\n\n")
        handle.write(f"Input: `{path}`\n\n")
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in data:
            handle.write(
                "| " + " | ".join(str(row[column]) for column in columns) + " |\n"
            )

    scalar_data = {
        key: _json_value(value) for key, value in matching_metrics.items()
    }
    scalars_json = output_dir / "eval_cov_mat_scalars.json"
    scalars_csv = output_dir / "eval_cov_mat_scalars.csv"
    with scalars_json.open("w", encoding="utf-8") as handle:
        json.dump(
            {"input_path": str(Path(path)), "metrics": scalar_data},
            handle,
            indent=2,
            allow_nan=False,
        )
    with scalars_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(scalar_data.items())

    print("Coverage metrics saved:")
    print(f"- {metrics_csv}")
    print(f"- {metrics_json}")
    print(f"- {metrics_md}")
    print(f"- {scalars_csv}")
    print(f"- {scalars_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", "-p", type=str, help="Path to the data file", required=True
    )
    parser.add_argument(
        "--num_workers", "-n", type=int, default=1, help="Number of workers"
    )
    parser.add_argument(
        "--use_alignmol",
        "-a",
        action="store_true",
        default=False,
        help="Use alignmol for matching",
    )
    args = parser.parse_args()

    path = args.path
    os.path.exists(path), f"Path {path} does not exist"
    packed_data_list = load_pkl(path)

    # log on weight and biases
    wandb.init(
        project="Energy-Aware-MCG",
        entity="doms-lab",
        name=f"Evaluation Coverage and Matching: Path {path}",
    )

    wandb.run.log({"Path": path})

    use_alignmol = args.use_alignmol
    wandb.run.log({"Use Alignmol": use_alignmol})

    num_workers = args.num_workers
    log.info(f"Using {num_workers} workers for evaluation...")
    evaluator = CovMatEvaluator(num_workers=num_workers, use_alignmol=args.use_alignmol)
    log.info("Evaluation Started...")
    results, rmsd_results = evaluator(packed_data_list)
    log.info("Evaluation finished...")

    # get dataframe of results
    cov_df, matching_metrics = print_covmat_results(results)

    # log as table
    table = wandb.Table(dataframe=cov_df)
    wandb.run.log({"Coverage Metrics": table})

    # log matching metrics
    wandb.run.log(matching_metrics)

    _save_metrics(path, cov_df, matching_metrics)
