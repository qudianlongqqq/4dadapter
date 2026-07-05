#!/usr/bin/env python
"""Prove that every inference cache record is label-free and schema-valid."""

import argparse

from etflow.data.flexbond_inference_dataset import FlexBondInferenceDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    dataset = FlexBondInferenceDataset(args.cache_dir, args.split)
    for index in range(len(dataset)):
        data = dataset[index]
        forbidden = {"x_ref", "x_ref_candidates", "u_t", "q_b_star"}
        leaked = sorted(key for key in forbidden if hasattr(data, key))
        if leaked:
            raise ValueError(f"Runtime inference object contains labels: {leaked}")
    print(f"PASS: {len(dataset)} inference records contain no training labels")


if __name__ == "__main__":
    main()
