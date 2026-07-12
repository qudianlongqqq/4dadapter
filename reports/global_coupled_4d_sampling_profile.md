# Global Coupled 4D sampling profile

- Profiled molecules: 5
- Profile source: synthetic_5_molecule_numeric_regression
- Dataset/manifest load: 0.000000 s
- Mean molecule: 0.017280 s
- Mean refinement step: 0.001684 s
- Devices: `{"backbone": "cpu", "gram": "cpu", "jacobian": "cpu", "solver": "cpu"}`
- Solver backends: `{"svd_fallback": 50}`
- Peak GPU memory: 0 bytes
- Threads: `{"source": "synthetic benchmark because formal artifacts are absent", "torch_num_interop_threads": 8, "torch_num_threads": 4}`
- Partial payload: `E:\3dconformergenerationcode\4dadapter\diagnostics\global_coupled_4d\checkpoint_sweep_5k\step1000_alpha02\partial_samples.pt`
- RDKit rollout operations: 0

## Mean component time per step

| Component | Seconds |
| --- | ---: |
| cartesian_projection_time | 0.000116 |
| cholesky_time | 0.000000 |
| egnn_forward_time | 0.000638 |
| fragment_pool_time | 0.000023 |
| gram_matrix_time | 0.000022 |
| internal_mapping_time | 0.000011 |
| jacobian_construction_time | 0.000185 |
| joint_head_time | 0.000131 |
| local_frame_time | 0.000172 |
| lstsq_time | 0.000000 |
| rdkit_time | 0.000000 |
| solve_projection_time | 0.000245 |
| solve_time | 0.000000 |
| svd_time | 0.000053 |
| topology_time | 0.000000 |
| torch_linalg_lstsq_time | 0.000000 |
| torch_linalg_solve_time | 0.000000 |
| torch_linalg_svd_time | 0.000053 |

## Coordinate-independent preparation per molecule

| Component | Seconds |
| --- | ---: |
| cache_key_time | 0.000057 |
| device_transfer_time | 0.000005 |
| mask_construction_time | 0.000045 |
| topology_construction_time | 0.000036 |
| total_preparation_time | 0.000350 |

## Per molecule

| Sample | Total s | Mean step s | CPU→device s | Device→CPU s |
| --- | ---: | ---: | ---: | ---: |
| synthetic-1 | 0.024564 | 0.002331 | 0.000000 | 0.000000 |
| synthetic-2 | 0.016790 | 0.001649 | 0.000000 | 0.000000 |
| synthetic-3 | 0.015201 | 0.001498 | 0.000000 | 0.000000 |
| synthetic-4 | 0.014897 | 0.001469 | 0.000000 | 0.000000 |
| synthetic-5 | 0.014949 | 0.001475 | 0.000000 | 0.000000 |
