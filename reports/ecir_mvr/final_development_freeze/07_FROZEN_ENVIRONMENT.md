# Frozen environment

Environment inspection was read-only. No package was installed, removed or upgraded.

## CUDA training/inference environment

```text
OS = Windows-10-10.0.26200-SP0
PYTHON = 3.11.15 (conda-forge; MSC v.1944; 64 bit)
PYTHON_EXECUTABLE = E:/miniconda/envs/etflow-5080-v2/python.exe
PYTORCH = 2.11.0+cu128
PYTORCH_CUDA_RUNTIME = 12.8
CUDA_AVAILABLE_AT_FREEZE = TRUE
CUDA_DEVICE_COUNT = 1
GPU = NVIDIA GeForce RTX 5080
NVIDIA_DRIVER = 610.88
GPU_MEMORY_TOTAL = 16303 MiB
RDKIT = 2026.03.4
NUMPY = 1.26.4
SCIPY = 1.17.1
PANDAS = 3.0.3
PYARROW = 25.0.0
```

The canonicalized live `pip freeze` contained 196 nonempty lines. Its LF-normalized UTF-8 SHA256 at freeze time was `647a12b0cf3fbb4d2d09e76ff1442c9d65d86b07d1a0d897f2cc88a83aadb905`.

## Stable external V3D/PoseBusters environment

```text
OS = Windows-10-10.0.26200-SP0
PYTHON = 3.11.15 (conda-forge; MSC v.1944; 64 bit)
PYTHON_EXECUTABLE = E:/miniconda/envs/external-validity/python.exe
RDKIT = 2025.09.6
POSEBUSTERS = 0.6.5
NUMPY = 2.0.2
SCIPY = 1.14.1
PANDAS = 2.2.3
PYARROW = 20.0.0
GENBENCH3D_COMMIT = 0926bc6614509aa10ccf6f69da0405d4be6af6b3
```

The canonicalized live `pip freeze` contained 130 nonempty lines. Its LF-normalized UTF-8 SHA256 at freeze time was `7d9ba5802e8bb79ac600af87c415762bf6c27c6795eec61f96d79536e0d27eeb`.

## External physical evaluator

```text
XTB = 6.7.1
XTB_EXECUTABLE = E:/tools/xtb/6.7.1-linux-x86_64/bin/xtb
XTB_EXECUTABLE_SHA256 = debf27a9e0fa4bfb5ca75aafe4b90d8211f08ec2f4a482f375a4987212eaa12a
METHOD = GFN2-xTB
TASK = SINGLE-POINT
GEOMETRY_OPTIMIZATION = NO
```

`ENVIRONMENT_LOCK_MINIMAL.txt` is the compact human-readable lock. Exact code/config/evaluator binary hashes are in `06_FINAL_RELEASE_MANIFEST.csv`.

```text
ENVIRONMENT_FROZEN = YES
PACKAGE_MUTATION_PERFORMED = NO
```
