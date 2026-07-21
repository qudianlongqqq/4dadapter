# MCVR External Refinement Environment Audit

Status: `MCVR_EXTERNAL_REFINEMENT_ENVIRONMENT_READY`

RDKit 2026.03.4 provides the required MMFF API and explicit MMFF94s support. Frozen validation coverage is 9,998/10,000, with the two unsupported records retained under bitwise Source fallback.

Official stable xTB 6.7.1 is installed outside the repository at `E:\tools\xtb\6.7.1-linux-x86_64\bin\xtb` and invoked through Ubuntu-22.04 WSL2 by absolute path. Executable SHA256 is `debf27a9e0fa4bfb5ca75aafe4b90d8211f08ec2f4a482f375a4987212eaa12a`; archive SHA256 is `62a8d18778286e815292ee53D76CE447DAF460A4DEA3782C0F25CBAC7019B5DF` (case-insensitive hexadecimal).

The Python environment, PATH, PyTorch, and RDKit were not modified. Frozen worker configuration is two processes with one numerical-library thread per process.

All isolation counters remain zero; formal test and frozen holdout assets were not opened.

The full audit is in `docs/MCVR_EXTERNAL_REFINEMENT_ENVIRONMENT_AUDIT.md`.
