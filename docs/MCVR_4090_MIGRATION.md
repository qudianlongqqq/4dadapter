# MCVR Linux 4090 Migration

Windows RTX 5080 is the current MCVR numerical reference. Linux dual RTX 4090 is the environment for subsequent formal MCVR seeds. Linux RTX 5090 remains only the historical Global4D/formal-large environment.

OS, driver and GPU model may differ. Python major/minor and the configured scientific package compatibility policy must pass, while identical Git commit, data SHA, D1-B checkpoint SHA and two-record H0 behavior are mandatory. An import-only check is insufficient.

Do not copy the Windows Conda environment or use its `pip freeze` as the Linux installer. Create `etflow-4090`, compare environment fingerprints, verify assets, reproduce the two-record smoke, and run the unified preflight. After acceptance, export Linux-specific `conda env export --from-history`, `conda list --explicit`, and `pip freeze` locks. All later formal seeds must run in the accepted 4090 environment.
