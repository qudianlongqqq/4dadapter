# V2 data-manifest audit

The existing molecule-disjoint formal-large training validation partition was used only as legal development data and deterministically re-frozen before v2 training.

- V2_VAL: 2,500 molecules / 5,000 real Source records; SHA256 `ca9a28d63720882828c202fd8ed7e9c219586cd5566a78df89ddba45d5def011`
- V2_DEV_TEST: 2,500 molecules / 5,000 real Source records; SHA256 `aca681f98702a36d4828c1a4299f7f50b424f8e934e1b276b68aa3952ef80d8d`
- V2_VAL/V2_DEV_TEST overlap: 0
- overlap with v2 TRAIN: 0 (inherited frozen formal-large TRAIN/VAL separation)
- V2_DEV_TEST outcomes read before training: no
- formal test / frozen holdout records read: 0 / 0
