# Raw filename to native split linkage audit

This audit used the already extracted archive inventory, `split.npy`, the
existing source-universe cache, and the frozen official preprocessing code. It
did not rescan the 300 standardized pickle chunks.

The official pipeline defines native indices as positions in the **sorted
complete raw filename list**. This is explicit in
`code_snapshot/utils/dataset.py` (`sorted(glob(...))`, followed by indexing with
`split.npy`) and in `code_snapshot/standardize_confs.py` (the same sorted list,
sliced into 1,000-file worker blocks). The archive has 304,339 native indices
but only 33,764 raw pickle filenames. Those 33,764 names are unique, but the
subset is neither the complete list nor a continuous prefix. Missing filenames
can sort between surviving names, so assigning indices by subset order would
be nondeterministic guessing.

The standardized source cache contains 282,953 canonical identities; 31,412
have a surviving raw archive member. That source/filename association does not
recover the missing position in the complete sorted list. `test_mols.pkl` is a
separate 1,000-molecule test object and does not supply the raw filename index.

```text
RAW_TO_NATIVE_INDEX_RECOVERED = NO
RAW_TO_NATIVE_SPLIT_RECOVERED = NO
N_RAW_RECORDS = 33764
N_MAPPED = 0
N_UNMAPPED = 33764
N_AMBIGUOUS = 0
RAW_NATIVE_MAPPING_STATUS = NOT_RECOVERED__COMPLETE_SORTED_304339_FILENAME_INDEX_MISSING
```

This limitation affects native-split provenance metadata. It does not define
eligibility under the frozen prospective selection rule.
