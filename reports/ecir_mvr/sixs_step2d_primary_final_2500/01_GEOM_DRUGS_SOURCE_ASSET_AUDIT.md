# GEOM-DRUGS source asset audit

```text
SOURCE_DATASET = GEOM-DRUGS-WITH-TORSIONAL-DIFFUSION-NATIVE-SPLITS
ARCHIVE_BYTES = 25308038362
ARCHIVE_SHA256 = 79812ffefcb51abc5ceca04ce1b67d1af385e1ffe7ffb3c74a5e9fa2d4bb69cd
ARCHIVE_MEMBERS = 34071
ARCHIVE_REGULAR_FILES = 34068
ARCHIVE_REGULAR_FILE_BYTES = 15158917373
STANDARDIZED_PICKLE_CHUNKS = 300
RAW_PICKLE_MEMBERS = 33764
NATIVE_SPLIT_COUNTS = {"test": 30433, "train": 243473, "val": 30433}
NATIVE_SPLIT_INDEX_UNIVERSE = 304339
STANDARDIZED_ENTRIES = 283424
SOURCE_UNIVERSE_N_MOLECULES = 282953
RAW_LISTING_NATIVE_LINKAGE = SUBSET_ONLY__NO_GLOBAL_NATIVE_INDEX_LINKAGE
NATIVE_LINKAGE_OUTSIDE_COMPLETE_RAW_INDEX = UNRECOVERABLE_FROM_DRUGS_TAR_GZ
```

`split.npy` is a three-array native index partition. Standardized chunks are
Python dictionaries keyed by source stems; each molecule stores a SMILES,
charge, ensemble metadata, and a list of conformer dictionaries containing an
RDKit molecule. Coordinates and energy/outcome fields were not materialized in
any final manifest. The released archive has only a raw filename subset, not
the complete 304,339-name raw index and not a continuous prefix. Standardized
identities therefore receive no guessed native index. This is a provenance-
metadata limitation, not an eligibility exclusion: the frozen rule selects
from the whole eligible unused GEOM-DRUGS universe.
