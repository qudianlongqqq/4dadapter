from pathlib import Path

from etflow.ecir.v8_validation_cache import file_sha256
from tests.external_refinement_test_utils import config


def test_xtb_version_and_binary_sha_are_frozen():
    cfg = config()["gfn2_xtb"]
    assert cfg["xtb_version"] == "6.7.1"
    assert file_sha256(Path(cfg["xtb_executable"])) == cfg["binary_sha256"]
