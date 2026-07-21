from etflow.ecir.external_refinement_baselines import build_xtb_command
from tests.external_refinement_test_utils import config


def test_xtb_workdirs_are_command_isolated(tmp_path):
    cfg = config()["gfn2_xtb"]
    first = build_xtb_command(input_name="input.xyz", workdir=tmp_path / "a", total_charge=0, unpaired_electrons=0, config=cfg)
    second = build_xtb_command(input_name="input.xyz", workdir=tmp_path / "b", total_charge=0, unpaired_electrons=0, config=cfg)
    assert first[first.index("--cd") + 1] != second[second.index("--cd") + 1]
