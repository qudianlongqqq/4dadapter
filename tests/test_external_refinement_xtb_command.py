from etflow.ecir.external_refinement_baselines import build_xtb_command
from tests.external_refinement_test_utils import config


def test_xtb_command_freezes_gfn2_charge_uhf_and_opt(tmp_path):
    cfg = config()["gfn2_xtb"]
    command = build_xtb_command(input_name="input.xyz", workdir=tmp_path, total_charge=-1, unpaired_electrons=2, config=cfg)
    joined = " ".join(command)
    assert "--gfn 2" in joined and "--opt normal" in joined and "--cycles 250" in joined
    assert "--chrg -1" in joined and "--uhf 2" in joined
    assert "--alpb" not in joined and "--gbsa" not in joined
