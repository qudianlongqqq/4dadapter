import json
import subprocess
import sys

from etflow.commons.run_state import atomic_write_json


def test_atomic_state_write_leaves_no_partial_file(tmp_path):
    path=tmp_path/"run_state.json";atomic_write_json(path,{"status":"completed"})
    assert json.loads(path.read_text())["status"]=="completed"
    assert not list(tmp_path.glob("*.tmp*"))


def test_direct_script_help_needs_no_pythonpath():
    result=subprocess.run([sys.executable,"scripts/report_gated_kinematic_progress.py","--help"],
        capture_output=True,text=True,check=False)
    assert result.returncode==0,result.stderr


def test_progress_report_handles_partial_failed_and_empty_csv(tmp_path):
    root=tmp_path;run=root/"logs_gated_kinematic"/"failed_run";run.mkdir(parents=True)
    atomic_write_json(run/"run_state.json",{"status":"failed"});(run/"partial_progress.json").write_text("{}")
    (run/"summary.csv").write_text("",encoding="utf-8")
    output=root/"reports"
    result=subprocess.run([sys.executable,"scripts/report_gated_kinematic_progress.py",
        "--root",str(root),"--output_dir",str(output)],capture_output=True,text=True,check=False)
    assert result.returncode==0,result.stderr
    report=json.loads((output/"gated_kinematic_latest.json").read_text())
    assert report["counts"]["failed"]==1 and report["partial_files"]
