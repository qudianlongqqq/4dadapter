#!/usr/bin/env python
"""Restart-safe post-cross-upstream matched-ablation supervisor.

The PowerShell launcher owns the single blocking Wait-Process.  This worker is
entered only once that wait has returned and never polls another process.
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, sys, time, traceback
from pathlib import Path
from typing import Any
import pandas as pd
import torch

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap
ROOT=bootstrap()
REPORT=ROOT/"reports/ecir_mvr/sixs_final_matched_ablation"
ARTIFACT=ROOT/"artifacts/ecir_mvr/sixs_final_matched_ablation"
ASSET=Path(r"E:\3dconformergenerationcode\dataset\sixs_final_matched_ablation_v1")
CROSS=ROOT/"reports/ecir_mvr/sixs_final_cross_upstream_unrestricted"
PRIMARY_REPORT=ROOT/"reports/ecir_mvr/sixs_primary_final_evaluation"
PRIMARY_ASSET=Path(r"E:\3dconformergenerationcode\dataset\sixs_primary_final_evaluation_v1")
CUDA=Path(r"E:\miniconda\envs\etflow-5080-v2\python.exe")
PRIMARY_PROTOCOL=PRIMARY_REPORT/"00_FROZEN_FINAL_EVALUATION_PROTOCOL.json"
PRIMARY_MANIFEST=ROOT/"reports/ecir_mvr/sixs_step2d_primary_final_2500/04_PRIMARY_FINAL_2500_MANIFEST.json"
SOURCE_MANIFEST=Path(r"E:\3dconformergenerationcode\dataset\prospective_final_etflow_nfe10_seed42\SOURCE_RECORD_MANIFEST.jsonl")
SOURCE_FREEZE=ROOT/"reports/ecir_mvr/sixs_step3a_etflow_source_generation/07_SOURCE_ASSET_FREEZE.json"
FIXED_TAU_SOURCE=ROOT/"reports/ecir_mvr/sixs_j1r1_joint_magnitude_interaction_seed307/FIXED_MOVEMENT_REPRODUCTION.json"
STATUS=REPORT/"STATUS.json"; PROTOCOL=REPORT/"00_PROTOCOL_FREEZE.json"
SEEDS=(307,331,353); VARIANTS=("reliability_off","equal_ba")

def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""): h.update(b)
 return h.hexdigest()
def canonical(v): return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()
def atomic_json(path,value):
 path.parent.mkdir(parents=True,exist_ok=True); t=path.with_name(f".{path.name}.{os.getpid()}.tmp"); t.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False,default=str)+"\n",encoding="utf-8"); os.replace(t,path)
def read_json(path):
 # Windows PowerShell 5.1's `Set-Content -Encoding UTF8` emits a BOM.  Accept
 # both forms at orchestration boundaries; scientific artifacts remain exact.
 return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def update(stage,state="RUNNING",**extra):
 try:
  old=read_json(STATUS) if STATUS.is_file() else {}
 except Exception as status_error:
  # Fail closed but never allow a damaged status document to prevent failure
  # accounting itself from being written.
  old={"status_recovery_error":f"{type(status_error).__name__}: {status_error}"}
 history=old.get("stage_history",{})
 entry=history.get(stage,{"START_TIME":time.time()}); entry.update({"STATUS":state,"UPDATED_AT":time.time(),**{k:v for k,v in extra.items() if k in {"END_TIME","INPUT_SHA","OUTPUT_SHA","EXIT_CODE"}}}); history[stage]=entry
 # PowerShell's ConvertFrom-Json treats keys case-insensitively.  Normalize
 # wrapper keys before merging stage metadata so EXIT_CODE/exit_code cannot
 # coexist in an otherwise valid status document.
 for key in list(old):
  if key.lower() in {"exit_code", "exception"}: old.pop(key)
 old.update({"schema_version":"sixs-post-cross-upstream-ablation-pipeline-v1","status":state,"stage":stage,"pipeline_pid":os.getpid(),"updated_at_epoch":time.time(),"waiting_for_pid":None,"cross_upstream_status":extra.pop("cross_upstream_status",old.get("cross_upstream_status","PENDING_VALIDATION")),"next_automatic_stage":extra.pop("next_automatic_stage",None),"output_root":str(REPORT),"no_repeated_polling":True,"no_busy_waiting":True,"no_continuous_log_tail":True,"stage_history":history,**extra}); atomic_json(STATUS,old)
def run(name,cmd,marker=None):
 if marker and marker.is_file():
  try:
   value=json.loads(marker.read_text(encoding="utf-8")) if marker.suffix==".json" else {}
   if value.get("status") in {"PASS","COMPLETE"}: update(name,"PASS",END_TIME=time.time(),EXIT_CODE=0,reused=True); return
  except Exception: pass
 logs=REPORT/"stage_logs"; logs.mkdir(parents=True,exist_ok=True); update(name,"RUNNING",next_automatic_stage=name)
 with (logs/f"{name}.stdout.log").open("a",encoding="utf-8") as out,(logs/f"{name}.stderr.log").open("a",encoding="utf-8") as err: result=subprocess.run([str(x) for x in cmd],cwd=ROOT,stdout=out,stderr=err)
 if result.returncode: update(name,"FAIL",END_TIME=time.time(),EXIT_CODE=result.returncode); raise RuntimeError(f"{name} failed ({result.returncode})")
 if marker and not marker.is_file(): raise RuntimeError(f"{name} marker missing: {marker}")
 update(name,"PASS",END_TIME=time.time(),EXIT_CODE=0,OUTPUT_SHA=sha(marker) if marker and marker.is_file() else None)

def validate_cross():
 final=CROSS/"FINAL_STATUS.json"
 if not final.is_file(): run("STAGE_1_REPAIR_COMBINED_SUMMARY",[CUDA,ROOT/"scripts/run_sixs_final_cross_upstream_unrestricted.py","combined"],final)
 required=[CROSS/"avgflow/RESULT.json",CROSS/"ditmc/RESULT.json",CROSS/"avgflow/XTB_FINAL_STATUS.json",CROSS/"ditmc/XTB_FINAL_STATUS.json",final,CROSS/"CROSS_UPSTREAM_SUMMARY.csv",CROSS/"CROSS_UPSTREAM_CONCLUSION.md"]
 for path in required:
  if not path.is_file(): raise RuntimeError(f"cross-upstream authoritative artifact missing: {path}")
 for path in required[:5]:
  if json.loads(path.read_text(encoding="utf-8")).get("status")!="PASS": raise RuntimeError(f"cross-upstream artifact not PASS: {path}")
 for branch in ("avgflow","ditmc"):
  result=json.loads((CROSS/branch/"RESULT.json").read_text(encoding="utf-8")); seeds=sorted(int(v["seed"]) for v in result["per_seed"])
  if seeds!=list(SEEDS) or any(int(v["records"])!=10000 or float(v["finite_coordinate_rate"])!=1 for v in result["per_seed"]): raise RuntimeError(f"cross-upstream seed/finite validation failed: {branch}")
 update("STAGE_1_VALIDATE_CROSS_UPSTREAM","PASS",END_TIME=time.time(),EXIT_CODE=0,cross_upstream_status="PASS",next_automatic_stage="STAGE_2_FREEZE_CROSS_UPSTREAM")

def freeze_cross():
 target=CROSS/"FINAL_CROSS_UPSTREAM_FREEZE.json"
 if target.is_file():
  existing=json.loads(target.read_text(encoding="utf-8"))
  if existing.get("status")!="PASS" or existing.get("CROSS_UPSTREAM_FROZEN")!="YES": raise RuntimeError("existing cross-upstream freeze is invalid")
  update("STAGE_2_FREEZE_CROSS_UPSTREAM","PASS",END_TIME=time.time(),EXIT_CODE=0,OUTPUT_SHA=sha(target),cross_upstream_status="PASS",next_automatic_stage="STAGE_3_PREPARE_FINAL_ABLATION",reused=True)
  return
 files=[CROSS/"00_FINAL_UNRESTRICTED_CHECKPOINT_MANIFEST.json",CROSS/"avgflow/RESULT.json",CROSS/"ditmc/RESULT.json",CROSS/"avgflow/XTB_FINAL_STATUS.json",CROSS/"ditmc/XTB_FINAL_STATUS.json",CROSS/"CROSS_UPSTREAM_SUMMARY.csv",CROSS/"CROSS_UPSTREAM_CONCLUSION.md",CROSS/"FINAL_STATUS.json"]
 dirty=subprocess.run(["git","status","--porcelain=v1"],cwd=ROOT,capture_output=True,text=True,check=True).stdout.splitlines()
 manifest=json.loads(files[0].read_text(encoding="utf-8")); checkpoints={}
 def walk(v):
  if isinstance(v,dict):
   for k,x in v.items():
    if "checkpoint" in k.lower() and isinstance(x,str) and Path(x).is_file(): checkpoints[x]=sha(x)
    walk(x)
  elif isinstance(v,list):
   for x in v: walk(x)
 walk(manifest)
 payload={"schema_version":"sixs-final-cross-upstream-freeze-v1","status":"PASS","frozen_at_epoch":time.time(),"seeds":list(SEEDS),"authoritative_files":{str(p.relative_to(ROOT)):sha(p) for p in files},"checkpoint_sha256":checkpoints,"raw_and_sixs_metrics":{"avgflow":json.loads(files[1].read_text(encoding="utf-8")),"ditmc":json.loads(files[2].read_text(encoding="utf-8"))},"xtb_completion":{"avgflow":"PASS","ditmc":"PASS"},"combined_summary_sha256":sha(CROSS/"CROSS_UPSTREAM_SUMMARY.csv"),"git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"dirty_state":dirty,"dirty_state_sha256":canonical(dirty),"CROSS_UPSTREAM_MODEL_SELECTION":"NONE","CROSS_UPSTREAM_TUNING":"NONE","ZERO_SHOT":"YES","CROSS_UPSTREAM_FROZEN":"YES"}
 if target.is_file() and json.loads(target.read_text(encoding="utf-8"))!=payload: raise RuntimeError("cross-upstream freeze already exists with different content")
 atomic_json(target,payload); update("STAGE_2_FREEZE_CROSS_UPSTREAM","PASS",END_TIME=time.time(),EXIT_CODE=0,OUTPUT_SHA=sha(target),cross_upstream_status="PASS",next_automatic_stage="STAGE_3_PREPARE_FINAL_ABLATION")

def prepare_protocol():
 if PROTOCOL.is_file(): return json.loads(PROTOCOL.read_text(encoding="utf-8"))
 primary=json.loads(PRIMARY_PROTOCOL.read_text(encoding="utf-8")); base={int(m["seed"]):m for m in primary["model_methods"] if m["formulation"]=="Unrestricted"}
 source=pd.read_parquet(PRIMARY_ASSET/"methods/source/PER_RECORD.parquet",columns=["record_id","molecule_id"])
 molecules=sorted(source.molecule_id.astype(str).unique(),key=lambda x:hashlib.sha256(("SIXS_XTB_ABLATION_V1|"+x).encode()).hexdigest())[:750]; chosen=set(molecules); record_ids=source[source.molecule_id.astype(str).isin(chosen)].record_id.astype(str).tolist()
 methods=[]
 for variant in VARIANTS:
  for seed in SEEDS:
   checkpoint=REPORT/f"{variant}_seed{seed}/FINAL_CHECKPOINT.pt"; methods.append({"id":f"{variant}_seed{seed}","variant":variant,"formulation":"Unrestricted","seed":seed,"step":17500,"checkpoint":str(checkpoint)})
 for variant in ("fixed_sigma_action","bond_only","angle_only","fixed_tau"):
  for seed in SEEDS:
   methods.append({"id":f"{variant}_seed{seed}","variant":variant,"formulation":"Unrestricted","seed":seed,"step":17500,"checkpoint":str(Path(base[seed]["checkpoint"]).resolve()),"checkpoint_sha256":base[seed]["checkpoint_sha256"]})
 payload={"schema_version":"sixs-final-matched-ablation-protocol-v1","status":"PASS","frozen_before_new_ablation_outcome":True,"FINAL_PRIMARY_METHOD":"J1_R1_FULL_JOINT_ADAPTIVE_BA_UNRESTRICTED_MOVEMENT","seeds":list(SEEDS),"new_gpu_training_runs":6,"training_steps":17500,"no_dev_checkpoint_selection":True,"one_gpu_training_at_a_time":True,"matched_training":{"train_manifest":"UNCHANGED_FROM_FINAL_UNRESTRICTED","batch_ordering":"SEED_MATCHED","sampler_seed":"EQUAL_TO_RUN_SEED","optimizer":"UNCHANGED","scheduler":"UNCHANGED","beta_nll_beta":0.5},"variants":{"reliability_off":{"change":"Reliability output fixed to one; all else matched"},"equal_ba":{"change":"w_B=w_A=0.5; all else matched"},"fixed_sigma_action":{"retraining":False,"scope":"predictive sigma action weighting only"},"bond_only":{"retraining":False,"change":"mask Angle primitive action contribution to zero without renormalizing Bond"},"angle_only":{"retraining":False,"change":"mask Bond primitive action contribution to zero without renormalizing Angle"},"fixed_tau":{"retraining":False,"value_angstrom":0.004321542759325059,"rule":"reuse pre-existing formal fixed-movement reproduction","source":str(FIXED_TAU_SOURCE),"source_sha256":sha(FIXED_TAU_SOURCE)}},"cohort":{"molecules":2500,"records":5000,"records_per_molecule":2,"resampling_unit":"molecule"},"bindings":{"primary_protocol":str(PRIMARY_PROTOCOL),"primary_protocol_sha256":sha(PRIMARY_PROTOCOL),"primary_manifest":str(PRIMARY_MANIFEST),"primary_manifest_sha256":sha(PRIMARY_MANIFEST),"source_record_manifest":str(SOURCE_MANIFEST),"source_record_manifest_sha256":sha(SOURCE_MANIFEST),"source_asset_freeze":str(SOURCE_FREEZE),"source_asset_freeze_sha256":sha(SOURCE_FREEZE),"cross_upstream_freeze":str(CROSS/"FINAL_CROSS_UPSTREAM_FREEZE.json"),"cross_upstream_freeze_sha256":sha(CROSS/"FINAL_CROSS_UPSTREAM_FREEZE.json")},"model_methods":methods,"xtb_subset":{"selection_rule":"first 750 molecule identities after SHA256 rank of SIXS_XTB_ABLATION_V1|molecule_id","molecules":molecules,"record_ids":record_ids,"molecule_count":750,"record_count":1500,"selected_before_new_ablation_outcomes":True,"priority_variants":["reliability_off","equal_ba","fixed_tau"]},"xtb_single_point":primary["xtb_single_point"],"statistics":{"bootstrap_resamples":10000,"bootstrap_seed":20260904,"cluster":"molecule_id","record_iid_bootstrap":False},"guards":{"outcome_dependent_selection":False,"formal_outcome_read":False,"large_holdout_outcome_read":False}}
 atomic_json(PROTOCOL,payload)
 pd.DataFrame([{k:v for k,v in m.items() if k!="checkpoint_sha256"} for m in methods]).to_csv(REPORT/"01_VARIANT_MANIFEST.csv",index=False)
 return payload

def make_config(variant,seed):
 source=ROOT/"configs/sixs_j1r1_full_joint_unrestricted_movement.json"; data=json.loads(source.read_text(encoding="utf-8")); data["seed"]=seed; data["experiment_id"]=f"SIXS_FINAL_UNRESTRICTED_{variant.upper()}_SEED{seed}_STEP17500"; data["matched_ablation"]={"variant":variant,"only_change":"reliability output=1" if variant=="reliability_off" else "adaptive BA weights=0.5/0.5"}
 target=REPORT/f"{variant}_seed{seed}/FROZEN_CONFIG.json"; encoded=json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+"\n"; target.parent.mkdir(parents=True,exist_ok=True)
 if target.is_file() and target.read_text(encoding="utf-8")!=encoded: raise RuntimeError(f"config drift {target}")
 target.write_text(encoded,encoding="utf-8"); return target

def training():
 rows=[]; stage=4
 for variant in VARIANTS:
  for seed in SEEDS:
   name=f"STAGE_{stage}_TRAIN_{variant.upper()}_{seed}"; stage+=1; report=REPORT/f"{variant}_seed{seed}"; artifact=ARTIFACT/f"{variant}_seed{seed}"; final=report/"FINAL_CHECKPOINT.pt"; config=make_config(variant,seed)
   if not final.is_file(): run(name,[CUDA,ROOT/"scripts/run_sixs_final_matched_ablation_training.py","--variant",variant,"--seed",seed,"--config",config,"--report-dir",report,"--artifact-dir",artifact],final)
   saved=torch.load(final,map_location="cpu",weights_only=False); rows.append({"variant":variant,"seed":seed,"step":int(saved.get("step",-1)),"checkpoint":str(final),"checkpoint_sha256":sha(final),"gpu_preflight":json.loads((report/"GPU_PREFLIGHT.json").read_text(encoding="utf-8")).get("status"),"dev_checkpoint_selection":False})
 pd.DataFrame(rows).to_csv(REPORT/"02_TRAINING_INTEGRITY.csv",index=False)

def execute():
 REPORT.mkdir(parents=True,exist_ok=True); ARTIFACT.mkdir(parents=True,exist_ok=True); ASSET.mkdir(parents=True,exist_ok=True)
 try:
  update("STAGE_1_VALIDATE_CROSS_UPSTREAM","RUNNING",cross_upstream_status="VALIDATING",next_automatic_stage="STAGE_2_FREEZE_CROSS_UPSTREAM"); validate_cross(); freeze_cross()
  prepare_protocol(); update("STAGE_3_PREPARE_FINAL_ABLATION","PASS",END_TIME=time.time(),EXIT_CODE=0,OUTPUT_SHA=sha(PROTOCOL),next_automatic_stage="STAGE_4_TRAIN_RELIABILITY_OFF_307")
  training()
  run("STAGE_10_INFERENCE_CONTROLS",[CUDA,ROOT/"scripts/run_sixs_final_matched_ablation_evaluation.py","coordinates","--protocol",PROTOCOL,"--output-dir",ASSET,"--report-dir",REPORT])
  run("STAGE_11_FULL_EVALUATION",[CUDA,ROOT/"scripts/run_sixs_final_matched_ablation_evaluation.py","external","--protocol",PROTOCOL,"--output-dir",ASSET,"--report-dir",REPORT])
  run("STAGE_11_XTB_SUBSET",[CUDA,ROOT/"scripts/run_sixs_final_matched_ablation_xtb.py","--protocol",PROTOCOL,"--coordinate-dir",ASSET,"--output-dir",ASSET/"xtb_subset","--report-dir",REPORT],REPORT/"XTB_FINAL_STATUS.json")
  run("STAGE_12_STATISTICS",[CUDA,ROOT/"scripts/finalize_sixs_final_matched_ablation.py","--protocol",PROTOCOL,"--output-dir",ASSET,"--report-dir",REPORT,"--xtb-dir",ASSET/"xtb_subset"],REPORT/"10_FINAL_ABLATION_CONCLUSION.md")
  update("STAGE_13_FINAL_SUMMARY","PASS",END_TIME=time.time(),EXIT_CODE=0,cross_upstream_status="PASS",next_automatic_stage="NONE")
  return 0
 except Exception as e:
  (REPORT/"PIPELINE_TRACEBACK.txt").write_text(traceback.format_exc(),encoding="utf-8"); update("PIPELINE_BLOCKED","FAIL",error_type=type(e).__name__,error=str(e),next_automatic_stage="NONE"); raise

def preflight():
 REPORT.mkdir(parents=True,exist_ok=True)
 checks={"cuda_python":CUDA.is_file(),"cuda_available":torch.cuda.is_available(),"primary_status":json.loads((PRIMARY_REPORT/"FINAL_STATUS.json").read_text(encoding="utf-8")).get("status")=="PASS","primary_manifest_sha":sha(PRIMARY_MANIFEST)=="2a1d07af8c9e3150d1f2f3719d0bd43bd33819ca7674c364d0770c010cb86ee1","fixed_tau_source":FIXED_TAU_SOURCE.is_file(),"cross_driver":(ROOT/"scripts/run_sixs_final_cross_upstream_unrestricted.py").is_file()}
 atomic_json(REPORT/"PREFLIGHT.json",{"status":"PASS" if all(checks.values()) else "FAIL","checks":checks,"CUDA_DEVICE_NAME":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"PYTORCH_CUDA_VERSION":torch.version.cuda})
 if not all(checks.values()): raise RuntimeError(f"pipeline preflight failed: {checks}")
 return 0
def recovery_audit():
 previous=read_json(STATUS) if STATUS.is_file() else {}
 atomic_json(REPORT/"PIPELINE_RECOVERY_AUDIT.json",{
  "schema_version":"sixs-matched-ablation-pipeline-recovery-audit-v1",
  "status":"PASS", "timestamp_epoch":time.time(),
  "previous_pid":2380, "previous_process":"NOT_RUNNING",
  "previous_stage":previous.get("stage"), "previous_status":previous.get("status"),
  "last_successful_stage":"PREFLIGHT",
  "exit_class":"ENGINEERING_VALIDATION_BUG",
  "exception":"json.decoder.JSONDecodeError: Unexpected UTF-8 BOM (decode using utf-8-sig): line 1 column 1 (char 0)",
  "exact_file":str(ROOT/"scripts/run_sixs_post_cross_upstream_ablation_pipeline.py"),
  "exact_line_original":43,
  "root_cause":"PowerShell 5.1 wrote STATUS.json with a UTF-8 BOM; Python decoded it with utf-8. The exception handler then repeated the same failing read, leaving stale RUNNING state.",
  "repair":["BOM-tolerant orchestration JSON reader","BOM-free PowerShell status writer","wrapper-level nonzero-exit failure status"],
  "resume_stage":"STAGE_1_VALIDATE_CROSS_UPSTREAM",
  "scientific_recomputation_required":False,
  "cross_scientific_artifacts":{"avgflow":"PASS","ditmc":"PASS","avgflow_xtb":"PASS","ditmc_xtb":"PASS","seeds":[307,331,353]},
  "previous_status_preserved":previous,
 })
 return 0
def main():
 p=argparse.ArgumentParser(); p.add_argument("mode",choices=("preflight","recovery-audit","run")); a=p.parse_args()
 if a.mode=="preflight": return preflight()
 if a.mode=="recovery-audit": return recovery_audit()
 return execute()
if __name__=="__main__": raise SystemExit(main())
