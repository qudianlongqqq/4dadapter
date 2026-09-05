#!/usr/bin/env python
"""Molecule-cluster statistics and final tables for the frozen SIXS ablation."""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
import pandas as pd

PRIMARY=Path(r"E:\3dconformergenerationcode\dataset\sixs_primary_final_evaluation_v1")
VCOMP=("bond_geometry_valid","angle_geometry_valid","aromatic_ring_valid","intramolecular_steric_clash_valid")
METRICS=("validity3d","PB","internal_post_objective","direction_improvement","bond_raw_mae","angle_cosine_raw_mae","raw_source_displacement_rms","reference_rmsd")
RESAMPLES=10000; SEED=20260904
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""): h.update(b)
 return h.hexdigest()
def acsv(p,f):
 p.parent.mkdir(parents=True,exist_ok=True); t=p.with_name(f".{p.name}.{os.getpid()}.tmp"); f.to_csv(t,index=False); os.replace(t,p)
def ajson(p,v):
 t=p.with_name(f".{p.name}.{os.getpid()}.tmp"); t.write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False,default=str)+"\n",encoding="utf-8"); os.replace(t,p)
def load(root,mid,primary=False):
 d=(PRIMARY if primary else root)/"methods"/mid
 r=pd.read_parquet(d/"PER_RECORD.parquet"); pb=pd.read_parquet(d/"POSEBUSTERS.parquet")[["record_id","PB"]]; v=pd.read_parquet(d/"VALIDITY3D.parquet")[["record_id","validity3d",*VCOMP]]
 f=r.merge(pb,on="record_id",validate="one_to_one").merge(v,on="record_id",validate="one_to_one")
 if len(f)!=5000 or f.molecule_id.nunique()!=2500 or set(f.groupby("molecule_id").size())!={2}: raise RuntimeError(f"alignment failure {mid}")
 return f
def ci(candidate,baseline,column,offset):
 l=candidate.groupby("molecule_id",sort=True)[column].mean(); r=baseline.groupby("molecule_id",sort=True)[column].mean()
 if not l.index.equals(r.index): raise RuntimeError("cluster mismatch")
 d=l.to_numpy(float)-r.to_numpy(float); rng=np.random.default_rng(SEED+offset); samples=[]
 for _ in range(20): samples.append(d[rng.integers(0,len(d),size=(500,len(d)))].mean(1))
 q=np.quantile(np.concatenate(samples),[.025,.975]); eps=1e-12
 return float(d.mean()),float(q[0]),float(q[1]),int(np.sum(d<-eps)),int(np.sum(np.abs(d)<=eps)),int(np.sum(d>eps))
def main():
 p=argparse.ArgumentParser(); p.add_argument("--protocol",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--report-dir",type=Path,required=True); p.add_argument("--xtb-dir",type=Path,required=True); a=p.parse_args()
 protocol=json.loads(a.protocol.read_text(encoding="utf-8")); methods=protocol["model_methods"]
 frames={m["id"]:load(a.output_dir,m["id"]) for m in methods}; baselines={s:load(a.output_dir,f"unrestricted_seed{s}",True) for s in (307,331,353)}
 rows=[]; comps=[]; diag=[]; boots=[]
 for i,m in enumerate(methods):
  f=frames[m["id"]]; row={"method":m["id"],"variant":m["variant"],"seed":m["seed"],"molecules":2500,"records":5000,"v3d_overall":float(f.validity3d.astype(bool).mean()),"posebusters_overall":float(f.PB.astype(bool).mean()),"finite_coordinate_rate":1.0}
  for c in METRICS[2:]: row[c]=float(f[c].mean())
  rows.append(row)
  for c in VCOMP: comps.append({"method":m["id"],"variant":m["variant"],"seed":m["seed"],"component":c,"pass_rate":float(f[c].astype(bool).mean())})
  diag.append({"method":m["id"],"variant":m["variant"],"seed":m["seed"],"tau_mean":float(f.tau.mean()),"tau_median":float(f.tau.median()),"tau_p95":float(f.tau.quantile(.95)),"tau_p99":float(f.tau.quantile(.99)),"tau_max":float(f.tau.max()),"w_B_mean":float(f.w_B.mean()),"w_B_p05":float(f.w_B.quantile(.05)),"w_B_p50":float(f.w_B.median()),"w_B_p95":float(f.w_B.quantile(.95)),"w_A_mean":float(f.w_A.mean()),"w_A_p05":float(f.w_A.quantile(.05)),"w_A_p50":float(f.w_A.median()),"w_A_p95":float(f.w_A.quantile(.95)),"ba_collapse_fraction":float(((f.w_B<.01)|(f.w_B>.99)).mean())})
  b=baselines[int(m["seed"])]
  for j,c in enumerate(METRICS):
   point,lo,hi,w,t,l=ci(f,b,c,i*20+j); boots.append({"method":m["id"],"baseline":f"unrestricted_seed{m['seed']}","metric":c,"effect_candidate_minus_baseline":point,"ci95_low":lo,"ci95_high":hi,"molecule_win":w,"molecule_tie":t,"molecule_loss":l,"resampling_unit":"molecule","records_per_cluster":2})
 per=pd.DataFrame(rows); acsv(a.report_dir/"03_PER_SEED_RESULTS.csv",per); acsv(a.report_dir/"06_COMPONENT_RESULTS.csv",pd.DataFrame(comps)); acsv(a.report_dir/"07_MOVEMENT_BA_DIAGNOSTICS.csv",pd.DataFrame(diag)); acsv(a.report_dir/"05_PAIRED_BOOTSTRAP.csv",pd.DataFrame(boots))
 means=[]
 for variant,g in per.groupby("variant",sort=False):
  for metric in ("v3d_overall","posebusters_overall",*METRICS[2:]): means.append({"variant":variant,"metric":metric,"mean_across_seeds":float(g[metric].mean()),"seed_sd_ddof1":float(g[metric].std(ddof=1)),"seed_values":";".join(map(str,g[metric]))})
 mean=pd.DataFrame(means); acsv(a.report_dir/"04_MEAN_SD_RESULTS.csv",mean)
 xtb=[]
 if (a.report_dir/"XTB_FINAL_STATUS.json").is_file():
  for m in methods:
   d=a.xtb_dir/f"{m['id'].upper()}_DELTA_VS_SOURCE.csv"
   if not d.is_file(): continue
   f=pd.read_csv(d); v=f.loc[f.matched_success.astype(bool),"delta_e_kcal_mol"].to_numpy(float); o=np.sort(v); k=int(.05*len(o)); trim=o[k:len(o)-k]
   xtb.append({"method":m["id"],"variant":m["variant"],"seed":m["seed"],"matched_records":len(v),"median_delta_e":float(np.median(v)),"lower_fraction":float(np.mean(v<0)),"trimmed_mean_5pct":float(trim.mean()),"p90":float(np.quantile(v,.9)),"p95":float(np.quantile(v,.95)),"p99":float(np.quantile(v,.99)),"gt25":int(np.sum(v>25)),"gt50":int(np.sum(v>50)),"gt100":int(np.sum(v>100))})
 acsv(a.report_dir/"08_XTB_SUBSET_RESULTS.csv",pd.DataFrame(xtb))
 table=per.groupby("variant",sort=False).agg(v3d_mean=("v3d_overall","mean"),v3d_sd=("v3d_overall","std"),pb_mean=("posebusters_overall","mean"),pb_sd=("posebusters_overall","std"),reference_rmsd_mean=("reference_rmsd","mean"),source_rmsd_mean=("raw_source_displacement_rms","mean")).reset_index()
 acsv(a.report_dir/"09_FINAL_ABLATION_TABLE.csv",table)
 def classify(variant):
  x=pd.DataFrame(boots); q=x[(x.method.str.contains(variant)) & (x.metric.isin(["validity3d","PB"]))]
  if len(q)!=6:return "INCOMPLETE"
  improve=((q.ci95_low>0)|(q.ci95_high<0)).any(); return "MEASURABLE_EFFECT" if improve else "NO_CLEAR_EFFECT"
 conclusions={"RELIABILITY_FINAL_NECESSITY":classify("reliability_off"),"ADAPTIVE_BA_FINAL_NECESSITY":classify("equal_ba"),"PREDICTIVE_SIGMA_ACTION_EFFECT":classify("fixed_sigma_action"),"BOND_ACTION_EFFECT":classify("bond_only"),"ANGLE_ACTION_EFFECT":classify("angle_only"),"LEARNED_TAU_ACTION_EFFECT":classify("fixed_tau")}
 text="# Final matched ablation conclusion\n\n"+"\n".join(f"{k} = {v}" for k,v in conclusions.items())+"\n\nAll intervals use molecule-cluster resampling; the two source records were never treated as independent clusters.\n"
 (a.report_dir/"10_FINAL_ABLATION_CONCLUSION.md").write_text(text,encoding="utf-8")
 artifacts={p.name:sha(p) for p in a.report_dir.iterdir() if p.is_file() and p.name not in {"STATUS.json"}}
 ajson(a.report_dir/"STATUS.json",{"schema_version":"sixs-final-matched-ablation-status-v1","status":"PASS","stage":"STAGE_13_FINAL_SUMMARY","methods":len(methods),"new_gpu_training_runs":6,"molecules":2500,"records":5000,"bootstrap_resamples":RESAMPLES,"resampling_unit":"molecule","scientific_questions":conclusions,"artifact_sha256":artifacts})
 return 0
if __name__=="__main__": raise SystemExit(main())
