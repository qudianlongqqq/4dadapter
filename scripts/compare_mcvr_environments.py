#!/usr/bin/env python
from __future__ import annotations
import argparse,json,re
from pathlib import Path
import yaml
def normalize(value,mode):
    text=str(value); match=re.match(r"(\d+)\.(\d+)",text)
    return ".".join(match.groups()) if mode=="major_minor" and match else text
def compare(reference,candidate,policy):
    hard=[]; warnings=[]
    for key,mode in policy["hard_match"].items():
        rv=reference.get("packages",{}).get(key,reference.get(key)); cv=candidate.get("packages",{}).get(key,candidate.get(key))
        if normalize(rv,mode)!=normalize(cv,mode): hard.append({"field":key,"reference":rv,"candidate":cv})
    for key in policy["warn_only"]:
        rv=reference.get(key);cv=candidate.get(key)
        if rv!=cv:warnings.append({"field":key,"reference":rv,"candidate":cv})
    return {"status":"INCOMPATIBLE" if hard else ("COMPATIBLE_WITH_WARNINGS" if warnings else "COMPATIBLE"),"hard_mismatches":hard,"warnings":warnings}
def main():
    p=argparse.ArgumentParser();p.add_argument("--reference",type=Path,required=True);p.add_argument("--candidate",type=Path,required=True);p.add_argument("--policy",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();r=compare(json.loads(a.reference.read_text()),json.loads(a.candidate.read_text()),yaml.safe_load(a.policy.read_text()));a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2),encoding="utf-8");print(r["status"]);raise SystemExit(r["status"]=="INCOMPATIBLE")
if __name__=="__main__":main()
