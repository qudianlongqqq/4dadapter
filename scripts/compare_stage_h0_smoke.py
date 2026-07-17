#!/usr/bin/env python
from __future__ import annotations
import argparse,json,math
from pathlib import Path
def compare(r,c,atol=1e-6,rtol=1e-5):
    exact=["checkpoint_sha256","validation_source_sha256","validation_target_sha256","record_ids","molecule_ids","validation_records","test_records","variants","clean_identity"]
    mismatches=[key for key in exact if r.get(key)!=c.get(key)];max_abs=max_rel=0.;numeric_mismatch=[]
    for method in sorted(set(r.get("methods",{}))|set(c.get("methods",{}))):
        for key,rv in r.get("methods",{}).get(method,{}).items():
            cv=c.get("methods",{}).get(method,{}).get(key)
            if isinstance(rv,(int,float)) and isinstance(cv,(int,float)):
                error=abs(float(rv)-float(cv));rel=error/max(abs(float(rv)),1e-30);max_abs=max(max_abs,error);max_rel=max(max_rel,rel)
                if not math.isclose(float(rv),float(cv),abs_tol=atol,rel_tol=rtol):numeric_mismatch.append(f"{method}.{key}")
            elif rv!=cv:mismatches.append(f"{method}.{key}")
    status="MISMATCH" if mismatches or numeric_mismatch else ("MATCH" if max_abs==0 else "MATCH_WITH_NUMERICAL_TOLERANCE")
    return {"status":status,"identity_mismatches":mismatches,"numeric_mismatches":numeric_mismatch,"max_absolute_error":max_abs,"max_relative_error":max_rel,"tolerance":{"absolute":atol,"relative":rtol}}
def main():
    p=argparse.ArgumentParser();p.add_argument("--reference",type=Path,required=True);p.add_argument("--candidate",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();out=compare(json.loads(a.reference.read_text()),json.loads(a.candidate.read_text()));a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2),encoding="utf-8");print(out["status"]);raise SystemExit(out["status"]=="MISMATCH")
if __name__=="__main__":main()
