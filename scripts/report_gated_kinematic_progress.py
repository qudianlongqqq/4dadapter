#!/usr/bin/env python
"""Read-only experiment scanner producing dependency-free Markdown/JSON reports."""

from __future__ import annotations

import argparse,csv,importlib.util,json,os,subprocess
from datetime import datetime,timezone
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap
ROOT=bootstrap()

def atomic_write_json(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.name+f".tmp.{os.getpid()}")
    with temporary.open("w",encoding="utf-8") as handle:
        json.dump(payload,handle,indent=2);handle.flush();os.fsync(handle.fileno())
    os.replace(temporary,path)


def command_output(command):
    try:return subprocess.run(command,capture_output=True,text=True,timeout=10,check=False).stdout.strip()
    except Exception:return ""


def processes():
    if os.name=="nt":
        text=command_output(["powershell","-NoProfile","-Command","Get-CimInstance Win32_Process | Where-Object {$_.CommandLine -match 'gated_kinematic'} | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"])
        try:
            values=json.loads(text) if text else [];return values if isinstance(values,list) else [values]
        except json.JSONDecodeError:return []
    text=command_output(["ps","-eo","pid=,args="])
    return [{"ProcessId":line.strip().split(None,1)[0],"CommandLine":line.strip().split(None,1)[1]} for line in text.splitlines() if "gated_kinematic" in line and len(line.strip().split(None,1))==2]


def gpu_processes():
    text=command_output(["nvidia-smi","--query-compute-apps=pid,process_name","--format=csv,noheader"])
    return [{"pid":parts[0].strip(),"process":parts[1].strip()} for line in text.splitlines() if len(parts:=[p for p in line.split(",")])>=2]


def read_state(path):
    try:return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:return {"status":"invalid_state","path":str(path)}


def read_summaries(paths):
    rows=[]
    for path in paths:
        try:
            with path.open(encoding="utf-8-sig") as handle:
                for row in csv.DictReader(handle):rows.append({"path":str(path),**row})
        except Exception:rows.append({"path":str(path),"status":"empty_or_invalid"})
    return rows


def numeric(row,name):
    try:return float(row.get(name,"nan"))
    except (TypeError,ValueError):return float("nan")


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,default=ROOT);parser.add_argument("--output_dir",type=Path,default=ROOT/"reports")
    args=parser.parse_args();patterns=("logs_gated_kinematic","diagnostics/gated_kinematic_eval","diagnostics/gated_kinematic_basis")
    roots=[args.root/p for p in patterns];state_paths=[p for root in roots if root.exists() for p in root.rglob("run_state.json")]
    states=[{"directory":str(p.parent),**read_state(p)} for p in state_paths]
    summaries=[p for root in roots if root.exists() for p in root.rglob("summary.csv")]
    rows=read_summaries(summaries);metrics=("rmsd_mean","COV-R","COV-P","MAT-R","MAT-P")
    candidates=[r for r in rows if r.get("subset","all")=="all" and numeric(r,"rmsd_mean")==numeric(r,"rmsd_mean")]
    best=min(candidates,key=lambda r:numeric(r,"rmsd_mean")) if candidates else {}
    partial=[str(p) for root in roots if root.exists() for p in root.rglob("*.tmp*")]+[str(p) for root in roots if root.exists() for p in root.rglob("partial_progress.json")]
    completed={s["directory"] for s in states if str(s.get("status","")).lower()=="completed"}
    conflicts=[str(p) for p in summaries if str(p.parent) not in completed and (p.parent/"COMPLETED").exists()]
    dependency_names=("torch","lightning","torch_geometric","yaml","numpy")
    process_rows=processes();command_by_pid={str(row.get("ProcessId")):row.get("CommandLine","") for row in process_rows}
    gpu_rows=gpu_processes()
    for row in gpu_rows: row["full_command"]=command_by_pid.get(str(row.get("pid")),"")
    report={"created_at":datetime.now(timezone.utc).isoformat(),"processes":process_rows,"gpu_processes":gpu_rows,
        "states":states,"counts":{"completed":sum(str(s.get("status","")).lower()=="completed" for s in states),
        "running":sum(str(s.get("status","")).lower() in {"started","running"} for s in states),
        "failed":sum(str(s.get("status","")).lower()=="failed" for s in states),
        "stopped":sum(str(s.get("status","")).lower()=="stopped" for s in states)},
        "latest_checkpoint":str(max((p for root in roots if root.exists() for p in root.rglob("*.ckpt")),key=lambda p:p.stat().st_mtime,default="")),
        "latest_logs":[str(p) for root in roots if root.exists() for p in sorted(root.rglob("*.log"),key=lambda p:p.stat().st_mtime,reverse=True)[:10]],
        "summary_csv":[str(p) for p in summaries],"best_all":best,"summary_rows":rows,
        "partial_files":partial,"output_conflicts":conflicts,
        "missing_dependencies":[name for name in dependency_names if importlib.util.find_spec(name) is None]}
    args.output_dir.mkdir(parents=True,exist_ok=True);atomic_write_json(args.output_dir/"gated_kinematic_latest.json",report)
    lines=["# Gated Kinematic Flow 实时进度","",f"更新时间：{report['created_at']}","",
        f"- 相关进程：{len(report['processes'])}",f"- GPU 进程：{len(report['gpu_processes'])}",
        f"- 已完成/进行中/失败/手动停止：{report['counts']['completed']}/{report['counts']['running']}/{report['counts']['failed']}/{report['counts']['stopped']}",
        f"- 最新 checkpoint：{report['latest_checkpoint'] or '无'}",f"- 输出冲突：{len(conflicts)}",f"- partial 文件：{len(partial)}",
        f"- 缺失依赖：{', '.join(report['missing_dependencies']) or '无'}","","## 当前最佳 all 指标","",
        "| RMSD | COV-R | COV-P | MAT-R | MAT-P |","|---:|---:|---:|---:|---:|",
        "| "+" | ".join(str(best.get(name,"NA")) for name in metrics)+" |","","## 进程与完整命令",""]
    lines.extend([f"- PID {p.get('ProcessId')}: `{p.get('CommandLine','')}`" for p in report["processes"]] or ["- 无"])
    lines.extend(["","## 状态",""]+[f"- {s.get('status')}: `{s['directory']}`" for s in states] or ["- 无"])
    lines.extend(["","## Summary / 诊断字段",""])
    for row in rows:
        lines.append(f"- `{row.get('path')}` subset={row.get('subset',row.get('group',''))}; explained={row.get('mean_torsion_explained_ratio',row.get('kinematic_explained_ratio','NA'))}; gate_mean={row.get('gate_mean','NA')}; active_gate_fraction={row.get('active_gate_fraction','NA')}; torsion_rate_norm={row.get('torsion_rate_norm','NA')}; failure_rate={row.get('failure_rate','NA')}")
    (args.output_dir/"gated_kinematic_latest.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(args.output_dir/"gated_kinematic_latest.md")


if __name__=="__main__":main()
