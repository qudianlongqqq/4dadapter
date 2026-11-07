"""Inventory an existing 5090 workspace; no installs, training, or data mutation."""
import argparse,hashlib,importlib.metadata,json,os,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.review_asset_paths import ASSET_ROOT,asset_path

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--hash-check',action='store_true');args=parser.parse_args()
    paths={
        'prepared_train_val':ASSET_ROOT/'4dadapter-lsgo-formal/reports/ecir_mvr/lsgo_formal/cache/FORMAL_PREPARED_GRAPHS.pt',
        'source_binding':ASSET_ROOT/'4dadapter-lsgoba-v2-joint-magnitude-full307/artifacts/ecir_mvr/lsgoba_v2_joint_magnitude_full307/SOURCE_BINDING.pt',
        'primary_protocol':ASSET_ROOT/'4dadapter-lsgoba-musigma-reliability-factorial/reports/ecir_mvr/sixs_primary_final_evaluation/00_FROZEN_FINAL_EVALUATION_PROTOCOL.json',
        'primary_topology_manifest':ASSET_ROOT/'dataset/sixs_primary_final_evaluation_v1/topology_reference_cache/COMPLETE.json',
        'source_records_manifest':ASSET_ROOT/'dataset/prospective_final_etflow_nfe10_seed42/SOURCE_RECORD_MANIFEST.jsonl',
        'genbench_adapter':ASSET_ROOT/'4dadapter-lsgo-v2/scripts/run_lsgo_standard_genbench3d.py',
    }
    expected={
        'prepared_train_val':'b19bd8074f8d1ba3c2d78b90238dc9bcb0cf2b9af2c5d0ca6ce9c17efd44f5b0',
        'source_binding':'c6b2eab6e56eab90e68e0f84ab465ef2308a3d13efdd4da5f0ac88823de2657d',
        'source_records_manifest':'e3e69398b3fd888c3e1a4a9dcc79e48d602a22d6098169704c327773c15ab3dc',
        'genbench_adapter':'6ebc450b4f841a9f6a3b463b7838e50bc2951e92570146cd0a473abbfd970450',
    }
    assets={}
    for name,path in paths.items():
        row={'path':str(path),'exists':path.is_file(),'bytes':path.stat().st_size if path.is_file() else None}
        if args.hash_check and path.is_file() and name in expected:
            row.update(sha256=sha(path));row['hash_matches']=row['sha256']==expected[name]
        assets[name]=row
    versions={}
    for package in ['torch','torch-geometric','rdkit','numpy','scipy','pandas','pyarrow','pytest']:
        try:versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:versions[package]=None
    gpu={}
    try:
        import torch
        gpu={'cuda_available':torch.cuda.is_available(),'torch_cuda':torch.version.cuda}
        if gpu['cuda_available']:gpu['devices']=[{'index':i,'name':torch.cuda.get_device_name(i),
            'memory_bytes':torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())]
    except Exception as e:gpu={'error':str(e)}
    # Inspect embedded locators without rewriting hash-bound original manifests.
    if paths['primary_topology_manifest'].is_file():
        meta=json.loads(paths['primary_topology_manifest'].read_text(encoding='utf8'))
        chunks=[asset_path(p) for p in meta['chunks']]
        assets['topology_chunks']={'count':len(chunks),'missing':sum(not p.is_file() for p in chunks)}
    result={'role':'read-only server inventory; not training approval or completed reproduction',
        'asset_root':str(ASSET_ROOT),'versions':versions,'gpu':gpu,'assets':assets,
        'training_started':False,'installs_performed':False,
        'scientific_next_step':'check existing experiment inventory and TRAIN Reference lineage before scheduling new runs'}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
