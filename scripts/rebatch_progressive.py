"""Losslessly regroup existing random rank files, preserving their row order."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import numpy as np
import pandas as pd
from rank_io import load_shard


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main(root,batch_size):
    root=Path(root).resolve()
    if batch_size<1: raise ValueError('Batch size must be positive')
    manifest=json.loads((root/'manifest.json').read_text())
    old_manifest_hash=digest(root/'manifest.json')
    staging=Path(tempfile.mkdtemp(prefix='.rebatch-',dir=root.parent))
    for path in root.iterdir():
        if path.name in {'progressive','manifest.json','validation_local.json','validation_colab.json','transfer_receipt.json'}:
            continue
        if path.is_dir(): shutil.copytree(path,staging/path.name)
        else: shutil.copy2(path,staging/path.name)
    (staging/'progressive').mkdir()
    cells=pd.read_parquet(root/'cells.parquet').sort_values('cell_index').reset_index(drop=True)
    entries=[]; parts=[]; pending=0; seen=[]
    def flush():
        nonlocal parts,pending
        pointers=[np.array([0],dtype=np.uint64)]; offset=0
        for part in parts:
            pointers.append(part['indptr'][1:].astype(np.uint64)+offset)
            offset+=len(part['rank2'])
        if offset>np.iinfo(np.uint32).max: raise ValueError('Batch has too many nonzero entries')
        joined={key:np.concatenate([p[key] for p in parts]) for key in ['cell_index','gene_index','rank2','zero_rank2']}
        joined['indptr']=np.concatenate(pointers).astype(np.uint32)
        relative=f'progressive/{len(entries):05d}.npz'; path=staging/relative
        np.savez_compressed(path,**joined)
        if path.stat().st_size>=95_000_000: raise ValueError('Batch exceeds file size limit; use a smaller batch')
        restored=load_shard(path)
        for key,value in joined.items(): np.testing.assert_array_equal(restored[key],value)
        rows=joined['cell_index']; obs=cells.iloc[rows]
        np.testing.assert_array_equal(obs.sampling_order,np.arange(len(seen),len(seen)+len(rows)))
        seen.extend(rows.tolist())
        entries.append(dict(path=relative,bytes=path.stat().st_size,sha256=digest(path),layout='progressive',
            n_cells=len(rows),bbox=[float(obs.x.min()),float(obs.y.min()),float(obs.x.max()),float(obs.y.max())],
            sampling_order_min=int(obs.sampling_order.min()),sampling_order_max=int(obs.sampling_order.max())))
        print(f'{relative}: {len(rows)} observations, {path.stat().st_size:,} bytes',flush=True)
        parts=[];pending=0
    for name in manifest['progressive']:
        old=load_shard(root/name)
        # Slice contiguous spans, preserving the exact existing random order.
        start=0
        while start<len(old['cell_index']):
            stop=min(len(old['cell_index']),start+batch_size-pending)
            lo,hi=map(int,[old['indptr'][start],old['indptr'][stop]])
            parts.append(dict(cell_index=old['cell_index'][start:stop],zero_rank2=old['zero_rank2'][start:stop],
                indptr=old['indptr'][start:stop+1]-lo,gene_index=old['gene_index'][lo:hi],rank2=old['rank2'][lo:hi]))
            pending+=stop-start;start=stop
            if pending==batch_size: flush()
    if pending: flush()
    np.testing.assert_array_equal(np.sort(seen),np.arange(len(cells)))
    cells['progressive_shard']=cells.sampling_order//batch_size
    cells['progressive_row']=cells.sampling_order%batch_size
    cells.to_parquet(staging/'cells.parquet',index=False)
    files=[entry for entry in manifest['files'] if entry.get('layout')!='progressive' and entry['path']!='cells.parquet']
    files.append(dict(path='cells.parquet',bytes=(staging/'cells.parquet').stat().st_size,sha256=digest(staging/'cells.parquet')))
    files.extend(entries)
    manifest['files']=files
    manifest['progressive']=[e['path'] for e in entries]
    manifest['progressive_batch_size']=batch_size
    manifest['rebatching']=dict(previous_manifest_sha256=old_manifest_hash,previous_created_utc=manifest['created_utc'],
        ranks_and_random_order_preserved=True,all_new_arrays_roundtripped=True,location='local')
    manifest['created_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (staging/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    env=dict(os.environ,FETAL_VALIDATION_REPORT='validation_local.json')
    subprocess.run([sys.executable,str(Path(__file__).with_name('validate_export.py')),str(staging)],env=env,check=True)
    # Only replace the live export after its complete replacement has passed validation.
    backup=root.with_name('.'+root.name+'-before-rebatch')
    if backup.exists():
        backup=root.with_name('.'+root.name+'-'+staging.name.removeprefix('.rebatch-')+'-before-rebatch')
    root.rename(backup)
    try: staging.rename(root)
    except BaseException:
        backup.rename(root)
        raise
    print(f'Updated {root}; prior export saved at {backup}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',type=Path)
    parser.add_argument('--batch-size',type=int,default=4096)
    args=parser.parse_args(); main(args.dataset,args.batch_size)
