"""Validate artifacts on the processing VM and benchmark the reader (no app)."""
from pathlib import Path
import sys, json, hashlib, time, os, platform
import numpy as np
import pandas as pd
from rank_io import load_shard, auroc, resolve_symbols


def main(directory):
    root=Path(directory)
    m=json.loads((root/'manifest.json').read_text())
    n,g=m['n_cells'],m['n_genes']
    for entry in m['files']:
        p=root/entry['path']
        assert p.stat().st_size==entry['bytes']<95_000_000
        with p.open('rb') as stream:
            assert hashlib.file_digest(stream,'sha256').hexdigest()==entry['sha256']
    cells=pd.read_parquet(root/'cells.parquet')
    genes=pd.read_parquet(root/'genes.parquet')
    assert len(cells)==n and len(genes)==g
    np.testing.assert_array_equal(cells.cell_index,np.arange(n))
    np.testing.assert_array_equal(genes.gene_index,np.arange(g))
    assert genes.symbol.is_unique and cells.source_obs_id.is_unique
    if (root/'pseudobulk.npz').exists():
        with np.load(root/'pseudobulk.npz', allow_pickle=False) as bulk:
            np.testing.assert_array_equal(bulk['symbols'], genes.symbol.astype(str))
            np.testing.assert_array_equal(bulk['counts'].sum(axis=0), genes.total_counts)
            labels, sizes = np.unique(cells.celltype.astype(str), return_counts=True)
            np.testing.assert_array_equal(bulk['labels'], labels)
            np.testing.assert_array_equal(bulk['n_bins'], sizes)
            assert bulk['counts'].shape == (len(labels), g)
            assert np.isfinite(bulk['counts']).all() and (bulk['counts'] >= 0).all()
    load_times=[]; auc_times=[]; fingerprints=np.empty((n,32),dtype=np.uint8); pairwise_checks=0
    chosen,missing=resolve_symbols(genes,['SOX2','PAX6','NES','DCX','TUBB3'])
    if not chosen or len(chosen)==g:
        chosen=list(range(min(5,g-1)))
    for layout in ['progressive','spatial']:
        if layout=='spatial' and not m[layout]: continue
        seen=np.zeros(n,dtype=np.uint32)
        for path in m[layout]:
            t=time.perf_counter(); z=load_shard(root/path); load_times.append(time.perf_counter()-t)
            rows=z['cell_index']; ptr=z['indptr']; idx=z['gene_index']; rank=z['rank2']; zero=z['zero_rank2']
            assert len(ptr)==len(rows)+1 and ptr[0]==0 and ptr[-1]==len(idx)==len(rank)
            assert np.all(np.diff(ptr.astype(np.int64))>=0)
            assert np.all(idx<g) and np.all(rank<=2*g)
            t=time.perf_counter(); scores=auroc(z,chosen,g); auc_times.append(time.perf_counter()-t)
            assert np.all((scores>=0)&(scores<=1))
            for j,row in enumerate(rows):
                a,b=map(int,ptr[j:j+2]); ix=idx[a:b]; rr=rank[a:b]
                assert np.all(np.diff(ix.astype(np.int64))>0)
                zeros=g-len(ix)
                assert zero[j]==(zeros+1 if zeros else 0)
                assert np.all(rr>zero[j])
                assert int(rr.sum(dtype=np.uint64))+zeros*int(zero[j])==g*(g+1)
                fp=np.frombuffer(hashlib.sha256(ix.tobytes()+rr.tobytes()+zero[j].tobytes()).digest(),dtype=np.uint8)
                if layout=='progressive': fingerprints[int(row)]=fp
                else: np.testing.assert_array_equal(fingerprints[int(row)],fp)
            # Independent pairwise test of the reader's output, once per shard.
            r=np.full(g,float(zero[0])/2); r[idx[:ptr[1]]]=rank[:ptr[1]]/2
            other=np.setdiff1d(np.arange(g),chosen)
            truth=np.mean((r[chosen,None]>r[other])+0.5*(r[chosen,None]==r[other]))
            np.testing.assert_allclose(scores[0],truth,atol=1e-14)
            pairwise_checks+=1
            np.add.at(seen,rows,1)
        np.testing.assert_array_equal(seen,np.ones(n,dtype=np.uint32))
    report=dict(checksums_verified=len(m['files']),n_cells=n,n_genes=g,
        all_rank_sums_valid=True,layouts_identical=True if m['spatial'] else None,reader_pairwise_auc_checks=pairwise_checks,
        median_load_ms=float(np.median(load_times)*1000),median_auroc_ms=float(np.median(auc_times)*1000),
        benchmark_gene_set=genes.iloc[chosen].symbol.tolist(),
        execution_host=platform.node(),execution_platform=platform.platform(),
        benchmark_note='Processing host filesystem; varying shard sizes, OS cache may apply; excludes network transfer',
        total_bytes=sum(f['bytes'] for f in m['files'])+(root/'manifest.json').stat().st_size,
        max_file_bytes=max(f['bytes'] for f in m['files']),
        progressive_bytes=sum(f['bytes'] for f in m['files'] if f.get('layout')=='progressive'),
        spatial_bytes=sum(f['bytes'] for f in m['files'] if f.get('layout')=='spatial'),
        genes_detected_zero_cells=int((genes.n_cells_detected==0).sum()),
        genes_detected_fewer_than_3_cells=int((genes.n_cells_detected<3).sum()))
    (root/os.environ.get('FETAL_VALIDATION_REPORT','validation_colab.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main(sys.argv[1])
