"""Bounded-memory H5AD -> exact sparse ranks and progressive/spatial shards.
Run locally or with colab exec; configuration is via FETAL_* environment variables.
"""
from pathlib import Path
import os, json, hashlib, urllib.request, datetime, time, platform
import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from scipy.stats import rankdata
import h5py
from anndata.io import read_elem

SOURCE_URL = 'https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9629nnn/GSM9629296/suppl/GSM9629296_CS12-13_E2S3_HESTA.h5ad'
HGNC_URL = 'https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt'


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def download(url, path):
    if not path.exists():
        tmp = path.with_suffix(path.suffix + '.part')
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)


def csr_rows(group, start, stop):
    ptr = group['indptr'][start:stop + 1]
    lo, hi = int(ptr[0]), int(ptr[-1])
    return sparse.csr_matrix((group['data'][lo:hi], group['indices'][lo:hi], ptr - lo),
                             shape=(stop-start, int(group.attrs['shape'][1])))


def map_genes(names, hgnc):
    approved = hgnc[hgnc.status.eq('Approved')].copy()
    current = dict(zip(approved.symbol, approved.hgnc_id))
    previous, ensembl = {}, {}
    for row in approved.itertuples():
        for token in str(row.prev_symbol).split('|') if pd.notna(row.prev_symbol) else []:
            previous.setdefault(token, set()).add(row.hgnc_id)
        if pd.notna(row.ensembl_gene_id):
            ensembl.setdefault(str(row.ensembl_gene_id), set()).add(row.hgnc_id)
    pc = approved[approved.locus_group.eq('protein-coding gene')].set_index('hgnc_id')
    rows = []
    for i, name in enumerate(names):
        name = str(name)
        candidates, method = set(), 'unmapped'
        if name in current:
            candidates, method = {current[name]}, 'approved_symbol'
        elif name.startswith('ENSG') and name.split('.')[0] in ensembl:
            candidates, method = ensembl[name.split('.')[0]], 'ensembl'
        elif name in previous:
            candidates, method = previous[name], 'previous_symbol'
        ident = next(iter(candidates)) if len(candidates) == 1 else None
        status = 'retained' if ident in pc.index else ('ambiguous' if len(candidates)>1 else 'not_protein_coding_or_unmapped')
        rows.append((i, name, ident, pc.loc[ident, 'symbol'] if ident in pc.index else None, method, status))
    return pd.DataFrame(rows, columns=['source_gene_index','source_gene','hgnc_id','symbol','mapping_method','status']), pc


def main():
    base = Path(os.environ.get('FETAL_WORK', '/content/fetal'))
    base.mkdir(parents=True, exist_ok=True)
    source = Path(os.environ.get('FETAL_INPUT', str(base/'source.h5ad')))
    out = Path(os.environ.get('FETAL_OUTPUT', str(base/'GSM9629296')))
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('Output must be empty to prevent stale shards')
    min_cells = int(os.environ.get('FETAL_MIN_CELLS', '0'))
    shard_cells = int(os.environ.get('FETAL_SHARD_CELLS', '512'))
    grid = int(os.environ.get('FETAL_GRID', '8'))
    seed = int(os.environ.get('FETAL_SEED', '20261008'))
    if min_cells < 0 or shard_cells < 1 or grid < 1:
        raise ValueError('Invalid configuration')
    download(os.environ.get('FETAL_URL', SOURCE_URL), source)
    ref = base/'hgnc_complete_set.tsv'
    download(HGNC_URL, ref)
    hgnc = pd.read_csv(ref, sep='\t', low_memory=False)
    files = []
    def record(path, **extra):
        size = path.stat().st_size
        if size >= 95_000_000:
            raise ValueError(f'File exceeds conservative GitHub limit: {path}')
        item = dict(path=str(path.relative_to(out)), bytes=size, sha256=sha256(path), **extra)
        files.append(item)
        return item
    with h5py.File(source) as f:
        obs, var = read_elem(f['obs']), read_elem(f['var'])
        n = len(obs)
        if not {'x','y'} <= set(obs.columns):
            raise ValueError('Expected obs x/y coordinates')
        coords = obs[['x','y']].to_numpy()
        if not np.isfinite(coords).all():
            raise ValueError('Nonfinite coordinates')
        mapping, pc = map_genes(var.index, hgnc)
        symbols = sorted(mapping.loc[mapping.status.eq('retained'), 'symbol'].unique())
        lookup = {s:i for i,s in enumerate(symbols)}
        kept = mapping[mapping.status.eq('retained')]
        projection = sparse.csr_matrix((np.ones(len(kept)),
                    (kept.source_gene_index, kept.symbol.map(lookup))), shape=(len(var),len(symbols)))
        # Aggregate duplicate identifiers before ranking raw counts.
        matrix_path = 'layers/counts'
        group = f[matrix_path]
        if group.attrs.get('encoding-type') != 'csr_matrix':
            raise ValueError('Expected CSR counts; convert counts to CSR before exporting')
        detection = np.zeros(len(symbols), dtype=np.int64)
        totals = np.zeros(len(symbols), dtype=np.float64)
        for start in range(0,n,256):
            x = csr_rows(group,start,min(start+256,n)) @ projection
            x.eliminate_zeros()
            if not np.isfinite(x.data).all() or (x.data<0).any() or not np.equal(x.data,np.floor(x.data)).all():
                raise ValueError('counts must be finite nonnegative integer values')
            detection += np.asarray((x>0).sum(axis=0)).ravel().astype(np.int64)
            totals += np.asarray(x.sum(axis=0)).ravel()
        keep = detection >= min_cells
        symbols = np.asarray(symbols)[keep]
        projection = projection[:,keep]
        g = len(symbols)
        if not 1 < g < 32768:
            raise ValueError('uint16 doubled ranks require 2 <= genes <= 32767')
        genes = pc.reset_index().set_index('symbol').loc[symbols, ['hgnc_id','ensembl_gene_id','locus_group']].reset_index()
        genes.insert(0,'gene_index',np.arange(g,dtype=np.uint16))
        genes['n_cells_detected'] = detection[keep]
        genes['total_counts'] = totals[keep]
        source_names = kept.groupby('symbol').source_gene.agg('|'.join)
        genes['source_genes'] = source_names.reindex(symbols).to_numpy()
        mapping.loc[mapping.status.eq('retained') & ~mapping.symbol.isin(symbols),'status'] = 'below_min_cells'
        mapping = mapping.join(var.reset_index(drop=True).add_prefix('source_var_'))
        genes.to_parquet(out/'genes.parquet',index=False); record(out/'genes.parquet')
        mapping.to_parquet(out/'gene_mapping.parquet',index=False); record(out/'gene_mapping.parquet')
        pc.reset_index()[['hgnc_id','symbol','ensembl_gene_id','prev_symbol']].to_parquet(out/'hgnc_protein_coding.parquet',index=False)
        record(out/'hgnc_protein_coding.parquet')
        # Disk-backed sparse arrays avoid a cells x genes dense allocation.
        total_nnz = int(detection[keep].sum())
        rank2 = np.lib.format.open_memmap(base/'rank2.tmp.npy',mode='w+',dtype='uint16',shape=(total_nnz,))
        indices = np.lib.format.open_memmap(base/'indices.tmp.npy',mode='w+',dtype='int32',shape=(total_nnz,))
        indptr = np.zeros(n+1,dtype=np.int64)
        zero2 = np.zeros(n,dtype=np.uint16)
        offset=0
        rng = np.random.default_rng(seed)
        validation_rows = set(rng.choice(n,min(n,64),replace=False).tolist())
        validated=0
        for start in range(0,n,256):
            x = (csr_rows(group,start,min(start+256,n)) @ projection).tocsr()
            x.eliminate_zeros(); x.sort_indices()
            for j in range(x.shape[0]):
                row = start+j
                a,b=x.indptr[j:j+2]; values=x.data[a:b]; cols=x.indices[a:b]
                zeros=g-len(values)
                zero2[row]=zeros+1 if zeros else 0
                doubled=(2*(rankdata(values,method='average')+zeros)).astype(np.uint16)
                rank2[offset:offset+len(values)]=doubled
                indices[offset:offset+len(values)]=cols
                offset+=len(values); indptr[row+1]=offset
                if row in validation_rows:
                    dense=x[j].toarray().ravel()
                    decoded=np.full(g,float(zero2[row])/2); decoded[cols]=doubled/2
                    np.testing.assert_array_equal(decoded,rankdata(dense,method='average'))
                    selected=rng.choice(g,min(100,g-1),replace=False)
                    other=np.setdiff1d(np.arange(g),selected)
                    # Independent pairwise ROC definition includes half credit for ties.
                    truth=np.mean((dense[selected,None]>dense[other])+0.5*(dense[selected,None]==dense[other]))
                    k=len(selected)
                    auc=(decoded[selected].sum()-k*(k+1)/2)/(k*(g-k))
                    np.testing.assert_allclose(auc,truth,atol=1e-14)
                    validated+=1
            if start%2048==0: print('ranked',min(start+256,n),'/',n,flush=True)
        assert offset==total_nnz
        rank2.flush(); indices.flush()
        def rank_rows(rows):
            # Gather only one shard. Constructing a global SciPy CSR can silently
            # copy the full indices array when nnz requires 64-bit pointers.
            lengths = indptr[rows+1]-indptr[rows]
            ptr = np.r_[0,np.cumsum(lengths)]
            cols = np.empty(int(ptr[-1]),dtype=np.int32)
            values = np.empty(int(ptr[-1]),dtype=np.uint16)
            for j,row in enumerate(rows):
                a,b = int(indptr[row]),int(indptr[row+1])
                cols[ptr[j]:ptr[j+1]] = indices[a:b]
                values[ptr[j]:ptr[j+1]] = rank2[a:b]
            return sparse.csr_matrix((values,cols,ptr),shape=(len(rows),g))
        sampling=os.environ.get('FETAL_SAMPLING','random')
        levels=[]
        if sampling=='spatial':
            from spatial_order import spatial_order
            order,levels=spatial_order(coords,seed)
        else:
            order=np.random.default_rng(seed).permutation(n)
        priority=np.empty(n,dtype=np.int64); priority[order]=np.arange(n)
        obs=obs.copy(); obs.insert(0,'source_obs_id',obs.index.astype(str)); obs.insert(0,'cell_index',np.arange(n,dtype=np.uint32))
        obs['sampling_order']=priority
        lo=coords.min(axis=0); hi=coords.max(axis=0)
        tile_xy=np.minimum(((coords-lo)/np.maximum(hi-lo,1)*grid).astype(int),grid-1)
        obs['tile_x']=tile_xy[:,0]; obs['tile_y']=tile_xy[:,1]
        obs['tile_id']=[f'x{x:03d}_y{y:03d}' for x,y in tile_xy]
        obs['progressive_shard']=priority//shard_cells
        obs['progressive_row']=priority%shard_cells
        obs.to_parquet(out/'cells.parquet',index=False); record(out/'cells.parquet')
        def emit(rows,layout,name):
            path=out/layout/(name+'.npz'); path.parent.mkdir(exist_ok=True)
            block=rank_rows(rows)
            np.savez_compressed(path, cell_index=np.asarray(rows,dtype=np.uint32),
                indptr=block.indptr.astype(np.uint32), gene_index=block.indices.astype(np.uint16),
                rank2=block.data.astype(np.uint16), zero_rank2=zero2[rows])
            if path.stat().st_size>=90_000_000 and len(rows)>1:
                path.unlink(); mid=len(rows)//2
                return emit(rows[:mid],layout,name+'a')+emit(rows[mid:],layout,name+'b')
            box=[float(coords[rows,0].min()),float(coords[rows,1].min()),float(coords[rows,0].max()),float(coords[rows,1].max())]
            return [record(path,layout=layout,n_cells=len(rows),bbox=box,
                           sampling_order_min=int(priority[rows].min()),sampling_order_max=int(priority[rows].max()))]
        progressive=[]; spatial=[]
        for start in range(0,n,shard_cells):
            progressive+=emit(order[start:start+shard_cells],'progressive',f'{start//shard_cells:05d}')
        for tile in ([] if os.environ.get('FETAL_SKIP_SPATIAL')=='1' else sorted(obs.tile_id.unique())):
            rows=np.flatnonzero(obs.tile_id.to_numpy()==tile)
            rows=rows[np.argsort(priority[rows])]
            for start in range(0,len(rows),shard_cells):
                entries=emit(rows[start:start+shard_cells],'spatial',f'{tile}_{start//shard_cells:05d}')
                for entry in entries: entry['tile_id']=tile
                spatial+=entries
        # Verify every serialized shard, all row/gene associations, and both covers.
        timings=[]
        for layout in [progressive,spatial]:
            if not layout: continue
            seen=[]
            for entry in layout:
                t=time.perf_counter()
                with np.load(out/entry['path'],allow_pickle=False) as z:
                    rows=z['cell_index']; block=rank_rows(rows)
                    np.testing.assert_array_equal(z['indptr'],block.indptr)
                    np.testing.assert_array_equal(z['gene_index'],block.indices)
                    np.testing.assert_array_equal(z['rank2'],block.data)
                    np.testing.assert_array_equal(z['zero_rank2'],zero2[rows])
                    seen.extend(rows.tolist())
                timings.append(time.perf_counter()-t)
            np.testing.assert_array_equal(np.sort(seen),np.arange(n))
        source_var_stale=bool('n_cells_by_counts' in var and var.n_cells_by_counts.max()>n)
        manifest=dict(format='sparse-doubled-average-ranks-v1',dataset=os.environ.get('FETAL_DATASET','GSM9629296'),
            created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            n_cells=n,n_genes=g,n_celltypes=int(obs.celltype.nunique()),seed=seed,min_cells=min_cells,
            sampling=dict(method=sampling,levels=levels),progressive_batch_size=shard_cells,
            source=dict(url=os.environ.get('FETAL_URL',SOURCE_URL),bytes=source.stat().st_size,sha256=sha256(source),matrix=matrix_path),
            hgnc=dict(url=HGNC_URL,sha256=sha256(ref),n_protein_coding=len(pc)),
            gene_mapping_counts=mapping.status.value_counts().to_dict(),
            retained_mapping_methods=mapping.loc[mapping.status.eq('retained'),'mapping_method'].value_counts().to_dict(),
            ranks=dict(direction='ascending: larger rank means higher expression',ties='average',scale=2,
                zeros='all unstored genes have rank zero_rank2 / 2; stored ranks are absolute',
                universe='HGNC approved protein-coding genes mapped from source and passing min_cells; absent source genes excluded',
                duplicate_genes='sum counts before ranking',auroc='(sum(rank of set) - k*(k+1)/2) / (k*(G-k))'),
            coordinates=dict(source='obs.x, obs.y',units='source units, not converted to micrometres',
                obsm_spatial_equal=bool('spatial' in f['obsm'] and np.array_equal(coords,read_elem(f['obsm/spatial']))),
                bbox=[float(lo[0]),float(lo[1]),float(hi[0]),float(hi[1])],grid=grid),
            source_var_detection_counts_stale=source_var_stale,
            validation=dict(exact_rank_and_pairwise_auc_rows=validated,all_shards_roundtripped=True,
                layouts_each_cover_all_cells_once=True,median_shard_read_and_verify_seconds=float(np.median(timings))),
            software=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,h5py=h5py.__version__),
            files=files,progressive=[x['path'] for x in progressive],spatial=[x['path'] for x in spatial])
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print(json.dumps({k:v for k,v in manifest.items() if k not in ['files','progressive','spatial']},indent=2))
        print('OUTPUT_BYTES',sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),flush=True)
    for name in ['rank2.tmp.npy','indices.tmp.npy']:
        (base/name).unlink()

if __name__=='__main__':
    main()
