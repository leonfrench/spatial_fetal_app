"""Add count pseudobulks to an existing export using its exact gene mapping.

Run on the preparation VM: python scripts/prepare_pseudobulk.py DATA_DIR SOURCE.h5ad
Only a tissues x genes matrix is dense; source counts are read in bounded blocks.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from scipy import sparse
import h5py
from anndata.io import read_elem


def build(root, source):
    root, source = Path(root), Path(source)
    manifest = json.loads((root/'manifest.json').read_text())
    with source.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != manifest['source']['sha256']:
        raise ValueError('Source checksum does not match the rank export')
    genes = pd.read_parquet(root/'genes.parquet').sort_values('gene_index')
    cells = pd.read_parquet(root/'cells.parquet').sort_values('cell_index')
    with h5py.File(source) as f:
        obs, var = read_elem(f['obs']), read_elem(f['var'])
        np.testing.assert_array_equal(obs.index.astype(str), cells.source_obs_id.astype(str))
        np.testing.assert_array_equal(obs.celltype.astype(str), cells.celltype.astype(str))
        lookup = {str(name): i for i, name in enumerate(var.index)}
        if len(lookup) != len(var):
            raise ValueError('Source gene identifiers must be unique')
        rows, cols = [], []
        for col, names in enumerate(genes.source_genes):
            for name in names.split('|'):
                rows.append(lookup[name]); cols.append(col)
        if len(set(rows)) != len(rows):
            raise ValueError('Source gene mapped more than once')
        projection = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(var), len(genes)))
        labels, codes = np.unique(obs.celltype.astype(str), return_inverse=True)
        n_bins = np.bincount(codes, minlength=len(labels))
        counts = np.zeros((len(labels), len(genes)), dtype=np.float64)
        group = f[manifest['source']['matrix']]
        if group.attrs.get('encoding-type') != 'csr_matrix':
            raise ValueError('Expected CSR source counts')
        for start in range(0, len(obs), 1024):
            stop = min(start+1024, len(obs))
            ptr = group['indptr'][start:stop+1]
            lo, hi = int(ptr[0]), int(ptr[-1])
            block = sparse.csr_matrix((group['data'][lo:hi], group['indices'][lo:hi], ptr-ptr[0]), shape=(stop-start, len(var))) @ projection
            if not np.isfinite(block.data).all() or (block.data < 0).any() or (block.data != np.floor(block.data)).any():
                raise ValueError('Expected nonnegative integer counts')
            grouping = sparse.csr_matrix((np.ones(stop-start), (codes[start:stop], np.arange(stop-start))), shape=(len(labels), stop-start))
            counts += (grouping @ block).toarray()
        np.testing.assert_array_equal(counts.sum(axis=0), genes.total_counts)
    path = root/'pseudobulk.npz'
    np.savez_compressed(path, counts=counts, labels=labels.astype(str), n_bins=n_bins,
                        symbols=genes.symbol.to_numpy(dtype=str))
    with path.open('rb') as stream:
        checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
    report = dict(method='sum raw mapped counts by celltype', source_sha256=digest,
                  validation='source checksum, observation order/labels, and per-gene count totals match original export',
                  n_bins=int(n_bins.sum()), n_profiles=len(labels), n_genes=len(genes), sha256=checksum)
    (root/'pseudobulk_validation.json').write_text(json.dumps(report, indent=2)+'\n')
    names = {'pseudobulk.npz', 'pseudobulk_validation.json'}
    manifest['files'] = [entry for entry in manifest.get('files', []) if entry['path'] not in names]
    for name in sorted(names):
        artifact = root/name
        with artifact.open('rb') as stream:
            artifact_digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        manifest['files'].append(dict(path=name, bytes=artifact.stat().st_size, sha256=artifact_digest))
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root'); parser.add_argument('source')
    args = parser.parse_args()
    print(build(args.root, args.source))
