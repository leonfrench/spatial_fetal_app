import hashlib
import json
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from anndata import AnnData
from scripts.prepare_pseudobulk import build


def test_count_aggregation_preserves_mapping_and_observation_alignment(tmp_path):
    source = tmp_path/'source.h5ad'
    x = np.array([[1, 2, 9, 0], [3, 4, 8, 5], [6, 7, 7, 1]])
    obs = pd.DataFrame({'celltype': ['B', 'A', 'B']}, index=['c0', 'c1', 'c2'])
    adata = AnnData(X=sparse.csr_matrix(x), obs=obs, var=pd.DataFrame(index=['g1', 'old_g1', 'excluded', 'g2']))
    adata.layers['counts'] = adata.X.copy()
    adata.write_h5ad(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = {'source': {'sha256': digest, 'matrix': 'layers/counts'}}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    cells = obs.reset_index(names='source_obs_id')
    cells.insert(0, 'cell_index', range(3))
    cells.to_parquet(tmp_path/'cells.parquet')
    pd.DataFrame({'gene_index': [0, 1], 'symbol': ['g1', 'g2'],
                  'source_genes': ['g1|old_g1', 'g2'], 'total_counts': [23, 6]}).to_parquet(tmp_path/'genes.parquet')
    build(tmp_path, source)
    with np.load(tmp_path/'pseudobulk.npz', allow_pickle=False) as data:
        np.testing.assert_array_equal(data['counts'], [[7, 5], [16, 1]])
        np.testing.assert_array_equal(data['labels'], ['A', 'B'])
        np.testing.assert_array_equal(data['n_bins'], [1, 2])
    cells['source_obs_id'] = ['c1', 'c0', 'c2']
    cells.to_parquet(tmp_path/'cells.parquet')
    with pytest.raises(AssertionError):
        build(tmp_path, source)
    manifest['source']['sha256'] = 'wrong'
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='checksum'):
        build(tmp_path, source)
