import urllib.request, pandas as pd, numpy as np, h5py
from anndata.io import read_elem
p='/content/fetal/'
u='https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt'
urllib.request.urlretrieve(u,p+'hgnc_complete_set.tsv')
h=pd.read_csv(p+'hgnc_complete_set.tsv',sep='\t',low_memory=False)
print('HGNC',h.shape,h.locus_group.value_counts().to_dict())
with h5py.File(p+'source.h5ad') as f:
 v=read_elem(f['var']); o=read_elem(f['obs'])
 pc=h[(h.status=='Approved')&(h.locus_group=='protein-coding gene')]
 print('direct_matches',v.index.isin(pc.symbol).sum())
 print('missing_examples',v.index[~v.index.isin(pc.symbol)][:100].tolist())
 print('counts data',f['layers/counts/data'][:30]); print('X data',f['X/data'][:30])
 print('spatial_equals',np.array_equal(o[['x','y']].values,read_elem(f['obsm/spatial'])))
 print('bounds',o[['x','y']].agg(['min','max']).to_dict()); print('types',o.celltype.value_counts().to_dict())
 print('counts shape',dict(f['layers/counts'].attrs))
