from pathlib import Path
import urllib.request, json, h5py
p=Path('/content/fetal'); p.mkdir(exist_ok=True)
u='https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9629nnn/GSM9629296/suppl/GSM9629296_CS12-13_E2S3_HESTA.h5ad'
f=p/'source.h5ad'
if not f.exists(): urllib.request.urlretrieve(u,f)
print('download_bytes',f.stat().st_size)
with h5py.File(f) as h:
 def show(name,obj):
  if name.count('/')<2: print(name, getattr(obj,'shape',''), dict(obj.attrs))
 h.visititems(show)
import anndata as ad
adata=ad.read_h5ad(f, backed='r')
print(adata)
print('VAR',adata.var.head().to_string(),adata.var.describe(include='all').to_string())
print('OBS',adata.obs.head().to_string())
print('OBSM',[(k,v.shape) for k,v in adata.obsm.items()])
print('X sample',adata.X[:3,:20].toarray() if hasattr(adata.X[:3,:20],'toarray') else adata.X[:3,:20])
