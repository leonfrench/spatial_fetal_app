"""Nested, spatially balanced sampling of occupied grid bins."""
import numpy as np


def spatial_order(coords, seed=20261008):
    coords=np.asarray(coords,dtype=np.int64)
    steps=np.array([np.gcd.reduce(np.diff(np.unique(coords[:,i])))
                    if len(np.unique(coords[:,i]))>1 else 1 for i in range(2)])
    grid=(coords-coords.min(axis=0))//steps
    if len(np.unique(grid,axis=0))!=len(grid):
        raise ValueError('Spatial sampling requires unique grid coordinates')
    selected=np.zeros(len(grid),dtype=bool)
    rng=np.random.default_rng(seed)
    order=[]; levels=[]
    for width in [16,8,4,2,1]:
        tiles=grid//width
        _,groups=np.unique(tiles,axis=0,return_inverse=True)
        represented=np.zeros(groups.max()+1,dtype=bool)
        represented[groups[selected]]=True
        candidates=np.flatnonzero(~represented[groups])
        centers=tiles[candidates]*width+(width-1)/2
        distance=((grid[candidates]-centers)**2).sum(axis=1)
        sorted_rows=candidates[np.lexsort((candidates,distance,groups[candidates]))]
        first=np.r_[True,np.diff(groups[sorted_rows])!=0] if len(sorted_rows) else np.array([],dtype=bool)
        additions=rng.permutation(sorted_rows[first])
        selected[additions]=True
        order.extend(additions.tolist())
        levels.append(dict(grid_width=width,added=len(additions),cumulative=len(order)))
    assert selected.all() and len(order)==len(grid)
    return np.asarray(order,dtype=np.int64),levels
