import numpy as np
from scripts.spatial_order import spatial_order


def test_nested_coverage_with_holes_and_deterministic_order():
    coords=np.indices((35,41)).reshape(2,-1).T
    coords=coords[(coords[:,0]<12)|(coords[:,1]>20)]*50+100
    order,levels=spatial_order(coords)
    np.testing.assert_array_equal(np.sort(order),np.arange(len(coords)))
    np.testing.assert_array_equal(order,spatial_order(coords)[0])
    grid=(coords-coords.min(axis=0))//50
    for level in levels:
        width=level['grid_width']
        selected=order[:level['cumulative']]
        assert set(map(tuple,grid[selected]//width))==set(map(tuple,grid//width))
        # Exactly one scored representative for each occupied block at each level.
        assert len(selected)==len(np.unique(grid//width,axis=0))
