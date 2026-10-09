import numpy as np
import pandas as pd
from spatial_app.view import spatial_figure


def test_arbitrary_rotation_preserves_values_and_counterclockwise_direction():
    from spatial_app.view import rotated_grid_indices
    np.testing.assert_array_equal(rotated_grid_indices((2,3),90),np.rot90(np.arange(6).reshape(2,3),-1))
    cells=pd.DataFrame({'x':[0,50,0,50],'y':[0,0,50,50],'celltype':['A','B','C','D']})
    scores=np.array([.1,.2,.3,.4])
    fig=spatial_figure(cells,scores,'Gene-set AUROC',rotation_degrees=70)
    values=np.asarray(fig.data[1].z)
    assert set(values[np.isfinite(values)])<=set(scores)
    assert fig.layout.uirevision.endswith('-70')


def test_counterclockwise_rotation_preserves_bin_values_and_source():
    cells=pd.DataFrame({'x':[0,50,0],'y':[0,0,50],'celltype':['A','B','C']})
    original=cells.copy()
    fig=spatial_figure(cells,np.array([.1,.2,.3]),'Gene-set AUROC',rotate_ccw=True)
    np.testing.assert_array_equal(fig.data[1].x,[-50,0])
    np.testing.assert_array_equal(fig.data[1].y,[0,50])
    np.testing.assert_array_equal(fig.data[1].z,[[.3,.1],[np.nan,.2]])
    pd.testing.assert_frame_equal(cells,original)


def test_tissue_auroc_uses_group_values_and_its_own_scale():
    cells=pd.DataFrame({'x':[0,50,100],'y':[0,0,0],'celltype':['A','B','A']})
    scores=np.array([.1,.9,.4])
    fig=spatial_figure(cells,scores,'Tissue AUROC',type_aurocs={'A':.45,'B':.55})
    np.testing.assert_array_equal(fig.data[1].z,[[.45,.55,.45]])
    assert fig.data[1].zmin==.45 and fig.data[1].zmax==.55
    assert fig.data[1].colorbar.title.text=='Tissue AUROC'


def test_preview_fills_only_occupied_unscored_bins_without_changing_scores():
    cells=pd.DataFrame({'x':[0,50,150],'y':[0,0,0],'celltype':['A','B','C']})
    scores=np.array([.2,np.nan,.8])
    fig=spatial_figure(cells,scores,'Gene-set AUROC',preview=True)
    np.testing.assert_array_equal(fig.data[1].z,[[.2,.2,np.nan,.8]])
    assert np.isnan(scores[1])
    assert 'Preview' in fig.data[1].customdata[0,1]
    empty=spatial_figure(cells,np.full(3,np.nan),'Gene-set AUROC',preview=True)
    assert np.isnan(empty.data[1].z).all()
    scores[1]=.6
    np.testing.assert_array_equal(spatial_figure(cells,scores,'Gene-set AUROC',preview=True).data[1].z,[[.2,.6,np.nan,.8]])


def test_tiles_preserve_missing_positions_and_mode_geometry():
    cells=pd.DataFrame({'x':[0,50,0],'y':[0,0,50],'celltype':['A','B','A']})
    scores=np.array([0.2,np.nan,0.8])
    numeric=spatial_figure(cells,scores,'Gene-set AUROC')
    categorical=spatial_figure(cells,scores,'Cell type')
    np.testing.assert_array_equal(numeric.data[0].z,[[0,0],[0,np.nan]])
    np.testing.assert_array_equal(numeric.data[1].z,[[0.2,np.nan],[0.8,np.nan]])
    assert categorical.layout.margin==numeric.layout.margin
    assert categorical.layout.xaxis==numeric.layout.xaxis
    assert categorical.layout.yaxis==numeric.layout.yaxis
    assert categorical.layout.uirevision==numeric.layout.uirevision
    assert all(t.xgap==0 and t.ygap==0 and not t.hoverongaps for t in numeric.data)
