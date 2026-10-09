"""Spatial grid tiles with a stable viewport across colour modes."""
import numpy as np
import plotly.graph_objects as go
import json
from pathlib import Path
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import rotate
from functools import lru_cache


@lru_cache(maxsize=4)
def rotated_grid_indices(shape, angle):
    # Array rows increase with Cartesian y here, unlike screen-image rows.
    return rotate(np.arange(np.prod(shape)).reshape(shape),-angle,
                  reshape=True,order=0,mode='constant',cval=-1,prefilter=False)

# Swatch interiors from the user-supplied manuscript legend, converted from
# its embedded display profile to sRGB for browser CSS colors.
TISSUE_COLORS = json.loads(Path(__file__).with_name('tissue_colors.json').read_text())


def tissue_color(name):
    return TISSUE_COLORS.get(str(name).strip().casefold(), '#B0B0B0')


SCALE = [[0,'#32265C'],[0.25,'#315B91'],[0.5,'#23958B'],[0.75,'#7ECF8D'],[1,'#F3DD65']]


def score_limits(scores):
    values = scores[np.isfinite(scores)]
    if not len(values):
        return None
    return float(values.min()), float(values.max())


def grid_geometry(cells):
    """Recover the source lattice, retaining empty positions as actual gaps."""
    axes=[]; indices=[]; steps=[]
    for name in ['x','y']:
        values=cells[name].to_numpy(dtype=np.int64)
        unique=np.unique(values)
        step=int(np.gcd.reduce(np.diff(unique))) if len(unique)>1 else 50
        axes.append(np.arange(unique[0],unique[-1]+step,step))
        indices.append((values-unique[0])//step)
        steps.append(step)
    return axes,indices,steps


def spatial_figure(cells, scores, mode, flip_y=False, view_revision='small-slice', type_aurocs=None, preview=False, rotate_ccw=False, rotation_degrees=0):
    if rotate_ccw:
        # Display transform only: counterclockwise in Cartesian plot coordinates.
        cells=cells.assign(x=-cells.y.to_numpy(),y=cells.x.to_numpy())
    (x,y),(ix,iy),(dx,dy)=grid_geometry(cells)
    shape=(len(y),len(x))
    labels=cells.celltype.astype(str).to_numpy()
    hover=np.full(shape,'',dtype=object); hover[iy,ix]=labels
    occupied=np.full(shape,np.nan); occupied[iy,ix]=0
    values=np.full(shape,np.nan)
    colourbar=dict(thickness=12,len=0.9,x=1.03,xanchor='left',y=0.5,
                   outlinewidth=0,tickfont=dict(size=11))
    category=mode=='Cell type'
    if category:
        names=sorted(set(labels)); lookup={s:i for i,s in enumerate(names)}
        values[iy,ix]=[lookup[s] for s in labels]
        scale=[]
        for i in range(len(names)):
            color=tissue_color(names[i])
            scale.extend([[i/len(names),color],[(i+1)/len(names),color]])
        low,high=-0.5,len(names)-0.5
        type_aurocs=type_aurocs or {}
        legend_labels=[]
        for name in names:
            auc=type_aurocs.get(name,np.nan)
            value=f'{auc:.2f}'.rstrip('0').rstrip('.') if np.isfinite(auc) else '—'
            legend_labels.append(f'{name} ({value})')
        colourbar.update(title=dict(text='Tissue · AUROC'),tickvals=list(range(len(names))),ticktext=legend_labels)
    else:
        tissue_mode=mode=='Tissue AUROC'
        display_scores=(np.array([(type_aurocs or {}).get(label,np.nan) for label in labels])
                        if tissue_mode else scores)
        values[iy,ix]=display_scores
        if preview and not tissue_mode and np.isfinite(scores).any() and not np.isfinite(scores).all():
            # Display-only nearest-neighbour fill. Preserve tissue gaps and every
            # measured score; the caller's scores remain NaN for untested bins.
            nearest=distance_transform_edt(~np.isfinite(values),sampling=(dy,dx),
                return_distances=False,return_indices=True)
            missing=~np.isfinite(scores)
            my,mx=iy[missing],ix[missing]
            values[my,mx]=values[nearest[0,my,mx],nearest[1,my,mx]]
            hover[my,mx]=[f'{label}<br>Preview from nearest scored bin' for label in labels[missing]]
        scale=SCALE
        limits=score_limits(display_scores)
        low,high=limits if limits else (0,1)
        if low==high: low,high=max(0,low-0.005),min(1,high+0.005)
        colourbar.update(title=dict(text='Tissue AUROC' if tissue_mode else 'AUROC'),tickformat='.3f')
    if rotation_degrees:
        if dx!=dy:
            raise ValueError('Display rotation requires square source bins')
        source=rotated_grid_indices(shape,rotation_degrees)
        valid=source>=0
        def remap(array,fill):
            result=np.full(source.shape,fill,dtype=array.dtype)
            result[valid]=array.ravel()[source[valid]]
            return result
        # Nearest-neighbour display resampling preserves actual AUROC values and
        # annotation labels instead of blending them. Source bins are unchanged.
        occupied=remap(occupied,np.nan)
        values=remap(values,np.nan)
        hover=remap(hover,'')
        x=(np.arange(source.shape[1])-(source.shape[1]-1)/2)*dx
        y=(np.arange(source.shape[0])-(source.shape[0]-1)/2)*dy
    common=dict(x=x,y=y,customdata=hover,xgap=0,ygap=0,zsmooth=False,
                hoverongaps=False,connectgaps=False)
    fig=go.Figure([
        go.Heatmap(uid='bin-footprint',z=occupied,colorscale=[[0,'#D4DCE4'],[1,'#D4DCE4']],
                   zmin=0,zmax=1,showscale=False,hovertemplate='%{customdata}<extra></extra>',**common),
        go.Heatmap(uid='bin-colours',z=values,colorscale=scale,zmin=low,zmax=high,
                   showscale=bool(category or np.isfinite(scores).any()),colorbar=colourbar,
                   hovertemplate=('%{customdata}<extra></extra>' if category else '%{customdata}<br>'+('Tissue AUROC' if mode=='Tissue AUROC' else 'AUROC')+' %{z:.4f}<extra></extra>'),**common)
    ])
    xpad=(x[-1]-x[0]+dx)*0.03; ypad=(y[-1]-y[0]+dy)*0.03
    xr=[float(x[0]-dx/2-xpad),float(x[-1]+dx/2+xpad)]
    yr=[float(y[0]-dy/2-ypad),float(y[-1]+dy/2+ypad)]
    if flip_y: yr=yr[::-1]
    fig.update_layout(height=680,
        # Reserve identical colour-key space in BOTH modes. Plotly must never
        # shrink the tissue panel to accommodate longer categorical labels.
        margin=dict(l=12,r=280,t=8,b=12,autoexpand=False),showlegend=False,
        paper_bgcolor='#FFFFFF',plot_bgcolor='#FFFFFF',
        font=dict(family='Arial, sans-serif',color='#43536A',size=12),
        dragmode='pan',uirevision=f'{view_revision}-{flip_y}-{rotate_ccw}-{rotation_degrees}',
        hoverlabel=dict(bgcolor='white',font_color='#203244'),
        xaxis=dict(visible=False,range=xr,domain=[0,1],automargin=False,
                   constrain='domain',showgrid=False,zeroline=False),
        yaxis=dict(visible=False,range=yr,domain=[0,1],automargin=False,
                   scaleanchor='x',scaleratio=1,showgrid=False,zeroline=False))
    return fig
