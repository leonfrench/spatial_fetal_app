"""Run with: .conda-env/bin/streamlit run app.py"""
from pathlib import Path
import json
import os
import time
import numpy as np
import pandas as pd
import streamlit as st

from scripts.rank_io import load_shard
from spatial_app.scoring import resolve_query, auroc_vectorized
from spatial_app.view import spatial_figure, score_limits
from spatial_app.chart import tile_chart
from spatial_app.enrichment import celltype_enrichment
from spatial_app.orthologs import SPECIES, map_input

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = Path(os.environ.get('FETAL_APP_DATA',str(ROOT/'data/GSM9629296'))).resolve()
SLICE_LABELS = {
    'GSM9629296':'CS12–13 · Embryo 2 · Section 3',
    'GSM9629349':'CS20 · Embryo 1 · Section 7',
    'GSM9629365':'CS23 · Embryo 2 · Section 8',
}

st.set_page_config(page_title='Spatial · Gene-set explorer',page_icon='◉',layout='wide',initial_sidebar_state='expanded')
st.markdown('''<style>
.block-container {padding-top:4rem; padding-bottom:1rem; max-width:1600px;}
[data-testid="stSidebar"] h2 {font-size:1.1rem;}
h1 {font-size:2rem !important; font-weight:650 !important; letter-spacing:-.04em;}
[data-testid="stMetricValue"] {font-size:1.55rem; font-weight:600;}
[data-testid="stMetricLabel"] {color:#69788a; font-size:.8rem;}
[data-testid="stSidebar"] .block-container {padding-top:1.5rem;}
.st-key-cycle_colour button {background:#187D92 !important; color:white !important;
 border:2px solid #126375 !important; min-height:3.4rem; box-shadow:0 2px 6px #187d9230;}
.st-key-cycle_colour button p {font-size:1.12rem !important; font-weight:700 !important;}
.st-key-cycle_colour button:hover {background:#126375 !important;}
</style>''',unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def metadata(path, version):
    root = Path(path)
    manifest = json.loads((root/'manifest.json').read_text())
    cells = pd.read_parquet(root/'cells.parquet').sort_values('cell_index').reset_index(drop=True)
    genes = pd.read_parquet(root/'genes.parquet').sort_values('gene_index').reset_index(drop=True)
    if not np.array_equal(cells.cell_index.to_numpy(),np.arange(len(cells))):
        raise ValueError('Cell indices must be consecutive and unique')
    if not np.array_equal(genes.gene_index.to_numpy(),np.arange(len(genes))):
        raise ValueError('Gene indices must be consecutive and unique')
    if len(cells)!=manifest['n_cells'] or len(genes)!=manifest['n_genes']:
        raise ValueError('Metadata does not match the manifest')
    return manifest,cells,genes


@st.cache_resource(max_entries=32,show_spinner=False)
def cached_shard(path, checksum):
    shard = load_shard(path)
    for value in shard.values():
        value.setflags(write=False)
    return shard


slice_paths = {str((ROOT/'data'/dataset).resolve()):label
               for dataset,label in SLICE_LABELS.items()
               if (ROOT/'data'/dataset/'manifest.json').exists()}
slice_paths.setdefault(str(DEFAULT_DATA),SLICE_LABELS.get(DEFAULT_DATA.name,DEFAULT_DATA.name))
selected_path = st.sidebar.selectbox('Slice',list(slice_paths),
    index=list(slice_paths).index(str(DEFAULT_DATA)),
    format_func=slice_paths.get,key='selected_slice',
    help='Switching slices retains your gene inputs and restarts scoring for the selected tissue section.')
DATA = Path(selected_path)

try:
    manifest,cells,genes = metadata(str(DATA),(DATA/'manifest.json').stat().st_mtime_ns)
except (OSError,ValueError,KeyError) as exc:
    st.error(f'Unable to load the prepared dataset: {exc}')
    st.stop()
paths = manifest['progressive']
checksums = {f['path']:f['sha256'] for f in manifest['files']}
version = f"{DATA}:{manifest['created_utc']}"
if st.session_state.get('dataset_version') != version:
    st.session_state.dataset_version = version
    st.session_state.scores = np.full(len(cells),np.nan)
    st.session_state.job = None
    st.session_state.query_error = None


def reset_scores():
    st.session_state.scores = np.full(len(cells),np.nan)
    st.session_state.job = None
    st.session_state.query_error = None


with st.sidebar:
    st.markdown('## Gene-set scoring')
    species = st.selectbox('Species of input genes',SPECIES,key='input_species',
        help='Applies to both target and background genes. Mouse and Rhesus Macaque symbols are mapped to human orthologs; use the source species’ symbol capitalization.')
    st.caption('Separate gene symbols with spaces, commas or new lines.')
    target_text = st.text_area('Target genes',value='SOX2\nPAX6\nNES\nDCX\nTUBB3',height=155,key='target_input',
        help='Scoring updates automatically when you finish editing and leave this field.')
    with st.expander('Background genes · optional'):
        background_text = st.text_area('Background universe',height=160,key='background_input',
            placeholder='Leave blank to use all retained genes.',
            help='The complete universe, including target and non-target genes. Targets outside it are excluded. Ranks are recomputed when you finish editing.')
        st.caption('Include your targets in this universe. Blank = all retained genes.')
    input_signature = (version,target_text,background_text,species)
    if st.session_state.get('input_signature') != input_signature:
        st.session_state.input_signature = input_signature
        reset_scores()
        st.session_state.mapping_unmapped = ([],[])
        try:
            targets,unmapped_targets = map_input(target_text,species)
            background,unmapped_background = map_input(background_text,species)
            st.session_state.mapping_unmapped = (unmapped_targets,unmapped_background)
            if not targets:
                raise ValueError('No target genes could be mapped to human symbols. Check the selected species and symbol capitalization.')
            if background_text.strip() and not background:
                raise ValueError('No background genes could be mapped to human symbols. Check the selected species and symbol capitalization, or clear the background to use all genes.')
            query = resolve_query(' '.join(targets),' '.join(background),genes.symbol)
            st.session_state.job = dict(query=query,next_batch=0,running=True,compute_seconds=0.0,load_seconds=0.0,
                                        target_text=target_text,background_text=background_text,error=None)
        except (ValueError,OSError) as exc:
            st.session_state.query_error = str(exc)
    st.session_state.setdefault('colour_mode','Gene-set AUROC')
    complete = np.isfinite(st.session_state.scores).all()
    if not complete and st.session_state.colour_mode == 'Tissue AUROC':
        st.session_state.colour_mode = 'Gene-set AUROC'
    def cycle_colour():
        modes = ['Gene-set AUROC','Cell type']
        if np.isfinite(st.session_state.scores).all():
            modes.append('Tissue AUROC')
        current = st.session_state.colour_mode
        st.session_state.colour_mode = modes[(modes.index(current)+1)%len(modes)]
    mode = st.session_state.colour_mode
    display_label = 'Tissue' if mode == 'Cell type' else mode
    st.button(f'↻ Switch colors · {display_label}',key='cycle_colour',on_click=cycle_colour,width='stretch',type='primary',
        help='Cycle through gene-set AUROC, tissue colours, and tissue AUROC. Tissue AUROC unlocks after all batches finish; its scale follows the tissue-level values.')
    if st.session_state.query_error:
        st.error(st.session_state.query_error)
    unmapped_targets,unmapped_background = st.session_state.get('mapping_unmapped',([],[]))
    if species != 'Human':
        st.caption(f'{species} → human orthologs (HomoloGene). All mapped orthologs are retained and deduplicated.')
    if unmapped_targets or unmapped_background:
        with st.expander('Genes without a human ortholog',expanded=False):
            if unmapped_targets: st.warning('Targets: '+', '.join(unmapped_targets))
            if unmapped_background: st.warning('Background: '+', '.join(unmapped_background))
    flip_y = False
    st.divider()
    st.caption(f"{manifest['dataset']} · {len(cells):,} observations\n\n{len(genes):,} protein-coding genes · {manifest['n_celltypes']} tissues")
    st.caption('Data source: Pan et al. (2026). '
        '[Spatiotemporal transcriptome atlas of human embryos after gastrulation]'
        '(https://doi.org/10.1038/s41586-026-10545-0). *Nature*, 654. '
        'doi:10.1038/s41586-026-10545-0.')

st.caption('SPATIAL TRANSCRIPTOMICS  /  HESTA')
sample_label = SLICE_LABELS.get(manifest['dataset'],manifest['dataset'])
st.markdown(f'**{sample_label}** &nbsp; · &nbsp; All observations stay visible as each batch is scored.')


@st.fragment(run_every=0.25 if st.session_state.job and st.session_state.job['running'] else None)
def render_explorer():
    job = st.session_state.job
    if job and job['running'] and job['next_batch'] < len(paths):
        path = paths[job['next_batch']]
        try:
            started = time.perf_counter()
            shard = cached_shard(str(DATA/path),checksums[path])
            job['load_seconds'] += time.perf_counter()-started
            started = time.perf_counter()
            q = job['query']
            batch_scores = auroc_vectorized(shard,q.target,len(genes),q.background)
            if not np.isfinite(batch_scores).all():
                raise ValueError('Scoring produced undefined values')
            st.session_state.scores[shard['cell_index']] = batch_scores
            job['compute_seconds'] += time.perf_counter()-started
            job['next_batch'] += 1
            if job['next_batch']==len(paths):
                job['running'] = False
                st.rerun()  # Stop the fragment timer and render the completed result.
        except (OSError,ValueError,KeyError) as exc:
            job['error'] = str(exc)
            job['running'] = False
            st.rerun()
    scores = st.session_state.scores
    count = int(np.isfinite(scores).sum())
    limits = score_limits(scores)
    cols = st.columns([1,1,1,1.25])
    cols[0].metric('OBSERVATIONS SHOWN',f'{len(cells):,}')
    cols[1].metric('SCORED',f'{count:,} / {len(cells):,}')
    cols[2].metric('TARGET / UNIVERSE',f"{len(job['query'].target):,} / {job['query'].universe_size:,}" if job else f'— / {len(genes):,}')
    cols[3].metric('OBSERVED AUROC RANGE',f'{limits[0]:.3f} – {limits[1]:.3f}' if limits else 'Not scored yet')
    if job:
        if job['error']:
            st.error(f"Scoring stopped: {job['error']}")
        label = ('Complete' if count==len(cells) else ('Scoring' if job['running'] else 'Paused'))
        st.progress(count/len(cells),text=f"{label} · {job['next_batch']} of {len(paths)} batches · vectorized scoring {job['compute_seconds']:.2f}s")
        if count < len(cells) and not job['error']:
            if st.button('Pause scoring' if job['running'] else 'Resume scoring',key='pause_resume'):
                job['running'] = not job['running']
                st.rerun()
        q=job['query']
        active=', '.join(q.target_symbols[:12]) + (' …' if len(q.target_symbols)>12 else '')
        st.caption(f"Active targets: {active}. " + ('Ranks recomputed within your background universe.' if q.background is not None else 'Universe: all retained protein-coding genes.'))
        if q.unknown_target or q.unknown_background or q.outside_background:
            with st.expander(f'Gene matching · {len(q.unknown_target)+len(q.outside_background)} target symbols excluded',expanded=False):
                if q.unknown_target: st.warning('Targets not in dataset: '+', '.join(q.unknown_target))
                if q.unknown_background: st.warning('Background genes not in dataset: '+', '.join(q.unknown_background))
                if q.outside_background: st.warning('Targets outside background: '+', '.join(q.outside_background))
    else:
        st.info('All bins are shown. Enter a valid target gene set to calculate AUROC automatically, or switch to tissue colours.')
    summary=celltype_enrichment(cells.celltype.astype(str),scores)
    type_aurocs=summary.set_index('Cell type')['AUROC'].to_dict()
    spatial_preview=manifest.get('sampling',{}).get('method')=='spatial'
    fig=spatial_figure(cells,scores,mode,flip_y,version,type_aurocs=type_aurocs,preview=spatial_preview,
                       rotate_ccw=manifest['dataset']=='GSM9629349',
                       rotation_degrees=70 if manifest['dataset']=='GSM9629365' else 0)
    tile_chart(fig,filename=f"{manifest['dataset']}-spatial")
    if mode=='Gene-set AUROC':
        if spatial_preview:
            st.caption('Unscored bins temporarily show the nearest scored bin’s colour; the preview gains detail as batches finish. Hover identifies preview values. Statistics and downloads use actual scores only. Colour limits follow scored AUROCs. Drag to pan, scroll to zoom, double-click to reset.')
        else:
            st.caption('Grey = not yet scored. Colour limits update to the minimum and maximum scored AUROC. 0.5 indicates equal target/background expression ordering; higher values favour the targets. Drag to pan, scroll to zoom, double-click to reset.')
    elif mode=='Tissue AUROC':
        st.caption('Each bin shows its tissue’s enrichment AUROC from the completed table. The colour scale spans the tissue AUROC minimum and maximum, independently of bin-level scores.')
    else:
        st.caption('Source tissue annotations. Each tile fills one spatial bin. Drag to pan, scroll to zoom.')
    slice_total=len(cells)*len(genes)
    slice_zeros=slice_total-int(genes.n_cells_detected.sum())
    if job:
        target_indices=list(job['query'].target)
        target_total=len(cells)*len(target_indices)
        target_zeros=target_total-int(genes.iloc[target_indices].n_cells_detected.sum())
        st.caption(f'For expression sparsity, the matched target genes in your scoring universe are {target_zeros/target_total:.2%} zero (not detected), versus {slice_zeros/slice_total:.2%} across all retained genes in the full slice, independent of the background input; both percentages include all bins.')
    else:
        st.caption(f'For expression sparsity, {slice_zeros/slice_total:.2%} of entries across all bins and retained genes in the full slice are zero (not detected), independent of the background input.')
    st.subheader('Tissue enrichment')
    st.caption('Ranks bins by their gene-set AUROC, then compares each tissue with all other tissues. AUROC > 0.5 = higher-ranked; < 0.5 = lower-ranked; 0.5 = no ranking preference.')
    if count < len(cells):
        st.caption(f'Provisional results from {count:,} scored bins; the table updates as batches finish.')
    st.dataframe(summary,width='stretch',hide_index=True,height=(len(summary)+1)*35+3,
        column_config={
            'Cell type':st.column_config.TextColumn('Tissue'),
            'AUROC':st.column_config.NumberColumn(format='%.4f',help='Probability that a bin of this tissue scores higher than a bin of another tissue, with half credit for ties.'),
            'p-value':st.column_config.NumberColumn(format='%.3e',help='Two-sided Mann–Whitney U test, with tie and continuity correction.'),
            'Corrected p-value (BH)':st.column_config.NumberColumn(format='%.3e',help='Benjamini–Hochberg correction across all annotated tissues in this query.')})
    st.caption('Two-sided Mann–Whitney p-values; Benjamini–Hochberg correction across tissues for this query. Values that underflow to zero are reported at the smallest positive float (≈4.94 × 10⁻³²⁴). Bin-level tests assume independence; spatial correlation can make these p-values optimistic.')
    if count and not (job and job['running']):
        st.download_button('Download enrichment table',summary.to_csv(index=False).encode(),
                           file_name=f"{manifest['dataset']}-celltype-enrichment.csv",mime='text/csv',key='download_enrichment')
        export=cells[['cell_index','source_obs_id','x','y','celltype']].copy()
        export['auroc']=scores
        st.download_button('Download bin scores',export.to_csv(index=False).encode(),
                           file_name=f"{manifest['dataset']}-auroc.csv",mime='text/csv',key='download_scores')



render_explorer()
