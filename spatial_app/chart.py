"""Compact, acknowledged score updates for the resolution-aware canvas viewer."""
from pathlib import Path
import base64
import time
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from spatial_app.view import grid_geometry, tissue_color

_component = components.declare_component('spatial_tiles', path=str(Path(__file__).with_name('tile_component')))


def encode_array(values, dtype):
    return base64.b64encode(np.asarray(values, dtype=dtype).tobytes()).decode('ascii')


def geometry_payload(cells, rotation):
    (x, y), (ix, iy), (dx, dy) = grid_geometry(cells)
    labels, codes = np.unique(cells.celltype.astype(str).to_numpy(), return_inverse=True)
    return dict(width=len(x), height=len(y), dx=dx, dy=dy,
                x0=int(x[0]), y0=int(y[0]), rotation=rotation,
                positions=encode_array(iy * len(x) + ix, '<u4'),
                categories=encode_array(codes, '<u2'),
                labels=labels.tolist(), colors=[tissue_color(s) for s in labels])


def score_patch(previous, scores):
    if previous is None:
        indices = np.flatnonzero(np.isfinite(scores))
    else:
        indices = np.flatnonzero(~((previous == scores) | (np.isnan(previous) & np.isnan(scores))))
    return dict(indices=encode_array(indices, '<u4'), values=encode_array(scores[indices], '<f8'))


def tile_chart(cells, scores, *, version, mode, type_aurocs, filename, rotation=0, key='spatial_plot'):
    started = time.perf_counter()
    state_key = key + '_transport'
    ack = st.session_state.get(key) or {}
    state = st.session_state.get(state_key)
    if state is None or state['version'] != version or 'history' not in state:
        state = dict(version=version, geometry=geometry_payload(cells, rotation), seq=0,
                     pending=None, baseline=None, client=None, history={})
        st.session_state[state_key] = state
    client = ack.get('client')
    if client != state['client']:
        state.update(client=client, baseline=None, pending=None, history={})
    pending = state['pending']
    acknowledged = state['history'].get(ack.get('seq')) if ack.get('version') == version else None
    if acknowledged is not None:
        state['baseline'] = acknowledged
    # Keep one pending snapshot; resend it until acknowledged. A missed update
    # is recovered by an absolute patch from the last acknowledged baseline.
    if pending is None or not np.array_equal(pending['scores'], scores, equal_nan=True):
        state['seq'] += 1
        packet = score_patch(state['baseline'], scores)
        pending = dict(seq=state['seq'], scores=scores.copy(), packet=packet,
                       reset=state['baseline'] is None)
        state['pending'] = pending
        state['history'][pending['seq']] = pending['scores']
        while len(state['history']) > 4:
            del state['history'][next(iter(state['history']))]
    geometry = None if ack.get('version') == version and client else state['geometry']
    packet = dict(version=version, seq=pending['seq'], reset=pending['reset'], **pending['packet'])
    aucs = {name: float(value) for name, value in type_aurocs.items() if np.isfinite(value)}
    payload_bytes = sum(len(packet[k]) for k in ('indices', 'values'))
    if geometry:
        payload_bytes += len(geometry['positions']) + len(geometry['categories'])
    _component(geometry=geometry, packet=packet, mode=mode, type_aurocs=aucs,
               filename=filename, key=key, default=None)
    print(f"[timing] chart={filename} compact_update_s={time.perf_counter()-started:.3f} "
          f"payload_mib={payload_bytes/1024**2:.3f} geometry_sent={geometry is not None}", flush=True)
