"""Stable local Plotly component; colour changes do not recreate its iframe."""
from pathlib import Path
import json
import time
from plotly.utils import PlotlyJSONEncoder
import streamlit.components.v1 as components

_component=components.declare_component('spatial_tiles',path=str(Path(__file__).with_name('tile_component')))


def tile_chart(figure,filename,key='spatial_plot'):
    started = time.perf_counter()
    print(f"[timing] chart={filename} serialize_start",flush=True)
    payload = json.dumps(figure.to_dict(),cls=PlotlyJSONEncoder)
    print(f"[timing] chart={filename} serialize_s={time.perf_counter()-started:.3f} "
          f"payload_mib={len(payload)/1024**2:.2f}",flush=True)
    started = time.perf_counter()
    _component(figure=payload,filename=filename,key=key,default=None)
    # This measures server enqueueing, not network transfer or browser painting.
    print(f"[timing] chart={filename} component_enqueue_s={time.perf_counter()-started:.3f}",flush=True)
