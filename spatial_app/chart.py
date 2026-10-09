"""Stable local Plotly component; colour changes do not recreate its iframe."""
from pathlib import Path
import json
from plotly.utils import PlotlyJSONEncoder
import streamlit.components.v1 as components

_component=components.declare_component('spatial_tiles',path=str(Path(__file__).with_name('tile_component')))


def tile_chart(figure,filename,key='spatial_plot'):
    _component(figure=json.dumps(figure.to_dict(),cls=PlotlyJSONEncoder),filename=filename,key=key,default=None)
