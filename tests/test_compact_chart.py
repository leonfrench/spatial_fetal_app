import base64
import numpy as np
import pandas as pd
from spatial_app.chart import geometry_payload, score_patch


def decode(text, dtype):
    return np.frombuffer(base64.b64decode(text), dtype=dtype)


def test_patch_roundtrip_including_query_reset_and_unchanged_nans():
    before = np.array([np.nan, .2, .7, np.nan])
    after = np.array([.9, np.nan, .7, np.nan])
    patch = score_patch(before, after)
    ids = decode(patch['indices'], '<u4')
    assert ids.tolist() == [0, 1]
    result = before.copy()
    result[ids] = decode(patch['values'], '<f8')
    np.testing.assert_equal(result, after)
    initial = score_patch(None, after)
    result = np.full(4, np.nan)
    result[decode(initial['indices'], '<u4')] = decode(initial['values'], '<f8')
    np.testing.assert_equal(result, after)


def test_geometry_preserves_original_bin_indices_gaps_and_labels():
    cells = pd.DataFrame(dict(x=[0, 50, 0], y=[0, 0, 50], celltype=['B', 'A', 'B']))
    g = geometry_payload(cells, 70)
    assert (g['width'], g['height'], g['rotation']) == (2, 2, 70)
    assert decode(g['positions'], '<u4').tolist() == [0, 1, 2]
    assert [g['labels'][i] for i in decode(g['categories'], '<u2')] == ['B', 'A', 'B']


def test_delayed_ack_sends_only_new_scores_and_remount_recovers(monkeypatch):
    import spatial_app.chart as chart
    session = {}
    sent = []
    monkeypatch.setattr(chart.st, 'session_state', session)
    monkeypatch.setattr(chart, '_component', lambda **kwargs: sent.append(kwargs))
    cells = pd.DataFrame(dict(x=[0, 50, 100, 150], y=[0]*4, celltype=['A']*4))
    args = dict(version='slice-1', mode='Gene-set AUROC', type_aurocs={}, filename='test')
    # Establish the browser before progressive score delivery.
    session['spatial_plot'] = dict(client='browser', version=None, seq=-1)
    chart.tile_chart(cells, np.array([.1, np.nan, np.nan, np.nan]), **args)
    first = sent[-1]['packet']['seq']
    chart.tile_chart(cells, np.array([.1, .2, np.nan, np.nan]), **args)
    session['spatial_plot'] = dict(client='browser', version='slice-1', seq=first)
    chart.tile_chart(cells, np.array([.1, .2, .3, np.nan]), **args)
    assert sent[-1]['geometry'] is None
    assert not sent[-1]['packet']['reset']
    assert decode(sent[-1]['packet']['indices'], '<u4').tolist() == [1, 2]
    # A replaced iframe gets geometry and a full numeric snapshot.
    session['spatial_plot'] = dict(client='new-browser', version=None, seq=-1)
    chart.tile_chart(cells, np.array([.1, .2, .3, np.nan]), **args)
    assert sent[-1]['geometry'] is not None
    assert sent[-1]['packet']['reset']
    assert decode(sent[-1]['packet']['indices'], '<u4').tolist() == [0, 1, 2]
