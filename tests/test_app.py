from pathlib import Path
import json
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest


@pytest.fixture(autouse=True)
def small_default(monkeypatch):
    monkeypatch.setenv('FETAL_APP_DATA',str(Path(__file__).resolve().parents[1]/'data/GSM9629296'))


@pytest.mark.parametrize('dataset',['GSM9629326','GSM9629338'])
def test_new_midstage_slices_complete(dataset):
    root=Path(__file__).resolve().parents[1]
    manifest=json.loads((root/'data'/dataset/'manifest.json').read_text())
    app=AppTest.from_file(str(root/'app.py'),default_timeout=60).run()
    app.selectbox(key='selected_slice').set_value(str(root/'data'/dataset)).run()
    assert not app.exception
    assert len(app.session_state['scores'])==manifest['n_cells']
    first=next(f for f in manifest['files'] if f['path']==manifest['progressive'][0])
    assert np.isfinite(app.session_state['scores']).sum()==first['n_cells']
    assert app.session_state['job']['query'].universe_size==manifest['n_genes']
    for _ in range(len(manifest['progressive'])-1):
        app.run()
        assert not app.exception
    assert np.isfinite(app.session_state['scores']).all()
    assert not app.session_state['job']['running']
    assert len(app.dataframe[0].value)==manifest['n_celltypes']


def test_cs23_slice_loads_and_scores_first_spatial_batch():
    root=Path(__file__).resolve().parents[1]
    app=AppTest.from_file(str(root/'app.py'),default_timeout=60).run()
    app.selectbox(key='selected_slice').set_value(str(root/'data/GSM9629365')).run()
    assert not app.exception
    assert len(app.session_state['scores'])==825115
    assert np.isfinite(app.session_state['scores']).sum()==32768
    assert app.session_state['job']['query'].universe_size==18646
    assert not app.dataframe
    assert app.session_state['enrichment_summary'] is None


def test_cs12_e2s7_loads_and_completes():
    root=Path(__file__).resolve().parents[1]
    app=AppTest.from_file(str(root/'app.py'),default_timeout=30).run()
    app.selectbox(key='selected_slice').set_value(str(root/'data/GSM9629300')).run()
    assert not app.exception
    assert len(app.session_state['scores'])==9603
    assert np.isfinite(app.session_state['scores']).sum()==8192
    assert app.session_state['job']['query'].universe_size==17189
    app.run()
    assert not app.exception
    assert np.isfinite(app.session_state['scores']).all()
    assert len(app.dataframe[0].value)==16


def test_slice_switch_restarts_scoring_and_preserves_inputs():
    root=Path(__file__).resolve().parents[1]
    app=AppTest.from_file(str(root/'app.py'),default_timeout=30).run()
    app.text_area(key='target_input').set_value('SOX2 PAX6').run()
    app.selectbox(key='selected_slice').set_value(str(root/'data/GSM9629349')).run()
    assert not app.exception
    assert len(app.session_state['scores'])==265988
    assert np.isfinite(app.session_state['scores']).sum()==32768
    assert app.session_state['job']['next_batch']==1
    assert app.session_state['job']['query'].universe_size==18622
    assert app.text_area(key='target_input').value=='SOX2 PAX6'
    assert not app.dataframe
    app.button(key='cycle_colour').click().run()
    assert app.session_state['colour_mode']=='Cell type'
    app.button(key='cycle_colour').click().run()
    assert app.session_state['colour_mode']=='Gene-set AUROC'
    app.selectbox(key='selected_slice').set_value(str(root/'data/GSM9629296')).run()
    assert not app.exception
    assert len(app.session_state['scores'])==13257
    assert np.isfinite(app.session_state['scores']).sum()==8192
    assert app.session_state['job']['query'].universe_size==17799


def test_progressive_app_and_background_reset():
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py'),default_timeout=30).run()
    assert not app.exception
    assert len(app.session_state['scores'])==13257
    assert not app.dataframe
    assert not any(e.label=='Results & scoring details' for e in app.expander)
    assert not any(b.label=='Run AUROC' for b in app.button)
    assert np.isfinite(app.session_state['scores']).sum()==8192
    for _ in range(1): app.run()
    assert not app.exception
    assert np.isfinite(app.session_state['scores']).all()
    assert not app.session_state['job']['running']
    table=app.dataframe[0].value
    assert len(table)==19
    assert list(table.columns)==['Cell type','Scored bins','AUROC','p-value','Corrected p-value (BH)']
    # A new background discards ALL old scores and starts the new universe.
    app.text_area(key='background_input').set_value('SOX2 PAX6 ACTB GAPDH').run()
    assert not app.exception
    assert app.session_state['job']['query'].universe_size==4
    assert len(app.session_state['job']['query'].target)==2
    assert np.isfinite(app.session_state['scores']).sum()==8192
    app.button(key='cycle_colour').click().run()
    assert not app.exception
    assert app.session_state['job']['next_batch']==2
    finished=app.session_state['scores'].copy()
    app.button(key='cycle_colour').click().run()
    assert app.session_state['colour_mode']=='Tissue AUROC'
    app.button(key='cycle_colour').click().run()
    assert app.session_state['colour_mode']=='Gene-set AUROC'
    np.testing.assert_array_equal(app.session_state['scores'],finished)
    # Invalid query clears prior results instead of displaying stale scores.
    app.text_area(key='target_input').set_value('NOT_A_GENE').run()
    assert not app.exception
    assert app.session_state['job'] is None
    assert np.isnan(app.session_state['scores']).all()
    assert len(app.error)>0
    # Species selection maps both inputs and restarts scoring automatically.
    app.text_area(key='target_input').set_value('Sox2 Pax6')
    app.text_area(key='background_input').set_value('Sox2 Pax6 Actb Gapdh')
    app.selectbox(key='input_species').set_value('Mouse').run()
    assert not app.exception
    assert app.session_state['job']['query'].target_symbols == ('PAX6','SOX2')
    assert app.session_state['job']['query'].universe_size == 4
    app.text_area(key='background_input').set_value('NOT_A_GENE').run()
    assert not app.exception
    assert app.session_state['job'] is None
    assert np.isnan(app.session_state['scores']).all()


def test_enrichment_only_once_per_completed_query(monkeypatch):
    import spatial_app.enrichment as enrichment
    original=enrichment.celltype_enrichment
    calls=[]
    def counted(labels,scores):
        assert np.isfinite(scores).all()
        calls.append(1)
        return original(labels,scores)
    monkeypatch.setattr(enrichment,'celltype_enrichment',counted)
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py'),default_timeout=30).run()
    assert not calls
    app.run()
    assert not app.exception
    assert len(calls)==1
    app.button(key='cycle_colour').click().run()
    app.run()
    assert len(calls)==1
    app.text_area(key='target_input').set_value('PAX6 DCX').run()
    assert app.session_state['enrichment_summary'] is None
    assert len(calls)==1
    app.run()
    assert not app.exception
    assert len(calls)==2


def test_plot_updates_after_each_batch_and_retains_results_for_invalid_input():
    root=Path(__file__).resolve().parents[1]
    app=AppTest.from_file(str(root/'app.py'),default_timeout=30).run()
    assert np.isfinite(app.session_state['displayed_plot']['scores']).sum()==8192
    app.run()
    assert not app.exception
    assert np.isfinite(app.session_state['displayed_plot']['scores']).all()
    previous_signature=app.session_state['displayed_plot']['signature']
    app.text_area(key='target_input').set_value('PAX6').run()
    assert not app.exception
    assert app.session_state['job']['running']
    assert np.isfinite(app.session_state['displayed_plot']['scores']).sum()==8192
    assert app.session_state['displayed_plot']['signature']!=previous_signature
    assert app.session_state['enrichment_summary'] is None
    app.run()
    np.testing.assert_equal(app.session_state['displayed_plot']['scores'],app.session_state['scores'])
    latest=app.session_state['displayed_plot']['scores'].copy()
    app.text_area(key='target_input').set_value('NOT_A_GENE').run()
    np.testing.assert_equal(app.session_state['displayed_plot']['scores'],latest)
    app.text_area(key='target_input').set_value('PAX6')
    app.selectbox(key='selected_slice').set_value(str(root/'data/GSM9629365')).run()
    assert not app.exception
    assert app.session_state['displayed_plot']['dataset']=='GSM9629365'
    assert np.isfinite(app.session_state['displayed_plot']['scores']).sum()==32768
