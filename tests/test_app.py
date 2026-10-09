from pathlib import Path
import numpy as np
from streamlit.testing.v1 import AppTest


def test_cs23_slice_loads_and_scores_first_spatial_batch():
    root=Path(__file__).resolve().parents[1]
    app=AppTest.from_file(str(root/'app.py'),default_timeout=60).run()
    app.selectbox(key='selected_slice').set_value(str(root/'data/GSM9629365')).run()
    assert not app.exception
    assert len(app.session_state['scores'])==825115
    assert np.isfinite(app.session_state['scores']).sum()==32768
    assert app.session_state['job']['query'].universe_size==18646
    assert len(app.dataframe[0].value)==22


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
    assert len(app.dataframe[0].value)==21
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
    table=app.dataframe[0].value
    assert len(table)==19
    assert list(table.columns)==['Cell type','Scored bins','AUROC','p-value','Corrected p-value (BH)']
    assert not any(e.label=='Results & scoring details' for e in app.expander)
    assert not any(b.label=='Run AUROC' for b in app.button)
    assert np.isfinite(app.session_state['scores']).sum()==8192
    for _ in range(1): app.run()
    assert not app.exception
    assert np.isfinite(app.session_state['scores']).all()
    assert not app.session_state['job']['running']
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
