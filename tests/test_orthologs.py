from spatial_app.orthologs import map_input, load_ortholog_map
from spatial_app.scoring import resolve_query


def test_real_mouse_mapping_and_background():
    targets, missing = map_input('Sox2, Pax6; Sox2 NOT_A_GENE', 'Mouse')
    assert targets == ['PAX6', 'SOX2']
    assert missing == ['NOT_A_GENE']
    background, missing = map_input('Sox2 Pax6 Actb Gapdh', 'Mouse')
    assert not missing
    query = resolve_query(' '.join(targets), ' '.join(background), ['SOX2','PAX6','ACTB','GAPDH'])
    assert query.universe_size == 4
    assert len(query.target) == 2


def test_reference_case_and_expansion_rules():
    assert map_input('SOX2', 'Mouse') == ([], ['SOX2'])
    assert map_input('', 'Mouse') == ([], [])
    mapping = load_ortholog_map()
    for species, sources in mapping.items():
        source = next(iter(sources))
        converted, missing = map_input(f'{source} {source}', species)
        assert set(converted) == sources[source]
        assert not missing
        for source, humans in sources.items():
            if len(humans) > 1:
                assert set(map_input(source, species)[0]) == humans
    assert map_input('SOX2 SOX2 PAX6', 'Human') == (['PAX6','SOX2'], [])
