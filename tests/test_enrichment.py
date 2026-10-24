import numpy as np
import pytest
from scipy.stats import mannwhitneyu, false_discovery_control
from spatial_app.enrichment import celltype_enrichment, specificity_z


COUNTS = np.array([[90, 40, 0, 1, 5, 3], [2, 3, 80, 40, 5, 1], [10, 15, 5, 6, 5, 9]])


def test_pseudobulk_matches_independent_normalization_and_gene_test():
    x = np.log2(COUNTS/COUNTS.sum(axis=1, keepdims=True)*1e6+1)
    z = ((x-x.mean(axis=0))/x.std(axis=0)).astype(np.float32)
    table = celltype_enrichment(['A', 'B', 'C'], COUNTS, [20, 25, 30], [0, 1]).set_index('Cell type')
    ps = []
    for i, label in enumerate(['A', 'B', 'C']):
        inside, outside = z[i, :2], z[i, 2:]
        expected = np.mean((inside[:, None] > outside)+.5*(inside[:, None] == outside))
        p = mannwhitneyu(inside, outside, alternative='two-sided', method='asymptotic').pvalue
        np.testing.assert_allclose(table.loc[label, 'AUROC'], expected)
        np.testing.assert_allclose(table.loc[label, 'p-value'], p)
        ps.append(p)
    np.testing.assert_allclose(table.loc[['A', 'B', 'C'], 'Corrected p-value (BH)'], false_discovery_control(ps))


def test_replicating_bins_does_not_inflate_significance():
    first = celltype_enrichment(['A', 'B', 'C'], COUNTS, [20]*3, [0, 1])
    repeated = celltype_enrichment(['A', 'B', 'C'], COUNTS*1000, [20000]*3, [0, 1])
    for col in ['AUROC', 'p-value', 'Corrected p-value (BH)']:
        np.testing.assert_array_equal(first[col], repeated[col])
    assert (repeated['Target genes'] == 2).all()
    assert (repeated['Other genes'] == 4).all()


def test_custom_background_restricts_test_after_full_normalization():
    _, z = specificity_z(COUNTS, [20]*3)
    table = celltype_enrichment(['A', 'B', 'C'], COUNTS, [20]*3, [0, 0, 1, 2], [0, 1, 3, 3]).set_index('Cell type')
    for i, label in enumerate(['A', 'B', 'C']):
        test = mannwhitneyu(z[i, [0, 1]], z[i, [3]], method='asymptotic')
        np.testing.assert_allclose(table.loc[label, 'p-value'], test.pvalue)
        np.testing.assert_allclose(table.loc[label, 'AUROC'], test.statistic/2)
    assert (table['Other genes'] == 1).all()


def test_ties_excluded_and_single_profiles():
    table = celltype_enrichment(['A', 'B', 'C'], np.ones((3, 4)), [20, 30, 19], [0]).set_index('Cell type')
    np.testing.assert_array_equal(table.loc[['A', 'B'], 'AUROC'], [.5, .5])
    np.testing.assert_array_equal(table.loc[['A', 'B'], 'p-value'], [1, 1])
    assert np.isnan(table.loc['C', 'AUROC'])
    table = celltype_enrichment(['A', 'B', 'C'], COUNTS, [20, 30, 0], [0, 1]).set_index('Cell type')
    p = table.loc[['A', 'B'], 'p-value'].to_numpy()
    np.testing.assert_allclose(table.loc[['A', 'B'], 'Corrected p-value (BH)'], false_discovery_control(np.r_[p, 1])[:2])
    assert celltype_enrichment(['A'], [[1, 2]], [20], [0]).AUROC.isna().all()
    assert celltype_enrichment(['A', 'B'], np.zeros((2, 4)), [20]*2, [0]).AUROC.isna().all()


def test_underflow_is_reported_after_correction():
    counts = np.array([[10]*2000+[1]*2000, [1]*2000+[10]*2000])
    table = celltype_enrichment(['A', 'B'], counts, [20]*2, range(2000))
    floor = np.nextafter(0., 1.)
    assert (table['p-value'] == floor).all()
    assert (table['Corrected p-value (BH)'] == floor).all()


@pytest.mark.parametrize('target,background', [([], None), ([0, 1, 2, 3, 4, 5], None), ([0], [1, 2]), ([6], None)])
def test_invalid_gene_sets(target, background):
    with pytest.raises(ValueError):
        celltype_enrichment(['A', 'B', 'C'], COUNTS, [20]*3, target, background)


def test_identical_profiles_have_no_specificity_despite_roundoff():
    counts = np.tile([1, 2, 4, 8, 17, 99], (3, 1))
    keep, z = specificity_z(counts, [20]*3)
    assert keep.all()
    np.testing.assert_array_equal(z, np.zeros((3, 6)))
