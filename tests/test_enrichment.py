import numpy as np
from scipy.stats import mannwhitneyu, false_discovery_control
from spatial_app.enrichment import celltype_enrichment


def test_underflow_is_reported_as_smallest_positive_float():
    scores=np.r_[np.zeros(2000),np.ones(2000)]
    labels=np.array(['A']*2000+['B']*2000)
    assert mannwhitneyu(scores[:2000],scores[2000:],method='asymptotic').pvalue==0
    table=celltype_enrichment(labels,scores)
    floor=np.nextafter(0.,1.)
    assert (table['p-value']==floor).all()
    assert (table['Corrected p-value (BH)']==floor).all()


def test_group_auroc_matches_direct_pairwise_with_ties():
    labels=np.array(['A','A','A','B','B','C','C'])
    scores=np.array([.8,.9,.9,.1,.2,.2,.8])
    table=celltype_enrichment(labels,scores).set_index('Cell type')
    ps=[]
    for name in sorted(set(labels)):
        x=scores[labels==name];y=scores[labels!=name]
        expected=np.mean((x[:,None]>y)+.5*(x[:,None]==y))
        np.testing.assert_allclose(table.loc[name,'AUROC'],expected)
        p=mannwhitneyu(x,y,alternative='two-sided',method='asymptotic').pvalue
        np.testing.assert_allclose(table.loc[name,'p-value'],p);ps.append(p)
    np.testing.assert_allclose(table.loc[['A','B','C'],'Corrected p-value (BH)'],false_discovery_control(ps))
    assert table.loc['A','AUROC']>.5 and table.loc['B','AUROC']<.5


def test_ties_and_unscored_bins():
    table=celltype_enrichment(['A','A','B','B'],[.5]*4)
    np.testing.assert_array_equal(table.AUROC,[.5,.5])
    np.testing.assert_array_equal(table['p-value'],[1,1])
    table=celltype_enrichment(['A','B','C'],[.9,.1,np.nan]).set_index('Cell type')
    assert table.loc['A','AUROC']==1 and table.loc['B','AUROC']==0
    assert table.loc['C','Scored bins']==0 and np.isnan(table.loc['C','AUROC'])
    p=table.loc['A','p-value']
    np.testing.assert_allclose(table.loc['A','Corrected p-value (BH)'],false_discovery_control([p,p,1])[0])
    assert celltype_enrichment(['A','B'],[np.nan,np.nan]).AUROC.isna().all()
    assert celltype_enrichment(['A','A'],[.2,.8]).AUROC.isna().all()
