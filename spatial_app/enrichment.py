"""Cell-type enrichment in the ranking of bins by their gene-set AUROC."""
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, false_discovery_control


def celltype_enrichment(labels, scores):
    labels=np.asarray(labels,dtype=str)
    scores=np.asarray(scores,dtype=float)
    if labels.shape!=scores.shape:
        raise ValueError('Labels and scores must align')
    names=np.unique(labels)
    valid=np.isfinite(scores)
    rows=[]
    for name in names:
        member=labels==name
        inside=scores[valid & member]
        outside=scores[valid & ~member]
        auc=p=np.nan
        if len(inside) and len(outside):
            if np.all(inside==inside[0]) and np.all(outside==inside[0]):
                # Zero rank variance: every pair ties, with no evidence of separation.
                auc,p=0.5,1.0
            else:
                test=mannwhitneyu(inside,outside,alternative='two-sided',method='asymptotic',use_continuity=True)
                auc=float(test.statistic)/(len(inside)*len(outside))
                p=float(test.pvalue)
        rows.append({'Cell type':name,'Scored bins':len(inside),'AUROC':auc,'p-value':p})
    table=pd.DataFrame(rows,columns=['Cell type','Scored bins','AUROC','p-value'])
    p=table['p-value'].to_numpy()
    # Keep the correction family fixed to all annotated types as batches arrive.
    corrected=false_discovery_control(np.where(np.isfinite(p),p,1.0),method='bh') if len(p) else p.copy()
    corrected[~np.isfinite(p)]=np.nan
    # Apply the reporting floor AFTER BH, so it does not alter the correction.
    # Undefined comparisons remain NaN; positive values are unchanged.
    floor=np.nextafter(np.float64(0),np.float64(1))
    table['p-value']=np.where(p==0,floor,p)
    corrected=np.where(corrected==0,floor,corrected)
    table['Corrected p-value (BH)']=corrected
    return table.sort_values(['AUROC','Cell type'],ascending=[False,True],na_position='last').reset_index(drop=True)
