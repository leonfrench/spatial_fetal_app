"""Gene-set specificity in count-summed tissue pseudobulks.

Matches the Han StereoSeq approach: log2(CPM + 1), gene-wise z-scores
across eligible tissues, then a target-versus-complement test over genes.
"""
import re
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, false_discovery_control
from spatial_app.scoring import _indices

MIN_BINS = 20


def specificity_z(counts, n_bins, min_bins=MIN_BINS):
    counts = np.asarray(counts, dtype=np.float64)
    n_bins = np.asarray(n_bins)
    if counts.ndim != 2 or n_bins.shape != (len(counts),):
        raise ValueError('Counts must be tissues x genes with one bin count per tissue')
    if not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError('Counts must be finite and nonnegative')
    if not np.isfinite(n_bins).all() or (n_bins < 0).any() or (n_bins != np.floor(n_bins)).any():
        raise ValueError('Bin counts must be nonnegative integers')
    totals = counts.sum(axis=1)
    keep = (n_bins >= min_bins) & (totals > 0)
    # Tissue specificity is undefined without at least two eligible profiles.
    if keep.sum() < 2:
        return np.zeros(len(counts), dtype=bool), np.empty((0, counts.shape[1]), dtype=np.float32)
    x = np.log2(counts[keep]/totals[keep, None]*1e6 + 1)
    mean, sd = x.mean(axis=0), x.std(axis=0)
    # Identical values can acquire a tiny nonzero std through mean rounding.
    variable = np.ptp(x, axis=0) > 0
    z = np.divide(x-mean, sd, out=np.zeros_like(x), where=(sd > 0) & variable)
    return keep, z.astype(np.float32)


def celltype_enrichment(labels, counts, n_bins, target_indices, background_indices=None, min_bins=MIN_BINS):
    """Test genes within each pseudobulk; bin counts never enter the U-test.

    CPM and z-scores use all retained genes before restricting a custom universe.
    Excluded tissues remain visible with NaN statistics and count as p=1 in BH.
    """
    labels = np.asarray(labels, dtype=str)
    counts = np.asarray(counts, dtype=np.float64)
    keep, z = specificity_z(counts, n_bins, min_bins)
    if labels.shape != (len(counts),) or len(np.unique(labels)) != len(labels):
        raise ValueError('Pseudobulk labels must be unique and align with counts')
    g = counts.shape[1]
    universe = np.arange(g) if background_indices is None else _indices(background_indices, g)
    target = np.intersect1d(_indices(target_indices, g), universe)
    other = np.setdiff1d(universe, target)
    if not len(target) or not len(other):
        raise ValueError('The universe needs target and non-target genes')
    auc = np.full(len(labels), np.nan)
    p = np.full(len(labels), np.nan)
    for index, profile in zip(np.flatnonzero(keep), z):
        inside, outside = profile[target], profile[other]
        if np.all(inside == inside[0]) and np.all(outside == inside[0]):
            auc[index], p[index] = 0.5, 1.0
        else:
            test = mannwhitneyu(inside, outside, alternative='two-sided', method='asymptotic', use_continuity=True)
            auc[index] = float(test.statistic)/(len(target)*len(other))
            p[index] = float(test.pvalue)
    corrected = false_discovery_control(np.where(np.isfinite(p), p, 1.0), method='bh') if len(p) else p.copy()
    corrected[~np.isfinite(p)] = np.nan
    floor = np.nextafter(np.float64(0), np.float64(1))
    table = pd.DataFrame({'Cell type': labels, 'Pseudobulk bins': np.asarray(n_bins, dtype=np.int64),
                          'Target genes': len(target), 'Other genes': len(other), 'AUROC': auc,
                          'p-value': np.where(p == 0, floor, p),
                          'Corrected p-value (BH)': np.where(corrected == 0, floor, corrected)})
    return table.sort_values(['AUROC', 'Cell type'], ascending=[False, True], na_position='last').reset_index(drop=True)


def format_p_value(p):
    """Three significant digits: standard notation from 1e-4 up, scientific below."""
    if not np.isfinite(p):
        return ''
    if p >= 1e-4:
        return f'{p:.3g}'
    return re.sub(r'e([+-])0*(\d)', r'e\1\2', f'{p:.2e}')
