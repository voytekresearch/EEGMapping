"""Utility functions for EEGMapping project."""

from itertools import combinations, product

from scipy.stats import pearsonr, spearmanr

####################################################################################################
####################################################################################################

def comb_corrs(lst):
    """Combine a list of correlations."""

    corrs = []
    for ii, jj in combinations(lst, 2):
        corrs.append(pearsonr(ii, jj)[0])

    return corrs


def bet_corrs(lst1, lst2, corr_func=pearsonr):
    """Calculate correlations between lists of elements."""

    corrs = []

    for ii, jj in product(lst1, lst2):
        corrs.append(corr_func(ii, jj)[0])

    return corrs
