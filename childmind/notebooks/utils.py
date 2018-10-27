"""   """

from scipy.stats import pearsonr
from itertools import combinations, product

####################################################################################################
####################################################################################################

def comb_corrs(lst):
    """

    Parameters
    ----------
    lst : list
        xx

    Returns
    -------
    corrs : 1d array
        xx
    """

    corrs = []
    for ii, jj in combinations(lst, 2):
        corrs.append(pearsonr(ii, jj)[0])

    return corrs


def bet_corrs(lst1, lst2):
    """

    Parameters
    ----------
    lst1 : list
        xx
    lst2 : list
        xx

    Returns
    -------
    corrs : 1d array
        xx
    """

    corrs = []

    for ii, jj in product(lst1, lst2):
        corrs.append(pearsonr(ii, jj)[0])

    return corrs
