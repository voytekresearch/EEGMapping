"""Utilities functions for EEG-Mapping analysis."""

import os
import pickle
import warnings
from copy import deepcopy

import numpy as np
from sklearn.preprocessing import scale

###################################################################################################
###################################################################################################

def load_pickle(f_name):
    """
    Load pickle from given directory
    f_name: str
    """    
    with open(os.path.join('../data/analysis/', f_name + '.pkl'), 'rb') as pickle_file:
        dat = pickle.load(pickle_file)

    return dat

def comb_dicts(dicts):
    """
    Combines dictionary
    dicts: list of dict
    """

    return {key: val for dd in dicts for key, val in dd.items()}


def mask_nan_array(dat):
    """
    Removes all nan values 
    dat: 4d array
    """
    return ~np.isnan(dat)


def combine_groups_dict(dataset1, dataset2):
    """
    Combines 2 dictionaries
    dataset1: dict
    dataset2: dict
    """
    bands = dataset1.keys()
    output = dict()

    for band in bands:
        output[band] = np.vstack([dataset1[band][:, 0, :, :], dataset2[band][:, 0, :, :]])

    return output


def combine_groups_array(dataset1, dataset2):
    """
    Combines 2 arrays
    dataset1: array
    dataset2: array
    """
    output = dict()
    output = np.vstack([dataset1[:, 0, :, :], dataset2[:, 0, :, :]])

    return output


def avg_for_topo(dataset, feat_in):
    """
    Averages values across 4th dimension of array
    dataset: 4d array
    feat_in: int
    """

    feat_set =  dataset[:,:,:,feat_in]
    bloc_aver_set = np.nanmean(feat_set, axis = 1)
    bloc_aver_set = np.nanmean(bloc_aver_set, axis = 0)

    return bloc_aver_set


def masking_cluster(pos_ch_cluster, eeg_dat):
    """
    Returns the index positions of electrodes to be masked after
    accepting the name of these electrodes as input.
    """

    pos_ch_cluster_index = []
    for pos_ch in pos_ch_cluster:
        pos_ch_cluster_index.append(eeg_dat.info['ch_names'].index(pos_ch))

    # Check the indices for the channel cluster
    return pos_ch_cluster_index


def demean(dataset):
    """
    Demean data.
    dataset: 4d array, [n_subjs, n_blocks, n_chs, n_feats]
    """

    dataset = deepcopy(dataset)

    for s_ind in range(dataset.shape[0]):
        for ch_ind in range(dataset.shape[2]):
            for f_ind in range(dataset.shape[3]):

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    dataset[s_ind, :, ch_ind, f_ind] = scale(dataset[s_ind, :, ch_ind, f_ind],
                                                             with_mean=True, axis=0, with_std=False)

    return dataset
