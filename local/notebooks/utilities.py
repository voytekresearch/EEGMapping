"""Utilities functions for EEG-Mapping analysis."""

import numpy as np

###################################################################################################
###################################################################################################

def mask_nan_array(dat):
    return ~np.isnan(dat)


def combine_groups(dataset1, dataset2):    
	bands = ["alpha", "beta", "theta"]
	output = dict()
	for band in bands:
		output[band] = np.vstack([dataset1[band][:, 0, :, :], dataset2[band][:, 0, :, :]])
	return output


def combine_slope_groups(dataset1, dataset2):
	output = dict()
	output = np.vstack([dataset1[:, 0, :, :], dataset2[:, 0, :, :]])
	return output


def avg_for_topo(dataset, band, feat_in):
    """
    dataset: 4d array
    band: string
    feat_in: int
    """ 
    feat_set =  dataset[band][:,:,:,feat_in]
    bloc_aver_set = np.nanmean(feat_set, axis = 1)
    bloc_aver_set = np.nanmean(bloc_aver_set, axis = 0)
    
    return bloc_aver_set


def avg_for_slope_topo(dataset, feat_in):
    """
    dataset: 4d array
    band: string
    feat_in: int
    """
    feat_set =  dataset[:,:,:,feat_in]
    bloc_aver_set = np.nanmean(feat_set, axis = 1)
    bloc_aver_set = np.nanmean(bloc_aver_set, axis = 0)
    
    return bloc_aver_set


def masking_cluster(pos_ch_cluster, eeg_dat):
    """Returns the index positions of electrodes to be masked after acception the name of these electrodes as input.
    """
    pos_ch_cluster_index = []
    for pos_ch in pos_ch_cluster:
        pos_ch_cluster_index.append(eeg_dat.info['ch_names'].index(pos_ch))

    # Check the indices for the channel cluster
    return pos_ch_cluster_index