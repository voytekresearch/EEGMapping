"""Utilities functions for EEG-Mapping analysis."""

import warnings

import numpy as np
from sklearn.preprocessing import scale

from plots import plot_across_blocks

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
    """Returns the index positions of electrodes to be masked after
        accepting the name of these electrodes as input.
    """

    pos_ch_cluster_index = []
    for pos_ch in pos_ch_cluster:
        pos_ch_cluster_index.append(eeg_dat.info['ch_names'].index(pos_ch))

    # Check the indices for the channel cluster
    return pos_ch_cluster_index


def demean(dataset):
    """Demean data.

    dataset: 4d array, [n_subjs, n_blocks, n_chs, n_feats]
    """

    for s_ind in range(dataset.shape[0]):
        for ch_ind in range(dataset.shape[2]):
            for f_ind in range(dataset.shape[3]):

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    dataset[s_ind, :, ch_ind, f_ind] = scale(dataset[s_ind, :, ch_ind, f_ind],
                                                             with_mean=True, axis=0, with_std=False)

    return dataset


def run_dict_across_blocks(label, dataset, ch_indices, SAVE_FIGS):
    """
    label: str
    dataset: dict
    ch_indices: list of str
    SAVE_FIGS: bool
    """

    bands = dataset.keys()
    feat_labels = ["CFS", "AMPS", "BWS"]

    for band in bands:

        curr_data = dataset[band]
        run_array_across_blocks(label + '_band_', curr_data, ch_indices, feat_labels=feat_labels, SAVE_FIGS=SAVE_FIGS)


def run_array_across_blocks(label, dataset, ch_indices, feat_labels, SAVE_FIGS):
    """Run analysis of FOOOF features across blocks.
    label:
    dataset:
    ch_indices
    feat_labels:
    SAVE_FIGS:
    """

    for feat_in, feat in enumerate(feat_labels):

        dataset = demean(dataset)

        demeaned_curr_masked_data = np.take(dataset, indices=ch_indices,  axis=2)
        demeaned_curr_mean_data = np.nanmean(demeaned_curr_masked_data, axis=2)
        demeaned_curr_data_matrix = demeaned_curr_mean_data[:,:,feat_in]

        means = np.nanmean(demeaned_curr_data_matrix, axis=0)
        stds = np.std(demeaned_curr_data_matrix, axis=0)

        plot_across_blocks(means, stds, label + "_" + feat + "_across_blocks_plot")
