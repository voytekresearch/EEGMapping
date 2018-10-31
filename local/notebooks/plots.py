"""Plotting functions for EEG-Mapping analysis."""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from scipy.stats import ttest_1samp, ttest_ind, sem, pearsonr

import mne

from utilities import *

###################################################################################################
###################################################################################################

def save_figure(save_out, name):

    fig_save_path = 'C:\\Users\\abc\\Documents\\Research\\figures'
    if save_out:
        plt.savefig(os.path.join(fig_save_path, name + '.png'))


def plot_comp(title, feature, dat1, dat2, save_fig=False, save_name=None):
    """Plot comparison between groups, as a mean value with an errorbar."""

    fig, ax = plt.subplots(figsize=[2, 4])

    # Split up data
    dat1 = dat1[~np.isnan(dat1)]
    dat2 = dat2[~np.isnan(dat2)]

    means = [np.mean(dat1), np.mean(dat2)]
    sems = [sem(dat1), sem(dat2)]

    plt.errorbar([1, 2], means, yerr=sems, xerr=None, fmt='.',
                 markersize=22, capsize=10, elinewidth=2, capthick=2)

    ax.set_xlim([0.5, 2.5])
    plt.xticks([1, 2], ['Trial', 'Rest'])

    # Titles & Labels
    ax.set_title(title)
    ax.set_xlabel('State')
    ax.set_ylabel(feature)

    save_figure(save_fig, title + "_" + feature + "_across_state")


def plot_across_blocks(means, stds, name):

        plt.figure()
        plt.errorbar(range(len(means)), means, yerr=stds, xerr=None, fmt='.',
                     markersize=22, capsize=10, elinewidth=2, capthick=2)
        save_figure(SAVE_FIGS, name)


def make_topos(datasets, state, eeg_dat_info, pos, SAVE_FIGS=True):
    """
    datasets: list of 4d arrays (?)
    """

    bands = ["alpha", "beta", "theta"]
    feats = ["CFS", "AMPS", "BWS"]

    for band in bands:
        for feat_in, feat in enumerate(feats):

            topo_dat = np.zeros(shape=[2, 64])

            for ind, dataset in enumerate(datasets):
                topo_dat[ind, :] =  avg_for_topo(dataset, band, feat_in)

            plot_topo(topo_dat[0, :], title='D1' + state + band + feat, eeg_dat_info=eeg_dat_info)
            plot_topo(topo_dat[0, :], title='D2' + state + band + feat, eeg_dat_info=eeg_dat_info)
            plot_topo(np.mean(topo_dat, 0), title='Both' + state + band + feat, eeg_dat_info=eeg_dat_info)

            #medial to lateral
            plt.figure()
            plt.scatter(abs(pos[:, 0]), np.mean(topo_dat,0))
            pearsonr(abs(pos[:, 0]), np.mean(topo_dat,0))
            save_figure(SAVE_FIGS, 'Both_' + state + band +  feat + "_medial_to_lateral_plot")

            # posterior to anterior
            plt.figure()
            plt.scatter(pos[:, 1], np.mean(topo_dat,0))
            pearsonr(pos[:, 1], np.mean(topo_dat,0))
            save_figure(SAVE_FIGS, 'Both_' + state + band + feat + "_posterior_to_anterior_plot")


def plot_topo(data, title, eeg_dat_info):
    """
    data: 1d array, len number of channels
    title: string
    """

    inds = np.where(np.isnan(data))
    data[inds] = np.nanmean(data)

    fig, ax = plt.subplots()
    mne.viz.plot_topomap(data, eeg_dat_info, cmap=cm.viridis, contours=0, axes=ax)
    fig_save_path = 'C:\\Users\\abc\\Documents\\Research\\figures'
    fig.savefig(os.path.join(fig_save_path, title + '.png'), dpi=600)


def make_slope_topos(datasets,state, eeg_dat_info, pos, SAVE_FIGS = True):

    feats = ["Offset", "Slope"]

    for feat_in, feat in enumerate(feats):

        topo_dat = np.zeros(shape=[2, 64])

        for ind, dataset in enumerate(datasets):
            topo_dat[ind, :] =  avg_for_slope_topo(dataset, feat_in)

        print (np.mean(topo_dat, axis = 0))
        plot_topo(topo_dat[0, :], title='D1' + state + feat, eeg_dat_info=eeg_dat_info)

        plot_topo(topo_dat[0, :], title='D2'+ state + feat, eeg_dat_info=eeg_dat_info)

        plot_topo(np.mean(topo_dat, 0), title='Both_' + state +feat, eeg_dat_info=eeg_dat_info)

        #medial to lateral
        plt.figure()
        plt.scatter(abs(pos[:, 0]), np.mean(topo_dat,0))
        pearsonr(abs(pos[:, 0]), np.mean(topo_dat,0))
        save_figure(SAVE_FIGS, 'Both_' + state + feat + "_medial_to_anterior_plot")

        # posterior to anterior
        plt.figure()
        plt.scatter(pos[:, 1], np.mean(topo_dat,0))
        pearsonr(pos[:, 1], np.mean(topo_dat,0))
        save_figure(SAVE_FIGS, 'Both_' + state + feat + "_posterior_to_anterior_plot")
