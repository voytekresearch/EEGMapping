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
        plt.savefig(os.path.join(fig_save_path, name + '.png'), bbox_inches='tight', dpi=300)


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
    ax.set_ylabel(feature)

    _set_lr_spines(ax, 4)
    _set_tick_sizes(ax)
    _set_label_sizes(ax)

    save_figure(save_fig, title + "_" + feature + "_across_state")


def plot_across_blocks(avgs, yerrs, ylabel, label, save_fig=True):
    """   """

    block_nums = [ii + 1 for ii in range(len(avgs))]

    fig, ax = plt.subplots()
    plt.errorbar(block_nums, avgs, yerr=yerrs, xerr=None, fmt='.',
                 markersize=22, capsize=10, elinewidth=2, capthick=2)

    # Titles & Labels
    ax.set_title(label)
    ax.set_xlabel('Block Number')
    ax.set_ylabel(ylabel)

    _set_lr_spines(ax, 4)
    _set_tick_sizes(ax)
    _set_label_sizes(ax)

    save_figure(save_fig, label)


def plot_topo(data, title, eeg_dat_info, save_fig=True):
    """
    data: 1d array, len number of channels
    title: string
    """

    inds = np.where(np.isnan(data))
    data[inds] = np.nanmean(data)

    fig, ax = plt.subplots()
    mne.viz.plot_topomap(data, eeg_dat_info, cmap=cm.viridis, contours=0, axes=ax)

    # This is saved differently because of MNE quirks
    if save_fig:
        fig_save_path = 'C:\\Users\\abc\\Documents\\Research\\figures'
        fig.savefig(os.path.join(fig_save_path, title + '.png'), dpi=600)


def plot_space_scatter(dat, pos, label, xlabel, ylabel, save_fig=True):
    """   """

    fig, ax = plt.subplots()
    plt.scatter(pos, dat)

    # Titles & Labels
    ax.set_title(label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    _set_lr_spines(ax, 4)
    _set_tick_sizes(ax)
    _set_label_sizes(ax)

    save_figure(save_fig, label)

##
##

def _set_lr_spines(ax, lw=None):
    """   """

    # Set the top and right side frame & ticks off
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # Set linewidth of remaining spines
    if lw:
        ax.spines['left'].set_linewidth(lw)
        ax.spines['bottom'].set_linewidth(lw)


def _set_tick_sizes(ax, x_size=12, y_size=12):

    # Set tick fontsizes
    plt.setp(ax.get_xticklabels(), fontsize=x_size)
    plt.setp(ax.get_yticklabels(), fontsize=y_size)

def _set_label_sizes(ax, x_size=14, y_size=14):

    # Set tick fontsizes
    ax.xaxis.label.set_size(x_size)
    ax.yaxis.label.set_size(y_size)
