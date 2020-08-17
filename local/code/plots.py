"""Plotting functions for EEG-Mapping analysis."""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors, colorbar
from scipy.stats import ttest_1samp, ttest_ind, sem, pearsonr

import mne

from fooof.core.funcs import gaussian_function, expo_nk_function

from utilities import *

###################################################################################################
###################################################################################################

def save_figure(save_out, name):
    """Saves a given plot to given location with specific name.

    save_out: bool
    name: str
    """

    if save_out:
        plt.savefig(os.path.join('../figures/', name + '.pdf'), bbox_inches='tight')


def plot_comp(title, feature, data1, data2, save_fig=False, save_name=None):
    """Plots the comparison between groups, as a mean value with an errorbar.

    title: str
    feature: str
    data1: 1d array
    data2: 1d array
    save_fig: bool
    save_name: str
    """

    fig, ax = plt.subplots(figsize=[2, 4])

    # Split up data
    data1 = data1[~np.isnan(data1)]
    data2 = data2[~np.isnan(data2)]

    means = [np.mean(data1), np.mean(data2)]
    sems = [sem(data1), sem(data2)]

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
    """Plots the correlations across blocks.

    avgs: 1d array
    yerrs: 1d array
    ylabel: str
    label: str
    save_fig: bool
    """

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


def plot_corrs_boxplot(corrs, corr_labels, label, save_fig=True):
    """Plots the correlations across epochs.

    corrs: 1d array
    corr_labels: 1d array
    label: str
    save_fig: bool
    """

    fig, ax = plt.subplots(figsize=(14, 6))
    plt.boxplot(corrs);

    plt.title('Feature Correlations Between Rest Blocks', fontsize=20);
    #plt.xlabel('Feature')
    plt.ylabel('Correlation Value', fontsize=16);

    xtickNames = plt.setp(ax, xticklabels=corr_labels)

    _set_lr_spines(ax, 4)
    _set_tick_sizes(ax, x_size=20, y_size=14)
    _set_label_sizes(ax, 18, 18)

    save_figure(save_fig, label)


def plot_topo(data, title, eeg_info, save_fig=True):
    """Plots the spatial topographical graphs of a given function.

    data: 1d array, len number of channels
    title: str
    eeg_info: MNE object
    save_fig: bool
    """

    inds = np.where(np.isnan(data))
    data[inds] = np.nanmean(data)

    vbuffer = 0.1 * (data.max() - data.min())
    vmin, vmax,  = data.min() - vbuffer, data.max() + vbuffer

    fig, ax = plt.subplots()
    mne.viz.plot_topomap(data, eeg_info, vmin=vmin, vmax=vmax,
                         cmap=cm.viridis, contours=0, axes=ax)

    plot_topo_colorbar(vmin, vmax, title, save_fig)

    # This is saved differently because of MNE quirks
    if save_fig:
        fig_save_path = 'C:\\Users\\abc\\Documents\\Research\\figures'
        fig.savefig(os.path.join(fig_save_path, title + '.png'), dpi=600)


def plot_topo_colorbar(vmin, vmax, label, save_fig=True):
    """Creates a colorbar for the topography plots.

    vmin: int
    vmax: int
    label: str
    saave_fig: bool
    """
    fig = plt.figure(figsize=(2, 3))
    ax1 = fig.add_axes([0.9, 0.25, 0.15, 0.9])

    cmap = cm.viridis
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    cb1 = colorbar.ColorbarBase(plt.gca(), cmap=cmap,
                                norm=norm, orientation='vertical')

    save_figure(save_fig, label + '_cb')


def plot_space_scatter(data, pos, label, xlabel, ylabel, save_fig=True):
    """Plot scatter graph of values across spatial positions .

    data : 1d array
    pos : 1d array
    label : str
    xlabel : str
    ylabel : str
    save_fig : bool
    """
    fig, ax = plt.subplots()
    plt.scatter(pos, data)

    # Titles & Labels
    ax.set_title(label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    _set_lr_spines(ax, 4)
    _set_tick_sizes(ax)
    _set_label_sizes(ax)

    save_figure(save_fig, label)


# NOTE: CAN BE USED FROM FOOOF NOW
def plot_oscillations(alphas, save_fig=False, save_name=None):
    """Plot a group of (flattened) oscillation definitions.

    alphas: 1d array
    save_fig: bool
    save_name: str
    Note: plot taken & adapated from EEGFOOOF.
    """

    n_subjs = alphas.shape[0]

    # Initialize figure
    fig, ax = plt.subplots(figsize=[6, 6])

    # Get frequency axis (x-axis)
    fs = np.arange(4, 18, 0.1)

    # Create the oscillation model from parameters
    osc_psds = np.empty(shape=[n_subjs, len(fs)])
    for ind, alpha in enumerate(alphas):
        osc_psds[ind, :] = gaussian_function(fs, *alphas[ind, :])

    # Plot each individual subject
    for ind in range(n_subjs):
        ax.plot(fs, osc_psds[ind, :], alpha=0.3, linewidth=1.5)

    # Plot the average across all subjects
    avg = np.nanmean(osc_psds, 0)
    ax.plot(fs, avg, 'k', linewidth=3)

    ax.set_ylim([0, 2.2])

    ax.set_xlabel('Frequency')
    ax.set_ylabel('Power')

    # Set the top and right side frame & ticks off
    _set_lr_spines(ax, 2)
    _set_tick_sizes(ax)
    _set_label_sizes(ax)

    save_figure(save_fig, save_name)


# NOTE: CAN BE USED FROM FOOOF NOW
def plot_aperiodic(bgs, control_offset=False, save_fig=False, save_name=None):
    """Plots the aperiodic components, comparing between groups.

    bgs: 2d array
    control_offset: bool
    save_fig: bool
    save_name: bool
    Note: Copied & adapted from EEGFOOOF.
    """

    n_subjs = bgs.shape[0]

    # Set offset to be zero across all PSDs
    tbgs = np.copy(bgs)
    if control_offset:
        tbgs[:, 0] = 1

    fig, ax = plt.subplots(figsize=[8, 6])

    # Get frequency axis (x-axis)
    fs = np.arange(1, 35, 0.1)

    # Create the aperiodic model from parameters
    bg_psds = np.empty(shape=[n_subjs, len(fs)])
    for ind, bg in enumerate(tbgs):
        bg_psds[ind, :] = expo_nk_function(fs, *tbgs[ind, :])

    # Drop any zero lines
    del_inds = []
    for ind, bgp in enumerate(bg_psds):
        if sum(bgp) == 0:
            del_inds.append(ind)
    bg_psds = np.delete(bg_psds, del_inds, 0)

    plt.ylim([-13.5, -9.5])

    # Set whether to plot x-axis in log
    plt_log = False
    fs = np.log10(fs) if plt_log else fs

    # Plot each individual subject
    for ind in range(bg_psds.shape[0]):
        ax.plot(fs, bg_psds[ind, :], "#0d82c1", alpha=0.25, linewidth=1.5)

    plt.plot(fs, np.mean(bg_psds, 0), '#000000', linewidth=2)

    ax.set_xlabel('Frequency')
    ax.set_ylabel('Power')

    _set_lr_spines(ax, 2)
    _set_tick_sizes(ax)
    _set_label_sizes(ax)

    save_figure(save_fig, save_name)


###################################################################################################
###################################################################################################

def _set_lr_spines(ax, lw=None):
    """Sets the values for spines.

    ax: axis obj
    lw: int
    """
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
    """Sets the tick sizes.

    ax: axis obj
    x_size: int
    y_size: int
    """

    plt.setp(ax.get_xticklabels(), fontsize=x_size)
    plt.setp(ax.get_yticklabels(), fontsize=y_size)

def _set_label_sizes(ax, x_size=16, y_size=16):
    """Sets the label fontsizes.

    ax: axis obj
    x_size: int
    y_size: int
    """

    ax.xaxis.label.set_size(x_size)
    ax.yaxis.label.set_size(y_size)
