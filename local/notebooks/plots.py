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

    vbuffer = 0.1 * (data.max() - data.min())
    vmin, vmax,  = data.min() - vbuffer, data.max() + vbuffer

    fig, ax = plt.subplots()
    mne.viz.plot_topomap(data, eeg_dat_info, vmin=vmin, vmax=vmax,
                         cmap=cm.viridis, contours=0, axes=ax)

    plot_topo_colorbar(vmin, vmax, title, save_fig)

    # This is saved differently because of MNE quirks
    if save_fig:
        fig_save_path = 'C:\\Users\\abc\\Documents\\Research\\figures'
        fig.savefig(os.path.join(fig_save_path, title + '.png'), dpi=600)


def plot_topo_colorbar(vmin, vmax, label, save_fig=True):

    # Create a colorbar for the topography plots
    fig = plt.figure(figsize=(2, 3))
    ax1 = fig.add_axes([0.9, 0.25, 0.15, 0.9])

    cmap = cm.viridis
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    cb1 = colorbar.ColorbarBase(plt.gca(), cmap=cmap,
                                norm=norm, orientation='vertical')

    save_figure(save_fig, label + '_cb')


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

def plot_oscillations(alphas, save_fig=False, save_name=None):
    """Plot a group of (flattened) oscillation definitions.

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

def plot_background(bgs, control_offset=False, save_fig=False, save_name=None):
    """Plot background components, comparing between groups.

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

    # Create the background model from parameters
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
