"""Plotting functions for EEG-Mapping analysis."""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp, ttest_ind, sem, pearsonr
from sklearn.preprocessing import scale
from utilities import *

###################################################################################################
###################################################################################################

def save_figure(save_out, name):
    fig_save_path = 'C:\\Users\\abc\\Documents\\Research\\figures'
    if save_out:
        plt.savefig(os.path.join(fig_save_path, name + '.png'))


def plot_comp(title, feature, dat1, dat2, save_fig=False, save_name=None):
    """Plot comparison between groups, as a mean value with an errorbar.
    Dat should be 1D vector, with data that can be split up by YOUNG & OLD _INDS.
    """
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

    save_figure(SAVE_FIGS, title +"_" + feature + "_across_state")


def make_across_blocks(curr_data, dataset, pos_ch_cluster_index, SAVE_FIGS):

    bands = ["alpha", "beta", "theta"]
    feats = ["CFS", "AMPS", "BWS"]

    for band in bands:
        for feat_in, feat in enumerate(feats):
            
            for s_ind in range(dataset[band].shape[0]):
                for ch_ind in range(dataset[band].shape[2]):
                    for f_ind in range(dataset[band].shape[3]):
                        dataset[band][s_ind, :, ch_ind, f_ind] = scale(dataset[band][s_ind, :, ch_ind, f_ind], with_mean=True, axis=0, with_std=False)
                        
            demeaned_curr_masked_data = np.take(dataset[band], indices=pos_ch_cluster_index,  axis=2 )
            demeaned_curr_mean_data = np.nanmean(demeaned_curr_masked_data, axis = 2)
            demeaned_curr_data_matrix = demeaned_curr_mean_data[:,:,feat_in]
            
            plt.figure()
            plt.plot(demeaned_curr_data_matrix.T, ".");
            save_figure(SAVE_FIGS, curr_data + "_" + band + "_" + feat + "_across_blocks_plot")



def make_slope_across_blocks(curr_data, dataset, pos_ch_cluster_index, SAVE_FIGS):

    feats = ["Offset", "Slope"]
    
    for feat_in, feat in enumerate(feats):

        for s_ind in range(dataset.shape[0]):
            for ch_ind in range(dataset.shape[2]):
                for f_ind in range(dataset.shape[3]):
                    dataset[s_ind, :, ch_ind, f_ind] = scale(dataset[s_ind, :, ch_ind, f_ind],
                                                             with_mean=True, axis=0, with_std=False)

        demeaned_curr_masked_data = np.take(dataset, indices=pos_ch_cluster_index,  axis=2)
        demeaned_curr_mean_data = np.nanmean(demeaned_curr_masked_data, axis=2)
        demeaned_curr_data_matrix = demeaned_curr_mean_data[:,:,feat_in]
        
        
        avg_across_subjs = np.nanmedian(demeaned_curr_data_matrix, axis=0)
        print(avg_across_subjs.shape)
        
        corr = pearsonr(avg_across_subjs, range(0,dataset.shape[1]))
        print(str(corr) + " for the dataset: " + curr_data + "_" + feat)

        plt.figure()
        plt.plot(demeaned_curr_data_matrix.T, ".");
        save_figure(SAVE_FIGS, curr_data + "_" + feat + "_across_blocks_plot")



def make_topos(datasets, state):
    """
    datasets: list of 4d arrays (?)
    """

    bands = ["alpha", "beta", "theta"]
    feats = ["CFS", "AMPS", "BWS"]

    for band in bands:
        for feat_in, feat in enumerate(feats):
            #print(feat_in)

            topo_dat = np.zeros(shape=[2, 64])

            for ind, dataset in enumerate(datasets):
                topo_dat[ind, :] =  avg_for_topo(dataset, band, feat_in)

            plot_topo(topo_dat[0, :], title='D1' + state + band + feat)
            plot_topo(topo_dat[0, :], title='D2' + state + band + feat)
            plot_topo(np.mean(topo_dat, 0), title='Both' + state + band + feat)
            
            #medial to lateral
            plt.figure()
            plt.scatter(abs(pos[:, 0]), np.mean(topo_dat,0))
            save_figure(SAVE_FIGS, 'Both_' + state + band +  feat + "_medial_to_anterior_plot")
        
            # posterior to anterior
            plt.figure()
            plt.scatter(pos[:, 1], np.mean(topo_dat,0))
            save_figure(SAVE_FIGS, 'Both_' + state + band + feat + "_posterior_to_anterior_plot")



def plot_topo(data, title):
    """
    data: 1d array, len number of channels
    title: string
    """
    
    inds = np.where(np.isnan(data))
    data[inds] = np.nanmean(data)
    
    fig, ax = plt.subplots()
    mne.viz.plot_topomap(data, eeg_dat.info, cmap=cm.viridis, contours=0, axes=ax)
    fig_save_path = 'C:\\Users\\abc\\Documents\\Research\\figures'
    fig.savefig(os.path.join(fig_save_path, title + '.png'), dpi=600)



def make_slope_topos(datasets,state):

    feats = ["Offset", "Slope"]

    for feat_in, feat in enumerate(feats):

        topo_dat = np.zeros(shape=[2, 64])

        for ind, dataset in enumerate(datasets):
            topo_dat[ind, :] =  avg_for_slope_topo(dataset, feat_in)
        print (np.mean(topo_dat, axis = 0))
        plot_topo(topo_dat[0, :], title='D1' + state + feat)

        plot_topo(topo_dat[0, :], title='D2'+ state + feat)

        plot_topo(np.mean(topo_dat, 0), title='Both_' + state +feat)

        #medial to lateral
        plt.figure()
        plt.scatter(abs(pos[:, 0]), np.mean(topo_dat,0))
        save_figure(SAVE_FIGS, 'Both_' + state + feat + "_medial_to_anterior_plot")
        
        # posterior to anterior
        plt.figure()
        plt.scatter(pos[:, 1], np.mean(topo_dat,0))
        save_figure(SAVE_FIGS, 'Both_' + state + feat + "_posterior_to_anterior_plot")
