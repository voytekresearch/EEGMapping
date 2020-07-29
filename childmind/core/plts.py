"""Plot function for EEGMapping project."""

import numpy as np
import matplotlib.pyplot as plt

###################################################################################################
###################################################################################################

def plot_scatter(x_data, y_data, add_line=True, add_equal=False, color=None,
                 xlim=None, ylim=None, title=None, xlabel=None, ylabel=None,
                 figsize=None, save_fig=False, file_name=None, ax=None):
    """Plot a scatter plot of data from a DataFrame."""

    if not ax:
        _, ax = plt.subplots(figsize=figsize)

    ax.scatter(x_data, y_data, color=color)

    if add_line:
        z = np.polyfit(x_data, y_data, deg=1)
        ax.plot(x_data, z[0]*x_data + z[1], '--k', alpha=0.5);

    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)

    if add_equal:
        lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
                np.max([ax.get_xlim(), ax.get_ylim()])]
        ax.plot(lims, lims, '--', alpha=0.75, zorder=0)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

    if title: ax.set_title(title, fontsize=20)
    if xlabel: ax.set_xlabel(xlabel, fontsize=16)
    if ylabel: ax.set_ylabel(ylabel, fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=14);

    if save_fig:
        plt.savefig(file_name + '.pdf', bbox_inches='tight')
