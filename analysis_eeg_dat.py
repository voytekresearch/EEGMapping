"""Analyzing EEG data ..."""

import mne
import os
import numpy as np
from fooof import FOOOFGroup
from fooof.analysis import *
from autoreject import LocalAutoRejectCV

####################################################################################################
####################################################################################################


# Set up paths

# This results and analysis path will need updating
results_path = '/Users/luyandamdanda/Documents/Research/Results'
analysis_path = '/Users/luyandamdanda/Documents/Research/Analysis'

# These should stay the same
subj_dat_num = list(range(3502, 3504))

def main():

	# START LOOP
    for sub in subj_dat_num:
	    print('Current Subject Results: ' + str(sub))

	    # load results data
	    fg = FOOOFGroup()
	    subj_dat_fname = str(sub)+ "fooof_group_results"
	    full_path = os.path.join(results_path, subj_dat_fname)
	    fg.load(file_name=subj_dat_fname, file_path=results_path)

	    if not fg.group_results:
	        print('Current Subject Results: ' + str(sub) + " failed to load")
	    else:
	        print('Current Subject Results: ' + str(sub) + " successfully loaded")

	    #number of channels
	    n_channels = len(fg)
	    # Set up indexes for accessing data, for convenience
	    cf_ind, am_ind, bw_ind = 0, 1, 2

	    # Define bands of interest
	    misc_band = [30, 40]
	    beta_band = [15, 30]
	    alpha_band = [7, 14]
	    theta_band = [2, 7]


	    # Initialize an array to store band oscillations
	    miscs = np.empty(shape=[n_channels, 3])
	    betas = np.empty(shape=[n_channels, 3])
	    alphas = np.empty(shape=[n_channels, 3])
	    thetas = np.empty(shape=[n_channels, 3])


	    # Extract all alpha oscillations from FOOOFGroup results
	    #  Note that this preserves information about which PSD each oscillation comes from
	    for ind, res in enumerate(fg):
	    	miscs[ind, :] = get_band_peak(res.peak_params, misc_band, True)
	    	betas[ind, :] = get_band_peak(res.peak_params, beta_band, True)
	    	alphas[ind, :] = get_band_peak(res.peak_params, alpha_band, True)
	    	thetas[ind, :] = get_band_peak(res.peak_params, theta_band, True)


	    # Save summarized analysis results in .txt file
	    f = open(analysis_path + str(sub) + "analysis_summary.txt", 'w')
	    f.write('Miscs CF : ' + str(np.nanmean(miscs[:, cf_ind])) + "\n")
	    f.write('Miscs Amp: ' + str(np.nanmean(miscs[:, am_ind])) + "\n")
	    f.write('Miscs BW : '+ str(np.nanmean(miscs[:, bw_ind])) + "\n")
	    f.write('Beta CF : ' + str(np.nanmean(betas[:, cf_ind])) + "\n")
	    f.write('Beta Amp: ' + str(np.nanmean(betas[:, am_ind])) + "\n")
	    f.write('Beta BW : ' + str(np.nanmean(betas[:, bw_ind])) + "\n")
	    f.write('Alpha CF : ' + str(np.nanmean(alphas[:, cf_ind])) + "\n")
	    f.write('Alpha Amp: ' + str(np.nanmean(alphas[:, am_ind])) + "\n")
	    f.write('Alpha BW : ' + str(np.nanmean(alphas[:, bw_ind])) + "\n")
	    f.write('Theta CF : ' + str(np.nanmean(thetas[:, cf_ind])) + "\n")
	    f.write('Theta Amp: ' + str(np.nanmean(thetas[:, am_ind])) + "\n")
	    f.write('Theta BW : ' + str(np.nanmean(thetas[:, bw_ind])) + "\n")
	    f.close

		# Save raw analysis data in .txt file
	    f = open(analysis_path + str(sub) + "analysis_raw.txt", 'w')
	    f.write("miscs: " + "\n" + str(miscs) + "\n")
	    f.write("betas: " + "\n" + str(betas) + "\n")
	    f.write("alphas: " + "\n" + str(alphas) + "\n")
	    f.write("thetas: " + "\n" + str(thetas) + "\n")
	    f.close

	    print("File SAVED")

if __name__ == "__main__":
    main()



