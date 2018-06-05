"""Analyzing EEG data ..."""

import mne
import os
import numpy as np
import fnmatch
from fooof import FOOOFGroup
from fooof.analysis import *
#from autoreject import LocalAutoRejectCV
from pathlib import Path

####################################################################################################
## SETTINGS ##

DATASET = 'rtPB'
STATE = 'rest'

####################################################################################################
####################################################################################################

# Set up paths
# This results and analysis path will need updating
# Set data path based on STATE
if STATE == 'rest':
	results_path = 'D:\\abc\\Documents\\Research\\Results\\Rest_Results'
elif STATE == 'trial':
	results_path = 'D:\\abc\\Documents\\Research\\Results\\Trial_Results'

# DATASET properties
# Properties of rtPB Dataset
if DATASET == 'rtPB':
	subj_dat_num = list(range(3502, 3516))
	if STATE == 'rest':
		num_blocks = 2
	elif STATE == 'trial':
		num_blocks = 10 
	n_subjects = len(subj_dat_num)
# Properties of PBA Dataset
elif DATASET == 'PBA':
	subj_dat_num = list(range(3001, 3015))
	if STATE == 'rest':
		num_blocks = 1
	elif STATE == 'trial':
		num_blocks = 12 
	n_subjects = len(subj_dat_num)

# General properties
#number of channels
n_channels = 64

# Set up indexes for accessing data, for convenience
cf_ind, am_ind, bw_ind = 0, 1, 2
n_feats = 3

# Define bands of interest
theta_band = [2, 7]
alpha_band = [7, 14]
beta_band = [15, 30]

def main():

	# GET LIST OF FILES - OUTDATED 
	#rest_f_names = fnmatch.filter(os.listdir(rest_results_path), '*fooof_group_results?.json')
	#trial_f_names = fnmatch.filter(os.listdir(trial_results_path), '*fooof_group_results?.json')

	# Initialize 3D group arrays
	thetas = np.zeros(shape=[n_subjects, num_blocks, n_channels, n_feats])
	alphas = np.zeros(shape=[n_subjects, num_blocks, n_channels, n_feats])
	betas = np.zeros(shape=[n_subjects, num_blocks, n_channels, n_feats])

	# START LOOP
	# DATASET: PBA 
	for sub_index, sub_num in enumerate(subj_dat_num):
		print('Current Subject Results: ' + str(sub_num))
		
		# load rest results data
		for block in range(0, num_blocks):
			subj_file = str(sub_num) + "fooof_group_results" + str(block) + ".json"
			fg = FOOOFGroup()
			full_path = os.path.join(results_path, subj_file)
			path_check = Path(full_path)
			if path_check.is_file():
				fg.load(file_name=subj_file, file_path=results_path)
			if not fg.group_results:
				print('Current Subject Results: ' + str(sub_num) + " failed to load")
			else:
				print('Current Subject Results: ' +  str(sub_num) +" successfully loaded")



			for ind, res in enumerate(fg):
				thetas[ind, block, :,  :] = get_band_peak(res.peak_params, theta_band, True)
				alphas[ind, block, :,  :] = get_band_peak(res.peak_params, alpha_band, True)
				betas[ind, block, :,  :] = get_band_peak(res.peak_params, beta_band, True)

			#alphas[sub_ind, block, :, :] = get_band_peak_group()
			#betas[sub_ind, block, :, :] = get_band_peak_group()

			# Save out matrices
			# Update to save out files using DATASET and STATE
			np.save('..\\data\\' + DATASET + "_" + STATE + "_thetas.npy" , thetas)
			np.save('..\\data\\' + DATASET + "_" + STATE + "_alphas.npy", alphas)
			np.save('..\\data\\' + DATASET + "_" + STATE + "_betas.npy", betas)

			print("File SAVED")
		else:
			print('Current Subject' + str(sub_num)+ ' does not exist')


if __name__ == "__main__":
	main()


