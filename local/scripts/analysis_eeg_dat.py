"""Analyzing EEG data ..."""

import mne
import os
import numpy as np
import fnmatch
from fooof import FOOOFGroup
from fooof.analysis import *
from pathlib import Path
import pickle

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
bands = {'theta': [2,7],
			'alpha': [8,14],
			'beta': [15,30]}

def main():
	#Create dictionary to store results
	slope_results = np.zeros(shape=[n_subjects, num_blocks, n_channels, 2])
	results = {}
	for band_name in bands.keys():
		results[band_name] = np.zeros(shape=[n_subjects, num_blocks, n_channels, n_feats])

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
				print('Current Subject Results: ' + str(sub_num) + " block:" + str(block) + " failed to load")
			else:
				print('Current Subject Results: ' +  str(sub_num) + " block" + str(block) + " successfully loaded")

			for ind, res in enumerate(fg):
				slope_results[sub_index, block, ind, :] = res.background_params
				for band_label, band_range in bands.items():
					results[band_label][sub_index, block, ind,  :] = get_band_peak(res.peak_params, band_range, True)

			
	# Save out matrices
	# Update to save out files using DATASET and STATE
	slope_output = open('..\\data\\analysis\\' + DATASET + "_" + STATE + "_slope_results.pkl" ,'wb')
	pickle.dump(slope_results, slope_output)
	slope_output.close()

	output = open('..\\data\\analysis\\' + DATASET + "_" + STATE + "_results.pkl" ,'wb')
	pickle.dump(results, output)
	output.close()
	print("File SAVED")

if __name__ == "__main__":
	main()


