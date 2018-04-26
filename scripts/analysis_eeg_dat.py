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
####################################################################################################


# Set up paths
# This results and analysis path will need updating
rest_results_path = 'D:\\abc\\Documents\\Research\\Results\\Rest_Results'
trial_results_path = 'D:\\abc\\Documents\\Research\\Results\\Trial_Results'

	# These should stay the same
subj_dat_num1 =list(range(3001, 3015))
subj_dat_num2 = list(range(3502, 3516))
all_subj = subj_dat_num1 + subj_dat_num2

def main():



		# GET LIST OF FILES
	rest_f_names = fnmatch.filter(os.listdir(rest_results_path), '*fooof_group_results?.json')
	trial_f_names = fnmatch.filter(os.listdir(trial_results_path), '*fooof_group_results?.json')
	rest_n_subjects = len(rest_f_names)
	trial_n_subjects = len(trial_f_names)

	#number of channels
	n_channels = 64

	# Set up indexes for accessing data, for convenience
	cf_ind, am_ind, bw_ind = 0, 1, 2

	# Define bands of interest
	theta_band = [2, 7]
	alpha_band = [7, 14]
	beta_band = [15, 30]

	# Initialize 3D group arrays
	rest_group_cube_results_theta = np.empty(shape=[n_channels, 3, rest_n_subjects])
	rest_group_cube_results_alpha = np.empty(shape=[n_channels, 3, rest_n_subjects])
	rest_group_cube_results_beta = np.empty(shape=[n_channels, 3, rest_n_subjects])


	trial_group_cube_results_theta = np.empty(shape=[n_channels, 3, trial_n_subjects])
	trial_group_cube_results_alpha = np.empty(shape=[n_channels, 3, trial_n_subjects])
	trial_group_cube_results_beta = np.empty(shape=[n_channels, 3, trial_n_subjects])



	# START LOOP
	#for sub in subj_dat_num:
	for subj_ind, subj_f_name in enumerate(rest_f_names):
		subj_file = os.path.splitext(subj_f_name)[0]
		print('Current Subject Results: ' + subj_file)

		# load results data
		fg = FOOOFGroup()
		full_path = os.path.join(rest_results_path, subj_f_name)
		path_check = Path(full_path)
		if path_check.is_file():
			fg.load(file_name=subj_file, file_path=rest_results_path)

			if not fg.group_results:
				print('Current Subject Results: ' + subj_f_name + " failed to load")
			else:
				print('Current Subject Results: ' + subj_f_name + " successfully loaded")


			# Initializae the 3d array
			#ind_cube_results = np.empty(n_channels, 3, 4)

			# Initialize arrays to store band oscillations - 2D arrays per band, per subj
			thetas = np.empty(shape=[n_channels, 3])
			alphas = np.empty(shape=[n_channels, 3])
			betas = np.empty(shape=[n_channels, 3])


			# Extract all alpha oscillations from FOOOFGroup results
			#  Note that this preserves information about which PSD each oscillation comes from
			for ind, res in enumerate(fg):
				# TODO: adding these into the matrices that are initialized above
				thetas[ind, :] = get_band_peak(res.peak_params, theta_band, True)
				alphas[ind, :] = get_band_peak(res.peak_params, alpha_band, True)
				betas[ind, :] = get_band_peak(res.peak_params, beta_band, True)


			# COLLECT INTO GROUP
			rest_group_cube_results_theta[:, :, subj_ind] = thetas
			rest_group_cube_results_alpha[:, :, subj_ind] = alphas
			rest_group_cube_results_beta[:, :, subj_ind] = betas



			# Save out matrices
			np.save('..\\data\\rest_theta_group.npy', rest_group_cube_results_theta)
			np.save('..\\data\\rest_alpha_group.npy', rest_group_cube_results_alpha)
			np.save('..\\data\\rest_beta_group.npy', rest_group_cube_results_beta)

			print("File SAVED")
		else:
			print('Current Subject' + str(sub)+ ' does not exist')

	for subj_ind, subj_f_name in enumerate(trial_f_names):
		subj_file = os.path.splitext(subj_f_name)[0]
		print('Current Subject Results: ' + subj_file)

		# load results data
		fg = FOOOFGroup()
		full_path = os.path.join(trial_results_path, subj_f_name)
		path_check = Path(full_path)
		if path_check.is_file():
			fg.load(file_name=subj_file, file_path=trial_results_path)

			if not fg.group_results:
				print('Current Subject Results: ' + subj_f_name + " failed to load")
			else:
				print('Current Subject Results: ' + subj_f_name + " successfully loaded")

			# Initializae the 3d array
			#ind_cube_results = np.empty(n_channels, 3, 4)

			# Initialize arrays to store band oscillations - 2D arrays per band, per subj
			thetas = np.empty(shape=[n_channels, 3])
			alphas = np.empty(shape=[n_channels, 3])
			betas = np.empty(shape=[n_channels, 3])


			# Extract all alpha oscillations from FOOOFGroup results
			#  Note that this preserves information about which PSD each oscillation comes from
			for ind, res in enumerate(fg):
				# TODO: adding these into the matrices that are initialized above
				thetas[ind, :] = get_band_peak(res.peak_params, theta_band, True)
				alphas[ind, :] = get_band_peak(res.peak_params, alpha_band, True)
				betas[ind, :] = get_band_peak(res.peak_params, beta_band, True)


			# COLLECT INTO GROUP
			trial_group_cube_results_theta[:, :, subj_ind] = thetas
			trial_group_cube_results_alpha[:, :, subj_ind] = alphas
			trial_group_cube_results_beta[:, :, subj_ind] = betas



			# Save out matrices
			np.save('..\\data\\trial_theta_group.npy', trial_group_cube_results_theta)
			np.save('..\\data\\trial_alpha_group.npy', trial_group_cube_results_alpha)
			np.save('..\\data\\trial_beta_group.npy', trial_group_cube_results_beta)

			print("File SAVED")
		else:
			print('Current Subject' + str(sub)+ ' does not exist')

if __name__ == "__main__":
	main()



