"""Analyzing local EEG data for EEGMapping project."""

import fnmatch
import pickle
from pathlib import Path
from os.path import join as pjoin

import numpy as np

import mne

from fooof import FOOOFGroup
from fooof.analysis.periodic import get_band_peak

import sys
sys.path.append('../code')
from settings import *
from db import EEGDB

###################################################################################################
###################################################################################################

## SETTINGS
GROUP = 'PBA'  # {'rtPB', 'PBA'}
STATE = 'rest'  # {'rest', 'trial'}

###################################################################################################
###################################################################################################

# Set paths for the dataset
PATHS = EEGDB(GROUP)

if GROUP == 'rtPB':
    DATASET_INFO = RTPB_INFO

if GROUP == 'PBA':
    DATASET_INFO = PBA_INFO

# Unpack dataset information
SUBJ_NUMS = DATASET_INFO['SUBJ_NUMS']
N_SUBJECTS = len(SUBJ_NUMS)
NUM_BLOCKS = DATASET_INFO['NUM_' + STATE.upper()]

if STATE =='rest':
    RES_PATH = PATHS.fooofs_rest_path

if STATE =='trial':
    RES_PATH = PATHS.fooofs_trial_path

###################################################################################################
###################################################################################################

def main():

    # Initialize a FOOOFGroup object to use
    fg = FOOOFGroup()

    # Create dictionary to store results
    exponent_results = np.zeros(shape=[N_SUBJECTS, NUM_BLOCKS, N_CHANNELS, 2])
    results = {}

    for band_name in BANDS.labels:
        results[band_name] = np.zeros(shape=[N_SUBJECTS, NUM_BLOCKS, N_CHANNELS, N_FEATS])

    for sub_index, subj_num in enumerate(SUBJ_NUMS):
        print('Current Subject Results: ' + str(subj_num))

        # load rest results data
        for block in range(0, NUM_BLOCKS):

            subj_file = str(subj_num) + "fooof_group_results" + str(block) + ".json"

            full_path = pjoin(RES_PATH, subj_file)

            print(full_path)

            if Path(full_path).is_file():
                fg.load(file_name=subj_file, file_path=RES_PATH)

            if not fg.group_results:
                print('Current Subject Results: ' + str(subj_num) + ' block: ' +
                      str(block) + ' failed to load.')
            else:
                print('Current Subject Results: ' +  str(subj_num) + ' block: ' +
                      str(block) + ' successfully loaded.')

            for ind, res in enumerate(fg):

                exponent_results[sub_index, block, ind, :] = res.aperiodic_params

                for band_label, band_range in BANDS:
                    results[band_label][sub_index, block, ind,  :] = \
                        get_band_peak(res.peak_params, band_range, True)

    # Save out processed results
    with open(pjoin(PATHS.results_path, GROUP + '_' + STATE + '_exp.pkl'), 'wb') as exp_output:
        pickle.dump(exponent_results, exp_output)

    with open(pjoin(PATHS.results_path, GROUP + '_' + STATE + '_results.pkl'), 'wb') as output:
        pickle.dump(results, output)

    print("File Saved.")


if __name__ == "__main__":
    main()
