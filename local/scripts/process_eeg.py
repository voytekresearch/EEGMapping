"""Preprocess local EEG data for EEGMapping project."""

import os
from os.path import join as pjoin
from pathlib import Path

import numpy as np

import mne
from mne.preprocessing import ICA
from autoreject import AutoReject

from fooof import FOOOFGroup

# Import custom code settings
import sys
sys.path.append('../code')
from settings import *
from db import EEGDB

###################################################################################################
###################################################################################################

## SETTINGS ##

# Set Group to Run
GROUP = 'PBA' # {'rtPB', 'PBA'}

# Processing Options
RUN_ICA = False
RUN_AUTOREJECT = False
COMPUTE_PSDS = False
RUN_FOOOF = False                 # `COMPUTE_PSDS` has to be True to be able to run fooof
EXTRACT_TIMESERIES = True

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
REST_EVENT_ID = list(DATASET_INFO['REST_EVENT_ID'].keys())[0]
TRIAL_EVENT_ID = list(DATASET_INFO['TRIAL_EVENT_ID'].keys())[0]

###################################################################################################
###################################################################################################

def main():

    if RUN_FOOOF:

        # Initialize FOOOFGroup to use for fitting
        fg = FOOOFGroup(*FOOOF_SETTINGS, verbose=False)

        # Save out a settings file
        fg.save(file_name=GROUP + '_fooof_group_settings',
                file_path=PATHS.fooofs_path, save_settings=True)

    if EXTRACT_TIMESERIES:

        extracted_data = np.zeros(shape=[len(SUBJ_NUMS), N_CHANNELS, ((TMAX-TMIN) * FS) + 1])

    for sub_ind, sub in enumerate(SUBJ_NUMS):

        print('\n\nRUNNING SUBJECT:  ' + str(sub) + '\n\n')

        subj_fname = str(sub)+ "_resampled.set"
        full_path = os.path.join(PATHS.data_path, subj_fname)

        if not os.path.exists(full_path):
            print('Current Subject' + str(sub)+ ' does not exist')
            print('Path: ', full_path)
            continue

        # Load data
        eeg_data = mne.io.read_raw_eeglab(full_path, preload=True)

        # Get events from annotations
        events, event_id = mne.events_from_annotations(eeg_data, verbose=False)

        # Set montage
        montage = mne.channels.make_standard_montage('standard_1020')
        eeg_data.set_montage(montage)

        # Set EEG to average reference
        eeg_data.set_eeg_reference()

        ## PRE-PROCESSING: ICA
        if RUN_ICA:

            # ICA settings
            method = 'fastica'
            n_components = 0.99
            random_state = 47
            reject = {'eeg' : 20e-4}

            # Initialize ICA object
            ica = ICA(n_components=n_components, method=method, random_state=random_state)

            # High-pass filter data for running ICA
            eeg_data.filter(l_freq=1., h_freq=None, fir_design='firwin');

            # Fit ICA
            ica.fit(eeg_data, reject=reject)

            # Find components to drop, based on correlation with EOG channels
            drop_inds = []
            for chi in EOG_CHS:
                inds, scores = ica.find_bads_eog(eeg_data, ch_name=chi, threshold=2.5,
                                                 l_freq=1, h_freq=10, verbose=False)
                drop_inds.extend(inds)
            drop_inds = list(set(drop_inds))

            # Set which components to drop, and collect record of this
            ica.exclude = drop_inds

            # Save out ICA solution
            #ica.save(pjoin(PATHS.ica_path, str(sub) + '-ica.fif'))

            # Apply ICA to data
            eeg_data = ica.apply(eeg_data);

        ## EPOCH BLOCKS
        rest_epochs = mne.Epochs(eeg_data, events=events, event_id=event_id[REST_EVENT_ID],
                                 tmin=TMIN, tmax=TMAX, baseline=None, preload=True)
        trial_epochs = mne.Epochs(eeg_data, events=events, event_id=event_id[TRIAL_EVENT_ID],
                                  tmin=TMIN, tmax=TMAX, baseline=None, preload=True)

        ## PRE-PROCESSING: AUTO-REJECT
        if RUN_AUTOREJECT:

            # Initialize and run autoreject across epochs
            ar = AutoReject(n_jobs=4, verbose=False)
            epochs, rej_log = ar.fit_transform(epochs, True)

            # Drop same trials from filtered data
            rest_epochs.drop(rej_log.bad_epochs)
            trial_epochs.drop(rej_log.bad_epochs)

            # Collect list of dropped trials
            dropped_trials[s_ind, 0:sum(rej_log.bad_epochs)] = np.where(rej_log.bad_epochs)[0]

        ## Compute Power Spectra
        if COMPUTE_PSDS:

            spectra_rest = rest_epochs.compute_psd(**PSD_SETTINGS)
            spectra_trial = trial_epochs.compute_psd(**PSD_SETTINGS)

            # ToDo: add saving

        ## Fit spectral models to the computed power spectra
        if RUN_FOOOF:

            rest_psds = np.squeeze(spectra_rest._data[0, :, :])
            for ind, entry in enumerate(rest_psds):
                rest_fooof_psds = rest_psds[ind, :, :]
                fg.fit(spectra_rest.freqs, rest_fooof_psds, FIT_RANGE)
                fg.save(file_name=str(sub) + 'fooof_group_results' + str(ind) ,
                        file_path=PATHS.fooofs_rest_path, save_results=True)

            rest_psds = np.squeeze(spectra_trial._data[0, :, :])
            for ind, entry in enumerate(trial_psds):
                trial_fooof_psds = trial_psds[ind, :, :]
                fg.fit(spectra_trial.freqs, trial_fooof_psds, FIT_RANGE)
                fg.save(file_name=str(sub) + 'fooof_group_results' + str(ind),
                        file_path=PATHS.fooofs_trial_path, save_results=True)

            print('Subject FOOOF results saved')

        # Extract a group collection of time series
        if EXTRACT_TIMESERIES:

            extracted_data[sub_ind, :, :] = rest_epochs._data[0, :, :]

        print('\n\nCOMPLETED SUBJECT:  ' + str(sub) + '\n\n')

    # When done all subjects, save out extracted data
    if EXTRACT_TIMESERIES:
        np.save(GROUP + '_extracted_block', extracted_data)

    print('\nProcessing Complete!')

if __name__ == "__main__":
    main()
