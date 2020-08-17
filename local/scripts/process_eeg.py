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
GROUP = 'rtPB' # {'rtPB', 'PBA'}

# Processing Options
RUN_ICA = True
RUN_AUTOREJECT = False

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
EV_DICT = DATASET_INFO['EV_DICT']
REST_EVENT_ID = DATASET_INFO['REST_EVENT_ID']
TRIAL_EVENT_ID = DATASET_INFO['TRIAL_EVENT_ID']
BLOCK_EVS = DATASET_INFO['BLOCK_EVS']

###################################################################################################
###################################################################################################

def main():

    # Initialize FOOOFGroup to use for fitting
    fg = FOOOFGroup(*FOOOF_SETTINGS, verbose=False)

    # Save out a settings file
    fg.save(file_name=GROUP + '_fooof_group_settings',
            file_path=PATHS.fooofs_path, save_settings=True)

    for sub in SUBJ_NUMS:

        print('Current Subject' + str(sub))

        # load subject data
        subj_fname = str(sub)+ "_resampled.set"
        full_path = os.path.join(PATHS.data_path, subj_fname)
        path_check = Path(full_path)

        if path_check.is_file():

            #eeg_data = mne.io.read_raw_eeglab(full_path, event_id_func=None, preload=True)
            eeg_data = mne.io.read_raw_eeglab(full_path, preload=True)
            evs = mne.io.eeglab.read_events_eeglab(full_path, EV_DICT)

            new_evs = np.empty(shape=(0, 3))

            for ev_label in BLOCK_EVS:
                ev_code = EV_DICT[ev_label]
                temp = evs[evs[:, 2] == ev_code]
                new_evs = np.vstack([new_evs, temp])

            eeg_data.add_events(new_evs)

            # Set EEG to average reference
            eeg_data.set_eeg_reference()

            ## PRE-PROCESSING: ICA
            if RUN_ICA:

                # ICA settings
                method = 'fastica'
                n_components = 0.99
                random_state = 47
                reject = {'eeg': 20e-4}

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
                ica.save(pjoin(PATHS.ica_path, str(sub) + '-ica.fif'))

                # Apply ICA to data
                eeg_data = ica.apply(eeg_data);

            ## EPOCH BLOCKS
            events = mne.find_events(eeg_data)
            rest_epochs = mne.Epochs(eeg_data, events=events, event_id=REST_EVENT_ID,
                                     tmin=5, tmax=125, baseline=None, preload=True)
            trial_epochs = mne.Epochs(eeg_data, events=events, event_id=TRIAL_EVENT_ID,
                                      tmin=5, tmax=125, baseline=None, preload=True)

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

            # Set montage
            chs = mne.channels.read_montage('standard_1020', rest_epochs.ch_names[:-1])
            rest_epochs.set_montage(chs)
            trial_epochs.set_montage(chs)

            # Calculate power spectra
            rest_psds, rest_freqs = mne.time_frequency.psd_welch(rest_epochs,
                fmin=1., fmax=50., n_fft=2000, n_overlap=250, n_per_seg=500)
            trial_psds, trial_freqs = mne.time_frequency.psd_welch(trial_epochs,
                fmin=1., fmax=50., n_fft=2000, n_overlap=250, n_per_seg=500)

            # Setting frequency range
            freq_range = [3, 30]

            ## FOOOF the Data

            # Rest Data
            for ind, entry in enumerate(rest_psds):
                rest_fooof_psds = rest_psds[ind, :, :]
                fg.fit(rest_freqs, rest_fooof_psds, freq_range)
                fg.save(file_name= str(sub) + 'fooof_group_results' + str(ind) ,
                        file_path=PATHS.fooofs_rest_path, save_results=True)

            # Trial Data
            for ind, entry in enumerate(trial_psds):
                trial_fooof_psds = trial_psds[ind, :, :]
                fg.fit(trial_freqs, trial_fooof_psds, freq_range)
                fg.save(file_name= str(sub) + 'fooof_group_results' + str(ind),
                        file_path=PATHS.fooofs_trial_path, save_results=True)

            print('Subject Saved')

        else:

            print('Current Subject' + str(sub)+ ' does not exist')
            print(path_check)

    print('Pre-processing Complete')

if __name__ == "__main__":
    main()
