"""Preprocess PBA EEG data for EEGMapping project."""

import os
from pathlib import Path

import numpy as np

import mne
from mne.preprocessing import ICA

from fooof import FOOOFGroup
from autoreject import AutoReject

###################################################################################################
###################################################################################################
## SETTINGS ##

## Processing Options ##
RUN_ICA = True
RUN_AUTOREJECT = True

## Paths ##
# can be changed if needed #
BASE_PATH = 'D:\\abc\\Documents\\Research\\PBA_Data'
SAVE_PATH = 'D:\\abc\\Documents\\Research\\Results'
REST_SAVE_PATH = 'D:\\abc\\Documents\\Research\\Results\\Rest_Results'
TRIAL_SAVE_PATH = 'D:\\abc\\Documents\\Research\\Results\\Trial_Results'

##  Subject Numbers ##
# should remain the same, but can be changed if needed #
SUBJ_DAT_NUM = list(range(3001, 3015))

## Event Dictionary ##
EV_DICT = {
    # Recording Blocks
    'Recording_Start': 1000,
    'Recording_End': 1001,

    # Instruction Blocks
    'Instructions_Start':2000,
    'Instructions_End': 2001,

    # Rest Blocks
    'Rest_Start':3000,
    'Rest_End':3001,

    # Threshold Blocks
    'Thresh_Block_Start':4000,
    'Thresh_Block_End': 4001,

    # Experiment Blocks
    'Exp_Block_Start':5000,
    'Exp_Block_End': 5001,

    # Trial Markers
    'Con_{}_Loc_{}': 6000,
    'Fix_On':6001,
    'Lines_On':6002,
    'Flash':6003,
    'Catch_Trail': 6004,
    'Saw':6005,
    'Missed':6006
    }

# Set channel groups
EOG_CHS = ['Fp1', 'Fp2']

###################################################################################################
###################################################################################################

def main():

    # Initialize fg
    fg = FOOOFGroup(verbose=False, peak_width_limits=[1, 6], min_peak_amplitude=0.075,
                    max_n_peaks=6, peak_threshold=1)

    # Save out a settings file
    fg.save(file_name='PBA_fooof_group_settings', file_path=SAVE_PATH, save_settings=True)

    # START LOOP
    for sub in SUBJ_DAT_NUM:
        print('Current Subject' + str(sub))

        # load subjec data
        subj_dat_fname = str(sub)+ "_resampled.set"
        full_path = os.path.join(BASE_PATH, subj_dat_fname)
        path_check = Path(full_path)

        if path_check.is_file():

            eeg_dat = mne.io.read_raw_eeglab(full_path, event_id_func=None, preload=True)
            evs = mne.io.eeglab.read_events_eeglab(full_path, EV_DICT)

            new_evs = np.empty(shape=(0, 3))

            for ev_label in ['Rest_Start', 'Exp_Block_Start']:
                ev_code = EV_DICT[ev_label]
                temp = evs[evs[:, 2] == ev_code]
                new_evs = np.vstack([new_evs, temp])

            eeg_dat.add_events(new_evs)

            # set EEG average reference
            eeg_dat.set_eeg_reference()

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
                eeg_dat.filter(l_freq=1., h_freq=None, fir_design='firwin');

                # Fit ICA
                ica.fit(eeg_dat, reject=reject)

                # Find components to drop, based on correlation with EOG channels
                drop_inds = []
                for chi in EOG_CHS:
                    inds, scores = ica.find_bads_eog(eeg_dat, ch_name=chi, threshold=2.5,
                                                     l_freq=1, h_freq=10, verbose=False)
                    drop_inds.extend(inds)
                drop_inds = list(set(drop_inds))

                # Set which components to drop, and collect record of this
                ica.exclude = drop_inds
                dropped_components[s_ind, 0:len(drop_inds)] = drop_inds

                # Save out ICA solution
                ica.save(pjoin(RES_PATH, 'ICA', subj_label + '-ica.fif'))

                # Apply ICA to data
                eeg_dat = ica.apply(eeg_dat);


            events = mne.find_events(eeg_dat)
            rest_event_id = {'Rest_Start':3000}
            trial_event_id = {'Exp_Block_Start':5000}

            epochs = mne.Epochs(eeg_dat, events=events, tmin=5, tmax=125,
                                baseline = None, preload=True)
            rest_epochs = mne.Epochs(eeg_dat, events=events, event_id=rest_event_id, tmin=5, tmax=125,
                                     baseline = None, preload=True)
            trial_epochs = mne.Epochs(eeg_dat, events=events, event_id=trial_event_id, tmin=5, tmax=125,
                                      baseline = None, preload=True)

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

            # Calculate PSDs
            rest_psds, rest_freqs = mne.time_frequency.psd_welch(rest_epochs,
                fmin=1., fmax=50., n_fft=2000, n_overlap=250, n_per_seg=500)
            trial_psds, trial_freqs = mne.time_frequency.psd_welch(trial_epochs,
                fmin=1., fmax=50., n_fft=2000, n_overlap=250, n_per_seg=500)

            # Setting frequency range
            freq_range = [3, 32]

            # FOOOFing Data
            # Rest
            for ind, entry in enumerate(rest_psds):
                rest_fooof_psds = rest_psds[ind,:,:]
                fg.fit(rest_freqs, rest_fooof_psds, freq_range)
                fg.save(file_name= str(sub) + 'fooof_group_results' + str(ind),
                        file_path=REST_SAVE_PATH, save_results=True)

            for ind, entry in enumerate(trial_psds):
                trial_fooof_psds = trial_psds[ind,:,:]
                fg.fit(trial_freqs, trial_fooof_psds, freq_range)
                fg.save(file_name= str(sub) + 'fooof_group_results' + str(ind),
                        file_path=TRIAL_SAVE_PATH, save_results=True)

            print('Subject Saved')

        else:
            print('Current Subject' + str(sub)+ ' does not exist')
            print(path_check)

    print('Pre-processing Complete')
if __name__ == "__main__":
    main()
