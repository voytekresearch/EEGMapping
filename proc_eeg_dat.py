"""Process EEG data ..."""

import mne
import os
import numpy as np
from fooof import FOOOFGroup
from autoreject import LocalAutoRejectCV
from pathlib import Path

####################################################################################################
####################################################################################################


# Set up paths

# This base path will need updating
base_path = '/Users/luyandamdanda/Documents/Research/EEG_Dat'
save_path = '/Users/luyandamdanda/Documents/Research/Results'

# These should stay the same
subj_dat_num = list(range(3502, 3516))

def main():

    # Initialize fg
    # TODO: add any settings we want to ue
    fg = FOOOFGroup(verbose=False)

    # Save out a settings file
    fg.save(file_name='fooof_group_settings', file_path = save_path, save_settings=True)

    # event dictionary to ensure "Start Block" and "End Block"
    ev_dict = {'Start Block': 1001., 'End Block': 1002., 'Start Labelling Block':1003., 'End Labelling Block':1004}

    # START LOOP
    for sub in subj_dat_num:
        print('Current Subject' + str(sub))

        # load subjec data
        subj_dat_fname = str(sub)+ "_resampled.set"
        full_path = os.path.join(base_path, subj_dat_fname)
        path_check = Path(full_path)
        if path_check.is_file():
            eeg_dat = mne.io.read_raw_eeglab(full_path, event_id=ev_dict, preload=True)


            # set EEG average reference
            eeg_dat.set_eeg_reference()


            events = mne.find_events(eeg_dat)
            event_id = {'Start Labelling Block':1003}



            epochs = mne.Epochs(eeg_dat, events=events, event_id=event_id, tmin = 5, tmax = 125,
                            baseline = None, preload=True)


            # Set montage
            chs = mne.channels.read_montage('standard_1020', epochs.ch_names[:-1])
            epochs.set_montage(chs)

            # Use autoreject to get trial indices to drop
            #ar = LocalAutoRejectCV()
            #epochs = ar.fit_transform(epochs)

            # Calculate PSDs
            psds, freqs = mne.time_frequency.psd_welch(epochs, fmin=3., fmax=40., n_fft=500)


            # FOOOFing Data
            fooof_psds = np.squeeze(psds[0,:,:])

            # Setting frequency range
            freq_range = [2, 40]

            # Run FOOOF across a group of PSDs
            fg.fit(freqs, fooof_psds, freq_range)

            fg.save(file_name= str(sub) + 'fooof_group_results', file_path= save_path, save_results=True)
            print('Subject Saved')
        else:
            print('Current Subject' + str(sub)+ ' does not exist')

if __name__ == "__main__":
    main()
