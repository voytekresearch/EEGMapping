"""Process EEG data ..."""

import mne
import os
import numpy as np
from fooof import FOOOFGroup
#from autoreject import LocalAutoRejectCV
from pathlib import Path

####################################################################################################
####################################################################################################


# Set up paths

# This base path will need updating
base_path = 'E:\\PBA_Data'
save_path = 'C:\\Users\\abc\\Documents\\Research'

# These should stay the same
subj_dat_num = list(range(3001, 3014))

def main():

    # Initialize fg
    # TODO: add any settings we want to ue
    fg = FOOOFGroup(verbose=False)

    # Save out a settings file
    fg.save(file_name='PBA_fooof_group_settings', file_path = save_path, save_settings=True)

    # event dictionary to ensure "Start Block" and "End Block"
    ev_dict = {'Rest_Start': 1001., 'Rest_End': 1002}

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
            event_id = {'Rest_Start':1001}



            epochs = mne.Epochs(eeg_dat, events=events, event_id=event_id, tmin = 5, tmax = 125,
                            baseline = None, preload=True)


            # Set montage
            chs = mne.channels.read_montage('standard_1020', epochs.ch_names[:-1])
            epochs.set_montage(chs)

            # Use autoreject to get trial indices to drop
            #ar = LocalAutoRejectCV()
            #epochs = ar.fit_transform(epochs)

            # Calculate PSD
            psd, freqs = mne.time_frequency.psd_welch(epochs, fmin=1., fmax=50., n_fft=1000, n_overlap=500)


            # FOOOFing Data
            fooof_psd = np.squeeze(psd[0,:,:])

            # Setting frequency range
            freq_range = [2, 40]

            # Run FOOOF across a group of PSDs
            fg.fit(freqs, fooof_psd, freq_range)

            fg.save(file_name= str(sub) + 'fooof_group_results', file_path= save_path, save_results=True)
            print('Subject Saved')
        else:
            print('Current Subject' + str(sub)+ ' does not exist')

if __name__ == "__main__":
    main()
