"""Process EEG channel information..."""
import os

import mne
####################################################################################################
####################################################################################################


# This base path will need updating
base_path = 'C:\\Users\\abc\\EEG-MNE'
save_path = 'C:\\Users\\abc\\EEG-MNE'
subj_dat_fname = '3502_resampled.set'

# This should stay the same
chan_dat = 'channel_dat.txt'


# Read in subject listed above
full_path = os.path.join(base_path, subj_dat_fname)
eeg_dat = mne.io.read_raw_eeglab(full_path)


channels = eeg_dat.ch_names


file = open(os.path.join(save_path, chan_dat), 'w')

channels=map(lambda x:x+'\n', channels)
file.writelines(channels)
file.close()