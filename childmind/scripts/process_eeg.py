"""Process EEG data from the MIPDB dataset, saving out PSDs from resting EEG data."""

import csv
import warnings
from os.path import join as pjoin

import numpy as np

import mne

# Import custom project related code
import sys
sys.path.append('../')
from code.db import EEGDB

###################################################################################################
###################################################################################################

# Settings
S_RATE = 500

# New skip subjects
SKIP_SUBJS = ['A00055628', 'A00062219', 'A00055623', 'A00056716',
              'A00056733', 'A00054866', 'A00054488']

###################################################################################################
###################################################################################################

def main():

    # Set warnings & MNE verbosity level
    mne.set_log_level(verbose=False)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Get project database objects, and list of available subjects
    db = EEGDB()
    subjs = db.check_subjs()
    done = db.get_psd_subjs()

    for cur_subj in subjs:

        # Print status
        print('\n\nRUNNING SUBJECT: ', str(cur_subj))

        # Skip specified subjects
        if cur_subj in SKIP_SUBJS:
            print('\t\tSKIPPING SUBJECT: ', str(cur_subj), '\n\n')
            continue

        # Skip subject if PSD already calculated
        if cur_subj in done:
            print('\t\tSUBJECT ALREADY RUN: ', str(cur_subj), '\n\n')
            continue

        # Get subject data files
        data_files, event_files, _ = db.get_subj_files(cur_subj)

        # Get the indices of the resting state data and event files
        try:
            event_ind = [ef.split('_')[1][-3:] for ef in event_files].index('001')
            data_ind = [df.split('.')[0][-3:] for df in data_files].index('001')
        except(ValueError):
            print('\tFiles not found. Can not proceed.')
            continue

        # Get file file path for data file & associated event file
        data_file = data_files[data_ind]
        data_fname = db.gen_data_path(cur_subj, data_file)

        event_file = event_files[event_ind]
        event_fname = db.gen_data_path(cur_subj, event_file)

        # Check file names
        assert data_file.split('.')[0][-3:] == '001'
        assert event_file.split('.')[0][-10:-7] == '001'

        # Get file file path for data file & associated event file
        data_fname = db.gen_data_path(cur_subj, data_file)
        event_fname = db.gen_data_path(cur_subj, event_file)

        # Load data file
        data = np.loadtxt(data_fname, delimiter=',')

        # Read in list of channel names that are kept in reduced 111 montage
        with open('../data/chans111.csv', 'r') as csv_file:
            reader = csv.reader(csv_file)
            ch_labels = list(reader)[0]

        # Read montage, reduced to 111 channel selection
        montage = mne.channels.read_montage('GSN-HydroCel-129', ch_names=ch_labels)

        # Create the info structure needed by MNE
        info = mne.create_info(ch_labels, S_RATE, 'eeg', montage)

        # Create the MNE Raw data object
        raw = mne.io.RawArray(data, info)

        # Create a stim channel
        stim_info = mne.create_info(['stim'], S_RATE, 'stim')
        stim_raw = mne.io.RawArray(np.zeros(shape=[1, len(raw._times)]), stim_info)

        # Add stim channel to data object
        raw.add_channels([stim_raw], force_update_info=True)

        # Load events from file
        # Initialize headers and variable to store event info
        headers = ['type', 'value', 'latency', 'duration', 'urevent']
        evs = np.empty(shape=[0, 3])

        # Load events from csv file
        with open(event_fname, 'r') as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:

                # Skip the empty rows & the header rows
                if row == []: continue
                if row[0] == 'type': continue

                # Collect actual event data rows
                evs = np.vstack((evs, np.array([int(row[2]), 0, int(row[0])])))

            # Drop any events that are outside the recorded EEG range
            evs = evs[np.invert(evs[:, 0] > data.shape[1])]

        # Add events to data object
        raw.add_events(evs, stim_channel='stim')

        # Check events
        data_evs = mne.find_events(raw)

        # Find flat channels and set them as bad
        flat_chans = np.mean(raw._data[:111, :], axis=1) == 0
        raw.info['bads'] = list(np.array(raw.ch_names[:111])[flat_chans])

        # Interpolate bad channels
        raw.interpolate_bads()

        # Set average reference
        raw.set_eeg_reference()
        raw.apply_proj()

        # Get good eeg channel indices
        eeg_chans = mne.pick_types(raw.info, meg=False, eeg=True)

        # Epoch resting eeg data events
        eo_epochs = mne.Epochs(raw, events=data_evs, event_id={'EO': 20}, tmin=2, tmax=18,
                               baseline=None, picks=eeg_chans, preload=True)
        ec_epochs = mne.Epochs(raw, events=data_evs, event_id={'EC': 30}, tmin=5, tmax= 35,
                               baseline=None, picks=eeg_chans, preload=True)

        # Calculate PSDs - EO Data
        eo_psds, eo_freqs = mne.time_frequency.psd_welch(eo_epochs, fmin=2., fmax=40., n_fft=1000,
                                                         n_overlap=250, verbose=False)

        # Calculate PSDs - EC Data
        ec_psds, ec_freqs = mne.time_frequency.psd_welch(ec_epochs, fmin=2., fmax=40., n_fft=1000,
                                                         n_overlap=250, verbose=False)

        # Save out PSDs
        np.savez(pjoin(db.psd_path, str(cur_subj) + '_ec_psds.npz'),
                 ec_freqs, ec_psds, np.array(ec_epochs.ch_names))
        np.savez(pjoin(db.psd_path, str(cur_subj) + '_eo_psds.npz'),
                 eo_freqs, eo_psds, np.array(eo_epochs.ch_names))

        # Print status
        print('\tPSD DATA SAVED FOR SUBJ: ', str(cur_subj), '\n\n')


if __name__ == "__main__":
    main()
