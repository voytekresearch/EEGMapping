""""Script to extract data of interest from the ChildMind dataset."""

import os
import csv
import warnings

import numpy as np
import pandas as pd

import mne

# Import custom project related code
import sys
sys.path.append('../code')
from db import EEGDB

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

    # Define
    AGES = []
    DATA = []

    # Set MNE verbosity level & shut up some warnings
    mne.set_log_level(verbose=False)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Initialize project database object
    data_path = '/Volumes/Data/Data/03-External/Childmind'
    db = EEGDB(data_path=data_path)

    # Load data readme file
    rmd_file = os.path.join(db.data_path, 'EEG', 'MIPDB_PublicFile.csv')
    df = pd.read_csv(rmd_file)

    # Read in list of channel names that are kept in reduced 111 montage
    with open('../data/chans111.csv', 'r') as csv_file:
        reader = csv.reader(csv_file)
        ch_labels = list(reader)[0]

    # Make standard montage
    montage = mne.channels.make_standard_montage('GSN-HydroCel-129')

    # Extract data from all subjects
    subjs = db.check_subjs()
    for cur_subj in subjs:

        # Skip specified subjects
        if cur_subj in SKIP_SUBJS:
            print('\t\tSKIPPING SUBJECT: ', str(cur_subj), '\n\n')
            continue

        # Print status
        print('\n\nRUNNING SUBJECT: ', str(cur_subj))

        # Get subject data files
        data_files, event_files, _ = db.get_subj_files(cur_subj)

        # Get the indices of the resting state data and event files
        try:
            event_ind = [ef.split('_')[1][-3:] for ef in event_files].index('001')
            data_ind = [df.split('.')[0][-3:] for df in data_files].index('001')
        except(ValueError):
            print('Files not found. Can not proceed.')
            continue

        # Get file file path for data file & associated event file
        data_file = data_files[data_ind]
        data_f_name = db.gen_data_path(cur_subj, data_file)

        event_file = event_files[event_ind]
        event_f_name = db.gen_data_path(cur_subj, event_file)

        # Check file names
        assert data_file.split('.')[0][-3:] == '001'
        assert event_file.split('.')[0][-10:-7] == '001'

        # Load data file
        data = np.loadtxt(data_f_name, delimiter=',')

        # Get subject age
        age = df[df['ID'] == cur_subj].Age.values[0]

        # Create the info structure needed by MNE
        info = mne.create_info(ch_labels, S_RATE, 'eeg')
        info.set_montage(montage)

        # Create the MNE Raw data object
        raw = mne.io.RawArray(data, info)

        # Create a stim channel
        stim_info = mne.create_info(['stim'], S_RATE, 'stim')
        stim_raw = mne.io.RawArray(np.zeros(shape=[1, len(raw.times)]), stim_info)

        # Add stim channel to data object
        raw.add_channels([stim_raw], force_update_info=True);

        ## Load Events from File

        # Initialize headers and variable to store event info
        headers = ['type', 'value', 'latency', 'duration', 'urevent']
        evs = np.empty(shape=[0, 3])

        # Load events from csv file
        with open(event_f_name, 'r') as csv_file:

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
        flat_inds = np.mean(raw._data[:111, :], axis=1) == 0
        flat_chans = list(np.array(raw.ch_names[:111])[flat_inds])
        raw.info['bads'] = flat_chans

        # Interpolate bad channels
        raw.interpolate_bads();

        # Set average reference
        raw.set_eeg_reference()
        raw.apply_proj();

        # Get good eeg channel indices
        eeg_chans = mne.pick_types(raw.info, meg=False, eeg=True)

        # Epoch resting eeg data events
        #eo_epochs = mne.Epochs(raw, events=data_evs, event_id={'EO': 20}, tmin=0, tmax=20,
        #                       baseline=None, picks=eeg_chans, preload=True)
        ec_epochs = mne.Epochs(raw, events=data_evs, event_id={'EC': 30}, tmin=5, tmax=35,
                               baseline=None, picks=eeg_chans, preload=True)

        # Get data for specified channel
        data = ec_epochs.get_data()
        cz_data = data[0, raw.ch_names.index('Cz'), :]

        # Collect info of interest from subject
        AGES.append(age)
        DATA.append(cz_data)

    # Save out extracted data
    np.save('ages.npy', np.array(AGES))
    np.save('data.npy', np.array(DATA))


if __name__ == "__main__":
    main()
