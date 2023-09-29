"""Process EEG data from the MIPDB dataset, saving out PSDs from resting EEG data."""

import csv
import warnings
from os.path import join as pjoin

import numpy as np

import mne

# Import custom project related code
import sys
sys.path.append('../code')
from db import EEGDB
from utils import save_pickle
from settings import *

###################################################################################################
###################################################################################################

## SCRIPT SETTINGS

# Run options
SKIP_DONE = False

# Processing Options
COMPUTE_PSDS = False
RUN_FOOOF = False                 # `COMPUTE_PSDS` has to be True to be able to run fooof
EXTRACT_TIMESERIES = True

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

    # Read in list of channel names that are kept in reduced 111 montage
    with open('../data/chans111.csv', 'r') as csv_file:
        reader = csv.reader(csv_file)
        ch_labels = list(reader)[0]

    # Set collection containers for info across subjects
    all_flat_chans = {}

    if RUN_FOOOF:

        # Initialize FOOOFGroup to use for fitting
        fg = FOOOFGroup(*FOOOF_SETTINGS, verbose=False)

        # Save out a settings file
        fg.save(file_name=GROUP + '_fooof_group_settings',
                file_path=PATHS.fooofs_path, save_settings=True)

    if EXTRACT_TIMESERIES:
        n_timepoints_extracted = ((EC_TIMES['tmax']-EC_TIMES['tmin']) * FS) + 1
        extracted_data = np.zeros(\
            #shape=[len(subjs) - len(SKIP_SUBJS), 5, N_CHANNELS, n_timepoints_extracted])
            shape=[len(subjs), 5, N_CHANNELS, n_timepoints_extracted])

    for sub_ind, cur_subj in enumerate(subjs):

        # Print status
        print('\n\nRUNNING SUBJECT (#{}): {}'.format(sub_ind, cur_subj))

        # Skip specified subjects
        if cur_subj in SKIP_SUBJS:
            print('\t\tSKIPPING SUBJECT: ', str(cur_subj), '\n\n')
            continue

        # Skip subject if PSD already calculated
        if SKIP_DONE and cur_subj in done:
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

        # Create channel montage
        montage = mne.channels.make_standard_montage('GSN-HydroCel-129')

        # Create the info structure needed by MNE
        info = mne.create_info(ch_labels, FS, 'eeg')
        info.set_montage(montage)

        # Create the MNE Raw data object
        raw = mne.io.RawArray(data, info)

        # Create a stim channel
        stim_info = mne.create_info(['stim'], FS, 'stim')
        stim_raw = mne.io.RawArray(np.zeros(shape=[1, len(raw.times)]), stim_info)

        # Add stim channel to data object
        raw.add_channels([stim_raw], force_update_info=True)

        ## Load events from file

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
        flat_inds = np.mean(raw._data[:111, :], axis=1) == 0
        flat_chans = list(np.array(raw.ch_names[:111])[flat_inds])
        raw.info['bads'] = flat_chans
        all_flat_chans[cur_subj] = flat_chans

        # Interpolate bad channels
        raw.interpolate_bads()

        # Set average reference
        raw.set_eeg_reference()
        raw.apply_proj()

        # Get good eeg channel indices
        eeg_chans = mne.pick_types(raw.info, meg=False, eeg=True)

        # Epoch resting eeg data events
        eo_epochs = mne.Epochs(raw, events=data_evs, event_id={'EO': 20},
                               tmin=EO_TIMES['tmin'], tmax=EO_TIMES['tmax'],
                               baseline=None, picks=eeg_chans, preload=True)
        ec_epochs = mne.Epochs(raw, events=data_evs, event_id={'EC': 30},
                               tmin=EC_TIMES['tmin'], tmax=EC_TIMES['tmax'],
                               baseline=None, picks=eeg_chans, preload=True)

        if COMPUTE_PSDS:

            spectra_eo = eo_epochs.compute_psd(**PSD_SETTINGS)
            spectra_ec = ec_epochs.compute_psd(**PSD_SETTINGS)

            # ToDo - check & fix
            # Save out PSDs
            # np.savez(pjoin(db.psd_path, str(cur_subj) + '_ec_psds.npz'),
            #          ec_freqs, ec_psds, np.array(ec_epochs.ch_names))
            # np.savez(pjoin(db.psd_path, str(cur_subj) + '_eo_psds.npz'),
            #          eo_freqs, eo_psds, np.array(eo_epochs.ch_names))

            # Print status
            print('\tPSD DATA SAVED FOR SUBJ: ', str(cur_subj), '\n\n')

        # TODO:
        if RUN_FOOOF:
            pass

        # Extract a group collection of time series
        if EXTRACT_TIMESERIES:

            n_blocks = ec_epochs._data.shape[0]
            n_blocks = 5 if n_blocks > 5 else n_blocks
            extracted_data[sub_ind, 0:n_blocks, :, :] = ec_epochs._data[0:n_blocks, :, :]

    # Save out any group level metadata
    #save_pickle(all_flat_chans, 'childmind_interp_chs.p', db.data_path)

    # When done all subjects, save out extracted data
    if EXTRACT_TIMESERIES:
        np.save('MIPDB_extracted_block', extracted_data)

    print('\nProcessing Complete!')

if __name__ == "__main__":
    main()
