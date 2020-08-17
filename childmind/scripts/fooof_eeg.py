"""Run FOOOF on the MIPDB dataset."""

from os.path import join as pjoin

import numpy as np

from fooof import FOOOFGroup
from fooof.data import FOOOFSettings

# Import custom project related code
import sys
sys.path.append('../')
from code.db import EEGDB
from code.io import save_pickle

###################################################################################################
###################################################################################################

F_RANGE = [3, 35]
FOOOF_SETTINGS = FOOOFSettings(
    peak_width_limits=[1, 8],
    max_n_peaks=6,
    min_peak_height=0.05,
    peak_threshold=2,
    aperiodic_mode='fixed')

###################################################################################################
###################################################################################################

def main():

    # Get project database objects, and list of available subjects
    db = EEGDB()
    subjs = db.get_psd_subjs()

    # Fit FOOOF to PSDs averaged across rest epochs
    fg = FOOOFGroup(*FOOOF_SETTINGS)

    for cur_subj in subjs:

        print('\n\n\nFOOOFING DATA FOR SUBJ: ', str(cur_subj), '\n\n\n')

        ## LOAD DATA

        temp_eo = np.load(pjoin(db.psd_path, str(cur_subj) + '_eo_psds.npz'))
        eo_freqs = temp_eo['arr_0']
        eo_psds = temp_eo['arr_1']

        temp_ec = np.load(pjoin(db.psd_path, str(cur_subj) + '_ec_psds.npz'))
        ec_freqs = temp_ec['arr_0']
        ec_psds = temp_ec['arr_1']

        ## ALL EPOCHS

        # Fit FOOOF to PSDs from each epoch
        eo_fgs = []
        for ep_psds in eo_psds:
            fg.fit(eo_freqs, ep_psds, F_RANGE)
            eo_fgs.append(fg.copy())
        exp_eo = [fg.get_params('aperiodic_params', 'exponent') for fg in eo_fgs]

        ec_fgs = []
        for ep_psds in ec_psds:
            fg.fit(ec_freqs, ep_psds, F_RANGE)
            ec_fgs.append(fg.copy())
        exp_ec = [fg.get_all_data('aperiodic_params', 'exponent') for fg in ec_fgs]

        ## AVERAGE ACROSS EPOCHS

        eo_avg_psds = np.mean(eo_psds, axis=0)
        fg.fit(eo_freqs, eo_avg_psds, F_RANGE)
        exp_eo_avg = fg.get_params('aperiodic_params', 'exponent')

        ec_avg_psds = np.mean(ec_psds, axis=0)
        fg.fit(ec_freqs, ec_avg_psds, F_RANGE)
        exp_ec_avg = fg.get_params('aperiodic_params', 'exponent')

        ## COLLECT & SAVE

        # Collect data together
        subj_data = {
            'ID' : cur_subj,
            'exp_eo_avg' : exp_eo_avg,
            'exp_ec_avg' : exp_ec_avg,
            'exp_eo' : exo_eo,
            'exp_ec' : exp_ec
        }

        # Save out exponent data
        f_name = str(cur_subj) + '_fooof.p'
        save_pickle(subj_data, f_name, db.fooof_path)

        # Print status
        print('\n\n\nFOOOF DATA SAVED AND FINISHED WITH SUBJ: ', str(cur_subj), '\n\n\n')


if __name__ == "__main__":
    main()
