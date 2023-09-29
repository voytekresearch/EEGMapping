"""Settings for local EEG data processing."""

from fooof import Bands
from fooof.data import FOOOFSettings

###################################################################################################
###################################################################################################

## General data settings
FS = 500
N_CHANNELS = 64
EOG_CHS = ['Fp1', 'Fp2']
TMIN = 10
TMAX = 40

## Extracted feature settings
N_FEATS = 3

## Oscillation Band Settings
BANDS = Bands({'theta' : [2, 7],
               'alpha' : [8, 14],
               'beta' : [15, 30]})

## PSD Settings
PSD_SETTINGS = {
    'method' : 'welch',
    'fmin' : 1.,
    'fmax' : 50.,
    'n_fft' : 2000,
    'n_overlap' : 250,
    'n_per_seg' : 500,
}

## FOOOF SETTINGS
FIT_RANGE = (3, 30)
FOOOF_SETTINGS = FOOOFSettings(
    peak_width_limits=[1, 6],
    max_n_peaks=6,
    min_peak_height=0.075,
    peak_threshold=1,
    aperiodic_mode='fixed',
)


## DATASET SPECIFIC INFORMATION
RTPB_INFO = {

    'SUBJ_NUMS' : list(range(3501, 3516)),

    'NUM_REST' : 2,
    'NUM_TRIAL' : 10,

    'REST_EVENT_ID' : {'StartRest' : 2000},
    'TRIAL_EVENT_ID' : {'Start Block' : 3000},
    'BLOCK_EVS' : ['Start Labelling Block', 'Start Block'],

    'EV_DICT' : {

        # Recording Blocks
        'Filt Labelling' : 1000,
        'Thresh Labelling Block' : 1001,

        # Instruction Blocks
        'Start Labelling Block' : 2000,
        'End Labelling Block' : 2001,

        # Rest Blocks
        'Start Block' : 3000,
        'End Block' : 3001,

        # Trial Markers
        'Label_Peak_filt' : 4000,
        'Label_Trough_filt' : 4001,
        'Markers0' : 4002,
        'Markers1' : 4002,
        'MissTrial': 4003,
        'HitTrial' : 4004
    }
}


## PBA Dataset Information
PBA_INFO = {

    'SUBJ_NUMS' : list(range(3001, 3015)),
    'NUM_REST' : 1,
    'NUM_TRIAL' : 12,

    'REST_EVENT_ID' : {'Rest_Start' : 3000},
    'TRIAL_EVENT_ID' : {'Exp_Block_Start' : 5000},
    'BLOCK_EVS' : ['Rest_Start', 'Exp_Block_Start'],

    'EV_DICT' : {

        # Recording Blocks
        'Recording_Start': 1000,
        'Recording_End': 1001,

        # Instruction Blocks
        'Instructions_Start' : 2000,
        'Instructions_End' : 2001,

        # Rest Blocks
        'Rest_Start' : 3000,
        'Rest_End' : 3001,

        # Threshold Blocks
        'Thresh_Block_Start' : 4000,
        'Thresh_Block_End' : 4001,

        # Experiment Blocks
        'Exp_Block_Start' : 5000,
        'Exp_Block_End' : 5001,

        # Trial Markers
        'Con_{}_Loc_{}' : 6000,
        'Fix_On' : 6001,
        'Lines_On' : 6002,
        'Flash' : 6003,
        'Catch_Trail' : 6004,
        'Saw' : 6005,
        'Missed' : 6006
    }
}
