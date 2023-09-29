"""Settings for ChildMind EEG data processing."""

#from fooof import Bands
#from fooof.data import FOOOFSettings

###################################################################################################
###################################################################################################

# EEG Data Settings
FS = 500
N_CHANNELS = 111

# Subject subjects
SKIP_SUBJS = ['A00054488', 'A00054866', 'A00055623', 'A00055628',
              'A00056716', 'A00056733', 'A00062219']

# Event Codes
EVENT_IDS = {
    'EO' : 20,
    'EC' : 30,
}

# Time definitions
EO_TIMES = {
    'tmin' : 2,
    'tmax' : 18,
}

EC_TIMES = {
    'tmin' : 5,
    'tmax' : 35,
}

# PSD Settings
PSD_SETTINGS = {
    'method' : 'welch',
    'fmin' : 2.,
    'fmax' : 40.,
    'n_fft' : 1000,
    'n_overlap' : 250,
    'verbose' : False,
}
