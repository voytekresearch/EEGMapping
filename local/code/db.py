"""Database and path information for the local EEG data. """

from os.path import join as pjoin

###################################################################################################
###################################################################################################

class EEGDB(object):
    """Class to hold database & path information for local eeg data."""

    def __init__(self, group):

        # Set data paths
        base_data_path = '/Users/tom/Documents/Data/01-Internal/{}/{}-3/processed/EEG/'
        self.data_path = base_data_path.format(group, group)

        # Set base project path
        base_project_path = '/Users/tom/Documents/GitCode/EEGMapping/local/'

        # Set results paths
        base_results = pjoin(base_project_path, 'data')

        # FOOOF paths
        self.fooofs_path = pjoin(base_results, 'fooofs', group)
        #self.fooofs_rest_path = pjoin(self.fooofs_path, 'rest')
        #self.fooofs_trial_path = pjoin(self.fooofs_path, 'trial')
        self.fooofs_rest_path = pjoin(self.fooofs_path, 'rest_fixed')
        self.fooofs_trial_path = pjoin(self.fooofs_path, 'trial_fixed')

        # EEG processing path
        self.eeg_path = pjoin(base_results, 'eeg', group)
        self.ica_path = pjoin(self.eeg_path, 'ICA')
        self.ar_path = pjoin(self.eeg_path, 'AR')

        # Results path
        #self.results_path = pjoin(base_results, 'results')
        self.results_path = pjoin(base_results, 'test')


def clean_files(files):
    """Clean a list of files, removing any hidden files."""

    return list(filter(lambda x: x[0] != '.', files))
