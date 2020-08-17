"""Database related organization and utilities for EEGMapping on ChildMind data."""

import os

###################################################################################################
###################################################################################################

class EEGDB(object):
    """Class to hold database information for ChildMind data.

    Attributes
    ----------
    project_path : str
        Base path for the project.
    data_path : str
        Path to all data.
    subjs_path : str
        Path to EEG subjects data.
    """

    def __init__(self, gen_paths=True):
        """Initialize EEGDB object."""

        # Set base path for data
        self.data_path = ("/Users/tom/Documents/Data/03-External/Childmind")

        # Initialize paths
        self.subjs_path = str()
        self.psd_path = str()
        self.fooof_path = str()

        # Generate project paths
        if gen_paths:
            self.gen_paths()


    def gen_paths(self):
        """Generate paths."""

        self.subjs_path = os.path.join(self.data_path, 'EEG', 'Subjs')
        self.psd_path = os.path.join(self.data_path, 'psds')
        self.fooof_path = os.path.join(self.data_path, 'fooof')


    def check_subjs(self):
        """Check which subjects are available in database."""

        subjs = _clean_files(os.listdir(self.subjs_path))

        return subjs


    def check_psd(self):
        """Get the available PSD files."""

        psd_files = _clean_files(os.listdir(self.psd_path))

        return psd_files


    def check_fooof(self):
        """Check which FOOOF files are available in the database."""

        fooof_files = _clean_files(os.listdir(self.fooof_path))

        return fooof_files


    def get_subj_files(self, subj_number):
        """Get the preprocessed, EEG data file names (csv format) a specified subject."""

        data_dir = os.path.join(self.subjs_path, subj_number,
                               'EEG', 'preprocessed', 'csv_format')
        files = os.listdir(data_dir)

        eeg = [fi for fi in files if 'events' not in fi and 'channels' not in fi]
        evs = [fi for fi in files if 'events' in fi]
        chs = [fi for fi in files if 'channels' in fi]

        return eeg, evs, chs


    def get_psd_subjs(self):
        """Get a list of subject numbers for whom PSDs are calculated."""

        psd_files = self.check_psd()

        return [fi.split('_')[0] for fi in psd_files]


    def get_fooof_subjs(self):
        """Get a list of subject number for whom FOOOF results are calculated."""

        fooof_files = self.check_fooof()

        return [fi.split('_')[0] for fi in fooof_files]


    def gen_data_path(self, subj_number, data_file):
        """Generate full file path to a data file."""

        return os.path.join(self.subjs_path, subj_number,
                            'EEG', 'preprocessed', 'csv_format',
                            data_file)

####################################################################################################
####################################################################################################

def _clean_files(files):
    """Clean a list of files, removing any hidden files."""

    return list(filter(lambda x: x[0] != '.', files))
