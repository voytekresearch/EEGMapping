# EEGMapping

Applying FOOOF across EEG datasets, mapping oscillations & 1/f.

## Data

Current datasets:
- Local Datasets, from VoytekLab
    - EEG datasets with rest and task data, young adults. Includes two different visual psychophysics tasks
- MIPDB Dataset, from ChildMind
    - EEG Dataset across developmental ages: childhood -> adult
        - Link: http://fcon_1000.projects.nitrc.org/indi/cmi_eeg/eeg.html

## Notebooks

Within each of 'local' and 'childmind', there are notebookd and associated scripts covering:

- 01-EEGDataProcessing
    - Processing pipeline for the dataset, calculating PSDs, and fitting PSD slope
- 02-EEGDataGroupAnalysis
    - Analysis of the group level results of the dataset.
