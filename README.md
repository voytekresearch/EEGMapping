# EEGMapping

Applying FOOOF across EEG datasets, mapping oscillations & 1/f. 

## Data

Current datasets:
- MIPDB Dataset, from ChildMind
    - EEG Dataset across developmental ages: childhood -> adult
        - Link: http://fcon_1000.projects.nitrc.org/indi/cmi_eeg/eeg.html
- rtPB & PBA Local Data (analysis by Luyanda, code in his repo)

## Notebooks

- 01-EEGDataProcessing
    - Processing pipeline for the MIPDB EEG dataset, calculating PSDs, and fitting PSD slope
- 02-EEGDataGroupAnalysis
    - Analysis of the group level results of the MIPDB dataset.
- EEGChi_Fitting
    - Sanity checking FOOOF fitting on other EEG data, from the Chicago group (OLD)

Notes:
- This project used to be combined in the synthetic testing the PSDSlopeFitting repository.
