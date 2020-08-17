# EEGMapping

Project repository as part of the `Mapping` project, exploring periodic and aperiodic activity across this the cortex.

This repository covers the EEG data analyses. A companion repository has the
[MEG analyses](https://github.com/voytekresearch/MEGMapping).

## Overview

ToDo!

## Reference

A preprint for this project is upcoming!

This project was presented at SfN 2018, the poster for which is available
[here](https://www.dropbox.com/s/alwwb6ahb1wjank/MdandaEtal-SfN2018.pdf?dl=0).

## Repository Layout

This repository contains analyses for two distinct datasets.

These datasets are labelled `local` and `ChildMind`, each with their own folder.

Within each folder, is the following organization:

- `code/` contains custom code for the project
- `notebooks/` contains Jupyter notebooks that step through analyses and creae figures
- `scripts/` contains stand alone scripts that run parts of the analyses

## Requirements

This project was written in Python 3 and requires Python >= 3.7 to run.

If you want to re-run this project, you will need some external dependencies.

Dependencies include 3rd party scientific Python packages:
- [numpy](https://github.com/numpy/numpy)
- [scipy](https://github.com/scipy/scipy)
- [matplotlib](https://github.com/matplotlib/matplotlib)

In addition, this project requires the following dependencies:

 - [mne](https://github.com/mne-tools/mne-python) >= 0.18.2
 - [fooof](https://github.com/fooof-tools/fooof) >= 1.0.0
 - [autoreject](https://github.com/autoreject/autoreject)

## Data

This repository analyses two distinct datasets of EEG data:

- `local` is a dataset of EEG dataset collected in the VoytekLab, at UC San Diego
    - Around 30 subjects, young adults
    - Main task is a visual psychophysics task
    - Includes resting state data
- `ChildMind` is the MIPDB Dataset, from the ChildMind institute
    - EEG Dataset across developmental ages: childhood -> adult
        - Link: http://fcon_1000.projects.nitrc.org/indi/cmi_eeg/eeg.html
