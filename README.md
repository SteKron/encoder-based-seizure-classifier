# BrainCodec

BrainCodec is an open source neural compressor for human electrophysiological data (EEG and iEEG).

This is the accompanying codebase for the paper [The Case for Cleaner Biosignals: High-fidelity Neural Compressor Enables Transfer from Cleaner iEEG to Noisier EEG](https://openreview.net/forum?id=b57IG6N20B). We provide the code for the model and a dataloader for BIDS-formatted files, instructions for installation and use are below.

## Requirements

The `requirements.txt` file is provided in the repository. Simply install all requirements with `pip install -r requirements.txt`.

## Use

We provide a sample config file `configs/BrainCodec_eval.yaml`. It leverages the BIDS dataloader to process any BIDS-compliant dataset (e.g., the BIDS version of the CHB-MIT Dataset can be found [here](https://zenodo.org/records/10259996)).

To compute the PRD on the chosen testing dataset and patient run
```
python main.py test --config configs/BrainCodec_eval.yaml --model.init_args.load_model '<checkpoint_path>' --data.init_args.folders ['<dataset_path>'] --data.init_args.test_patients [['<dataset_subject>']]
```

## Citation

```
@inproceedings{
carzaniga2025the,
title={The Case for Cleaner Biosignals: High-fidelity Neural Compressor Enables Transfer from Cleaner i{EEG} to Noisier {EEG}},
author={Francesco S. Carzaniga and Gary Tom Hoppeler and Michael Hersche and Kaspar Schindler and Abbas Rahimi},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=b57IG6N20B}
}
```