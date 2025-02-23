from math import ceil
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from ancpbids.pybids_compat import BIDSLayout
from pyedflib import EdfReader
from torch import Tensor
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader, Dataset

from datasets import EEGBatch, EEGDataset

_SAMPLING_RATE = 512


def read_edf(
    edf_file: str,
) -> np.ndarray:
    with EdfReader(edf_file) as f:
        n_chrs = f.signals_in_file

        ch_nrs = range(n_chrs)

        ch_nrs = [n_chrs + ch if ch < 0 else ch for ch in ch_nrs]
        nsamples = f.getNSamples()

        assert len(set(nsamples)) == 1, ValueError(
            "Not all signals have the same length."
        )

        signals = np.empty((n_chrs, nsamples[0]), dtype=np.float32)
        for i, c in enumerate(ch_nrs):
            signal = f.readSignal(c)
            signals[i, :] = signal

    return signals


class BIDSEEGData(EEGDataset):
    def __init__(
        self,
        folders: List[Union[str, Path]],
        batch_size: Optional[int] = 32,
        train_patients: List[Optional[List[Optional[int | str]]]] = [None],
        val_patients: List[Optional[List[Optional[int | str]]]] = [[""]],
        test_patients: List[Optional[List[Optional[int | str]]]] = [[""]],
        segment_size: Optional[int] = 10000,
        stride: Optional[int] = None,
        num_workers: Optional[int] = 0,
        limit_train_batches: Optional[int | float] = None,
    ) -> None:
        super().__init__()
        self.folders = folders
        self.batch_size = batch_size
        self._get_dataset_info()
        self.num_workers = num_workers
        self.limit_train_batches = limit_train_batches
        self.train_patients = self._sanitize_patients(train_patients)
        self.val_patients = self._sanitize_patients(val_patients)
        self.test_patients = self._sanitize_patients(test_patients)
        self.segment_size = segment_size
        self.segment_samples = int(self.segment_size / 1000.0 * _SAMPLING_RATE)
        self.stride = stride if stride is not None else segment_size
        self.window_size = segment_size

    def _get_dataset_info(self) -> None:
        layouts = []
        for folder in self.folders:
            lay = BIDSLayout(folder)
            layouts.append(lay)

        self.layouts = layouts

    def _sanitize_patients(self, subset):
        patients_sane = []
        if len(subset) != len(self.layouts):
            return [lay.get_subjects() for lay in self.layouts]
        for idx, lay in enumerate(self.layouts):
            dataset_subset = subset[idx]
            if dataset_subset == None:
                patients_sane.append(lay.get_subjects())
            elif len(dataset_subset) == 1 and dataset_subset[0] == "":
                patients_sane.append([""])
            else:
                sub_pat = [str(p).zfill(2) for p in dataset_subset]
                patients_sane.append(list(set(lay.get_subjects()) & set(sub_pat)))
        return patients_sane

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.dataset_train = BIDSEEGDataset(
                layouts=self.layouts,
                n_patient=self.train_patients,
                window=self.window_size,
                stride=self.stride,
            )
            if self.val_patients:
                self.dataset_val = BIDSEEGDataset(
                    layouts=self.layouts,
                    n_patient=self.val_patients,
                    window=self.window_size,
                    stride=self.stride,
                )
        if stage == "validate":
            self.dataset_val = BIDSEEGDataset(
                layouts=self.layouts,
                n_patient=self.val_patients,
                window=self.window_size,
                stride=self.stride,
            )
        if stage == "test" or stage == "predict":
            self.dataset_test = []
            dataset_test = BIDSEEGDataset(
                layouts=self.layouts,
                n_patient=self.test_patients,
                window=self.window_size,
                stride=self.stride,
            )
            self.dataset_test.append(dataset_test)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_patients:
            return DataLoader(
                self.dataset_val,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
            )
        else:
            return None

    def test_dataloader(self) -> DataLoader:
        test_dataloader = []
        for dataset in self.dataset_test:
            test_dataloader.append(
                DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    pin_memory=True,
                )
            )
        return test_dataloader

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()


class BIDSEEGDataset(Dataset[EEGBatch]):
    def __init__(
        self,
        layouts: List[Union[str, Path]],
        n_patient: List[List[str]],
        sampling_rate: int = _SAMPLING_RATE,
        window: int = 80000,
        stride: Optional[int] = None,
    ) -> None:
        self.window = window / 1000.0
        self.srate = sampling_rate
        self.stride = stride / 1000.0
        self.n_patient = n_patient
        self.window_samples = int(self.window * self.srate)
        self.stride_samples = int(self.stride * self.srate)

        self.layouts = layouts

        self.map_items()

        self._patient_files = []

        self._datasets = [{} for _ in range(len(self.layouts))]
        self._sampling_rates = [{} for _ in range(len(self.layouts))]

    @property
    def datasets(self):
        for idx, dataset in enumerate(self._datasets):
            if len(dataset) == 0:
                patient_files = self.layouts[idx].get(
                    subject=self.n_patient[idx], extension="edf", return_type="files"
                )
                for pf in patient_files:
                    dataset[Path(pf).name] = read_edf(pf)
                
                # Extract seizure events from tsv files
                patient_files_evt = self.layouts[idx].get(
                    subject=self.n_patient[idx], suffix="events", extension="tsv", return_type="files"
                )
                for ef in patient_files_evt:
                    dataset[Path(ef).name] = self.seizure_mask(pd.read_csv(ef, sep="\t"))

        return self._datasets

    @property
    def sampling_rates(self):
        for idx, sampling_rate in enumerate(self._sampling_rates):
            if len(sampling_rate) == 0:
                patient_files = self.layouts[idx].get(
                    subject=self.n_patient[idx], extension="edf"
                )
                for pf in patient_files:
                    sampling_rate[pf.name] = pf.get_metadata()["SamplingFrequency"]

        return self._sampling_rates

    def __getitem__(self, n: int) -> Tuple[Tensor, Tensor, Tensor, int, int, float, float, int]:
        dataset = self.dataset_id[n]
        patient = self.patient_id[n]
        session = self.session_id[n]
        run = self.run_id[n]
        mapped = self.map_id[n]

        edf_filename = f"sub-{str(patient).zfill(2)}_ses-{str(session).zfill(2)}_task-szMonitoring_run-{str(run).zfill(2)}_eeg.edf"
        tsv_filename = f"sub-{str(patient).zfill(2)}_ses-{str(session).zfill(2)}_task-szMonitoring_run-{str(run).zfill(2)}_events.tsv"
  
        srate = self.sampling_rates[dataset][edf_filename]

        patient_dataset = self.datasets[dataset][edf_filename]
        patient_seizures = self.datasets[dataset][tsv_filename]

        sample = patient_dataset[
            :,
            int(mapped * self.stride * srate) : int(
                mapped * self.stride * srate + self.window * srate
            ),
        ]

        # Use _SAMPLING_RATE, so there is no need to upsample the mask
        mask = patient_seizures[
            int(mapped * self.stride * _SAMPLING_RATE) : int(
                mapped * self.stride * _SAMPLING_RATE + self.window * _SAMPLING_RATE
            )
        ]

        # TODO: Turn mask at 0.25Hz into mask at 1Hz; 
        # if one window is masked for more than 50%, mask the whole window
        if np.sum(mask) > (0.5 * self.window * _SAMPLING_RATE):
            mask = np.array([1])
        else:
            mask = np.array([0])
        torch.from_numpy(mask)

        sample_torch = torch.from_numpy(sample)
        downsampled_seizure = interpolate(
            sample_torch.unsqueeze(0), size=self.window_samples, mode="linear"
        )
        sample = downsampled_seizure.squeeze(0)

        return EEGBatch(sample, mask, n, mapped, patient, dataset)

    def map_items(self) -> None:
        dataset_id = [np.empty((0,), dtype=int)]
        patient_id = [np.empty((0,), dtype=int)]
        session_id = [np.empty((0,), dtype=int)]
        run_id = [np.empty((0,), dtype=int)]
        map_id = [np.empty((0,), dtype=int)]
        for idx, lay in enumerate(self.layouts):
            patient_files = lay.get(subject=self.n_patient[idx], extension="edf")
            for pf in patient_files:
                dur = pf.get_metadata()["RecordingDuration"]
                pat = pf.get_entities()
                if dur < self.window:
                    continue
                windows_dur = ceil((dur - self.window) / self.stride)
                dataset_id.append(np.zeros((windows_dur), dtype=int) + idx)
                patient_id.append(np.zeros((windows_dur), dtype=int) + int(pat["sub"]))
                session_id.append(np.zeros((windows_dur), dtype=int) + int(pat["ses"]))
                run_id.append(np.zeros((windows_dur), dtype=int) + int(pat["run"]))
                map_id.append(np.arange(windows_dur, dtype=int))

        self.dataset_id = np.concatenate(dataset_id)
        self.patient_id = np.concatenate(patient_id)
        self.session_id = np.concatenate(session_id)
        self.run_id = np.concatenate(run_id)
        self.map_id = np.concatenate(map_id)

    def seizure_mask(self, events) -> np.ndarray:
        # Use _SAMPLING_RATE, so there is no need to upsample the mask
        # This would of course need to be changed when working with multiple datasets

        mask = np.zeros(int(events["recordingDuration"].values[0]) * _SAMPLING_RATE)

        if 'sz' in events['eventType'].values:
            seizure_events = events[events['eventType'] == 'sz'] 
            for idx, seizure in seizure_events.iterrows(): 
                mask[
                    int(seizure["onset"] * _SAMPLING_RATE) : int((seizure["onset"] + seizure["duration"]) * _SAMPLING_RATE)
                ] = 1
            assert np.sum(mask) == seizure_events["duration"].sum() * _SAMPLING_RATE
            
        return mask

    def __len__(self) -> int:
        total_len = len(self.dataset_id)
        return total_len
