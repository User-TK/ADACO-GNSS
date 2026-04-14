import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets.attack_labeler import compute_attack_type_from_time

class GNSSDataset(Dataset):
    def __init__(self, h5_path, indices=None, multiclass=True):
        self.h5_path    = h5_path
        self.h5         = None
        self.multiclass = multiclass

        with h5py.File(h5_path, "r") as hf:
            days  = hf["day"][:]
            hours = hf["hour"][:]
            N     = hf["label"].shape[0]

            # precompute second_within_hour for all samples
            self.seconds = np.zeros(N, dtype=np.int32)
            for day in [b"1221"]:
                for hour in range(24):
                    mask    = (days == day) & (hours == hour)
                    indices_block = np.where(mask)[0]
                    if len(indices_block) > 0:
                        self.seconds[indices_block] = np.arange(len(indices_block))

            self.days  = days
            self.hours = hours

        self.indices = indices if indices is not None else np.arange(N)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        if self.h5 is None:
            self.h5 = h5py.File(self.h5_path, "r")

        idx = self.indices[i]

        if self.multiclass:
            label = compute_attack_type_from_time(
                self.days[idx],
                int(self.hours[idx]),
                int(self.seconds[idx])
            )
        else:
            label = int(self.h5["label"][idx])

        spectrum_01 = self.h5["spectrum_01"][idx]
        spectrum_02 = self.h5["spectrum_02"][idx]

        return {
            "spectrum" : torch.tensor(np.stack([spectrum_01, spectrum_02]),
                                      dtype=torch.float32),
            "pvt"      : torch.tensor(self.h5["nav_pvt"][idx],   dtype=torch.float32),
            "clock"    : torch.tensor(self.h5["nav_clock"][idx], dtype=torch.float32),
            "dop"      : torch.tensor(self.h5["nav_dop"][idx],   dtype=torch.float32),
            "sat"      : torch.tensor(self.h5["nav_sat"][idx],   dtype=torch.float32),
            "rawx"     : torch.tensor(self.h5["rxm_rawx"][idx],  dtype=torch.float32),
            "label"    : torch.tensor(label, dtype=torch.long)
        }