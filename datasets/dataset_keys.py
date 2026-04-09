import h5py
import numpy as np

with h5py.File("gnss_dataset.h5", "r") as hf:
    print(list(hf.keys()))