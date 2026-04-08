import h5py
import numpy as np

with h5py.File("gnss_dataset.h5", "r") as hf:
    days = hf["day"][:]
    unique, counts = np.unique(days, return_counts=True)
    for d, c in zip(unique, counts):
        print(f"Day {d}: {c:,} samples")