import h5py
import numpy as np

target_day = "1221"
with h5py.File("gnss_dataset.h5", "r") as hf:
    
    days = hf["day"][:]
    labels = hf["label"][:] # loads full array into memory

    mask = days == target_day.encode()
    day_labels = labels[mask]
    pvt_first_hour = hf["nav_pvt"][0:3600]
    print (f"Thingy {pvt_first_hour}")
    print(f"Day {target_day}:")
    print(f" Total samples : {mask.sum():,}")
    print(f"Raw labels: {day_labels.dtype}")
    print(f"  Unique values: {np.unique(day_labels)}")
    # unique, counts = np.unique(label, return_counts=True)
    # for d, c in zip(unique, counts):
    #     print(f"Day {d}: {c:,} samples")\
