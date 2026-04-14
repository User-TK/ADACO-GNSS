import h5py

with h5py.File("gnss_dataset.h5", "r") as hf:
  for key in hf.keys():
    print(f"{key:15} shape={hf[key].shape}")
