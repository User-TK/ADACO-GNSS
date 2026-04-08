import h5py

with h5py.File("gnss_dataset.h5", "r") as hf:
    
    # Get the spectrum for sample 0
    print(hf["spectrum_01"][0])        # 256 numbers
    
    # Get the label for sample 0
    print(hf["label"][0])              # 0 or 1
    
    # Get all labels at once
    print(hf["label"][:])              # all N labels
    
    # Get nav_pvt for samples 0-5
    print(hf["nav_pvt"][0:5])          # 5 rows × 15 columns
    
    # Get all attacked samples' spectrums
    labels = hf["label"][:]
    attacked_idx = (labels == 1)
    print(hf["spectrum_01"][attacked_idx])  # only attacked rows