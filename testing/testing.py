import h5py
import numpy as np

with h5py.File("datasets/gnss_dataset.h5", "r") as f:
    # Load label dataset
    labels = f["label"][:]   # adjust key if needed

# Unique values
unique_labels = np.unique(labels)
print("Unique labels:", unique_labels)

# Count per label
print("\nCounts per label:")
for val in unique_labels:
    count = np.sum(labels == val)
    print(f"Label {val}: {count}")

#Unique labels: [0 1]
#Counts per label:
#Label 0: 1641600
#Label 1: 43200
