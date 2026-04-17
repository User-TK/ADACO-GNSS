import sys
from pathlib import Path
import h5py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
#meant to make sure project root is on the path so package's for dataset is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datasets.attack_labeler import compute_labels_batch

# replace with your path to the dataset
H5_PATH = r"datasets/gnss_dataset.h5"

# ----- 1. LOAD DATA ------------
print("Loading data...")
with h5py.File(H5_PATH, "r") as hf:
    days  = hf["day"][:]
    hours = hf["hour"][:]
    

    # only use scalar features for RF — no spectra or matrices yet
    X = np.hstack([
        hf["nav_pvt"][:],       # (N, 15)
        hf["nav_clock"][:],     # (N, 4)
        hf["nav_dop"][:],       # (N, 7)
        hf["nav_posecef"][:]    # (N, 4)
    ])  # final shape: (N, 30)

# ------ 2. INFER SECONDS --------------
print("Inferring seconds within each hour...")
N = len(days)
seconds = np.zeros(N, dtype=np.int32)
for hour in range(24):
    mask = (days == b"1221") & (hours == hour)
    idx  = np.where(mask)[0]
    if len(idx) > 0:
        seconds[idx] = np.arange(len(idx))

# ------ 3. APPLY EXACT TIMESTAMPS FROM THE TABLE ------------
# Now uses exact time stamps from Table 13 of Wang et al (2024).
#labels: 0 = clean, 1 = spoofing, 2 = jamming
print("Applying labels based on native H5 data and exact table times...")

# Copy the labels we pulled from the file earlier
labels = compute_labels_batch(days, hours, seconds)



is_spoofing = (labels==1)

is_jamming = (labels==2)

# sanity check for knowing the distribution of each label
print(f"\nFull dataset -- Clean: {(labels==0).sum():,} | Spoofing: {(labels==1).sum():,} | Jamming: {(labels==2).sum():,}")
# ---- 4. SPLIT DATA ---------
CLEAN_TEST_DAYS  = [b"29", b"30"]
CLEAN_TRAIN_DAYS = [b"12", b"13", b"14", b"15", b"16", b"17", b"18",
                b"19", b"20",b"21", b"22", b"23", b"24", b"25", 
                b"26", b"27", b"28"]

clean_train = np.isin(days, CLEAN_TRAIN_DAYS)
clean_test  = np.isin(days, CLEAN_TEST_DAYS)

# Spoofing Split: Train on Hours 12-14, Test on Hours 15-16
spoof_train = is_spoofing & np.isin(hours, [12, 13, 14])
spoof_test  = is_spoofing & np.isin(hours, [15, 16])

# Jamming Split: Train on Hour 17, Test on Hour 16
    # This should be done because hours 16 has less data on jamming compared to hour 17
jam_train = is_jamming & (hours == 17)
jam_test  = is_jamming & (hours == 16)

train_mask = clean_train | spoof_train | jam_train
test_mask  = clean_test  | spoof_test  | jam_test

X_train, y_train = X[train_mask], labels[train_mask]
X_test,  y_test  = X[test_mask],  labels[test_mask]

print(f"\nTrain: {len(X_train):,} samples")
print(f"  Clean: {(y_train==0).sum():,} | Spoofing: {(y_train==1).sum():,} | Jamming: {(y_train==2).sum():,}")
print(f"Test:  {len(X_test):,} samples")
print(f"  Clean: {(y_test==0).sum():,} | Spoofing: {(y_test==1).sum():,} | Jamming: {(y_test==2).sum():,}")

# ---- 5. TRAIN ----
print("\n Training Random Forest...")
clf = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",   # handles the 19:1 imbalance
    n_jobs=-1,                 # uses all CPU cores
    random_state=42
)
clf.fit(X_train, y_train)

# ---- 6. EVALUATE -------
print("Evaluating...")
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, labels=[0, 1, 2], target_names=["Clean", "Spoofing", "Jamming"], zero_division=0))

# Dynamic AUC-ROC handler so it won't crash if a class is missing in your split
if len(clf.classes_) > 2:
    print(f"AUC-ROC (OvR): {roc_auc_score(y_test, y_prob, multi_class='ovr'):.4f}")
elif len(clf.classes_) == 2:
    print(f"AUC-ROC (Binary): {roc_auc_score(y_test, y_prob[:, 1]):.4f}")

print("\n=== Confusion Matrix ===")
print("Classes: Clean=0, Spoofing=1, Jamming=2")
print(confusion_matrix(y_test, y_pred, labels=[0, 1, 2]))

# ---- 7. FEATURE IMPORTANCE -----
feature_names = (
    ["pvt_" + f for f in ["fixType","gnssFixOk","numSV","lat","lon","height",
                           "hMSL","hAcc","vAcc","gSpeed","pDOP","carrSoln",
                           "difSoln","invalidLlh","tAcc"]] +
    ["clock_" + f for f in ["clkB","clkD","tAcc","fAcc"]] +
    ["dop_"   + f for f in ["gDOP","pDOP","tDOP","vDOP","hDOP","nDOP","eDOP"]] +
    ["pos_"   + f for f in ["ecefX","ecefY","ecefZ","pAcc"]]
)

importances = clf.feature_importances_
top10 = np.argsort(importances)[::-1][:10]

print("\n=== Top 10 Most Important Features ===")
for i, idx in enumerate(top10):
    print(f"  {i+1}. {feature_names[idx]:<25} {importances[idx]:.4f}")

# --- 8. SAVE MODEL ----

joblib.dump(clf, "rf_baseline.pkl")
print("\nModel saved to models/rf_baseline.pkl")