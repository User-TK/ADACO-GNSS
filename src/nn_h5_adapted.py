#used ChatGPT to try to bug fix, maybe wrong idk
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# replace with your path to the dataset
H5_PATH = r"datasets/gnss_dataset.h5"


# ---------- activation helpers ----------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def function_derivative(z):
    fd = sigmoid(z)
    return fd * (1 - fd)


def softmax(z):
    # subtract max per row for numerical stability
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# ---------- label / metric helpers ----------
def one_hot_encode(labels, num_classes):
    encoded = np.zeros((len(labels), num_classes))
    encoded[np.arange(len(labels)), labels.astype(int)] = 1
    return encoded


def accuracy(t, y_pred):
    return np.sum(t == y_pred) / len(t)


class NN:
    def __init__(self, features, hidden_neurons, output_neurons, learning_rate):
        self.features = features
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons
        self.learning_rate = learning_rate

        # initialize input -> hidden weights
        self.V = np.random.randn(self.features, self.hidden_neurons)

        # initialize hidden -> output weights
        self.W = np.random.randn(self.hidden_neurons, self.output_neurons)

        # initialize biases to 0
        self.V0 = np.zeros((self.hidden_neurons))
        self.W0 = np.zeros((self.output_neurons))

    def train(self, X, t, epochs=1000):
        costs = []
        sample_count = X.shape[0]

        for epoch in range(epochs):
            # ----- forward pass -----
            net_u = X.dot(self.V) + self.V0
            H = sigmoid(net_u)

            net_z = H.dot(self.W) + self.W0
            O = softmax(net_z)

            # ----- backpropagation pass -----
            # for softmax + cross-entropy, output delta simplifies to (O - t)
            error_output = O - t

            d_W = H.T.dot(error_output) / sample_count
            d_W0 = np.sum(error_output, axis=0) / sample_count

            error_hidden_layer = error_output.dot(self.W.T) * function_derivative(net_u)
            d_V = X.T.dot(error_hidden_layer) / sample_count
            d_V0 = np.sum(error_hidden_layer, axis=0) / sample_count

            # ----- update weights and biases -----
            self.W -= self.learning_rate * d_W
            self.W0 -= self.learning_rate * d_W0
            self.V -= self.learning_rate * d_V
            self.V0 -= self.learning_rate * d_V0

            # ----- track the cost function every 10 epochs -----
            if epoch % 10 == 0:
                loss = -np.mean(np.sum(t * np.log(O + 1e-12), axis=1))
                costs.append(loss)

        return costs

    def predict(self, X):
        net_u = X.dot(self.V) + self.V0
        H = sigmoid(net_u)

        net_z = H.dot(self.W) + self.W0
        O = softmax(net_z)

        return np.argmax(O, axis=1)


if __name__ == "__main__":
    # ----- 1. LOAD DATA FROM H5 -----
    print("Loading data...")
    with h5py.File(H5_PATH, "r") as hf:
        # every dataset shares the same row index, so row i matches across all arrays
        days = hf["day"][:]
        hours = hf["hour"][:]
        native_labels = hf["label"][:]

        # use the same scalar feature layout as the random-forest baseline
        X = np.hstack([
            hf["nav_pvt"][:],       # (N, 15)
            hf["nav_clock"][:],     # (N, 4)
            hf["nav_dop"][:],       # (N, 7)
            hf["nav_posecef"][:]    # (N, 4)
        ])  # final shape: (N, 30)

    # ----- 2. INFER SECONDS WITHIN THE ATTACK DAY -----
    print("Inferring seconds...")
    N = len(days)
    seconds = np.zeros(N, dtype=np.int32)

    for hour in range(24):
        mask = (days == b"1221") & (hours == hour)
        idx = np.where(mask)[0]
        if len(idx) > 0:
            seconds[idx] = np.arange(len(idx))

    # ----- 3. BUILD CLEAN / SPOOFING / JAMMING LABELS -----
    print("Applying labels based on native H5 data and exact table times...")

    # start from the original H5 label array
    labels = np.copy(native_labels)

    # spoofing: 12:00:00 through 16:50:00 on day 1221
    is_spoofing = (native_labels == 1) & (days == b"1221") & (
        ((hours >= 12) & (hours < 16)) |
        ((hours == 16) & (seconds <= 3000))
    )

    # jamming: 16:51:00 through 17:21:00 on day 1221
    is_jamming = (native_labels == 1) & (days == b"1221") & (
        ((hours == 16) & (seconds >= 3060)) |
        ((hours == 17) & (seconds <= 1260))
    )

    # overwrite the attack labels with explicit class ids
    labels[is_spoofing] = 1   # Spoofing
    labels[is_jamming] = 2    # Jamming

    print(
        f"\nFull dataset — Clean: {(labels == 0).sum():,} | "
        f"Spoofing: {(labels == 1).sum():,} | Jamming: {(labels == 2).sum():,}"
    )

    # ----- 4. SPLIT DATA -----
    # clean days are split by day to avoid leakage
    CLEAN_TEST_DAYS = [b"29", b"30"]
    CLEAN_TRAIN_DAYS = [
        b"12", b"13", b"14", b"15", b"16", b"17", b"18",
        b"19", b"20", b"21", b"22", b"23", b"24", b"25",
        b"26", b"27", b"28"
    ]

    clean_train = np.isin(days, CLEAN_TRAIN_DAYS)
    clean_test = np.isin(days, CLEAN_TEST_DAYS)

    # spoofing split: train on earlier spoofing hours, test on later spoofing hours
    spoof_train = is_spoofing & np.isin(hours, [12, 13, 14])
    spoof_test = is_spoofing & np.isin(hours, [15, 16])

    # jamming split: train on hour 17, test on hour 16
    jam_train = is_jamming & (hours == 17)
    jam_test = is_jamming & (hours == 16)

    train_mask = clean_train | spoof_train | jam_train
    test_mask = clean_test | spoof_test | jam_test

    X_train = X[train_mask]
    y_train = labels[train_mask]

    X_test = X[test_mask]
    y_test = labels[test_mask]

    print(f"Train: {len(X_train):,} samples")
    print(f"Test:  {len(X_test):,} samples")
    print(
        f"Train — Clean: {(y_train == 0).sum():,} | "
        f"Spoofing: {(y_train == 1).sum():,} | Jamming: {(y_train == 2).sum():,}"
    )
    print(
        f"Test  — Clean: {(y_test == 0).sum():,} | "
        f"Spoofing: {(y_test == 1).sum():,} | Jamming: {(y_test == 2).sum():,}"
    )

    # ----- 5. PREP LABELS FOR THE NEURAL NETWORK -----
    # convert class ids into one-hot rows for the 3-output network
    t = one_hot_encode(y_train, num_classes=3)
    t_test = one_hot_encode(y_test, num_classes=3)

    # ----- 6. CREATE THE NEURAL NETWORK -----
    nn = NN(
        features=X_train.shape[1],   # 30 scalar H5 features
        hidden_neurons=20,
        output_neurons=3,            # Clean / Spoofing / Jamming
        learning_rate=0.01
    )

    # ----- 7. TRAIN THE NETWORK -----
    cost = nn.train(X_train, t, epochs=1000)

    # ----- 8. MAKE PREDICTIONS ON THE TEST DATA -----
    y_pred = nn.predict(X_test)

    # ----- 9. EVALUATE THE ACCURACY -----
    acc = accuracy(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}")

    print("\n=== Classification Report ===")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1, 2],
            target_names=["Clean", "Spoofing", "Jamming"],
            zero_division=0
        )
    )

    print("=== Confusion Matrix ===")
    print("Classes: Clean=0, Spoofing=1, Jamming=2")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1, 2]))

    # ----- 10. PLOT TRAINING LOSS -----
    plt.plot(cost)
    plt.xlabel("Logged step (every 10 epochs)")
    plt.ylabel("Cross-entropy loss")
    plt.title("Neural Network Training Loss")
    plt.show()
