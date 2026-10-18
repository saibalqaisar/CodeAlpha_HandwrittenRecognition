"""
CodeAlpha ML Internship - Task 3: Handwritten Character Recognition
CNN trained on MNIST (digits 0-9) or EMNIST Letters (A-Z).

Usage:
    python handwritten_recognition.py                      # MNIST digits
    python handwritten_recognition.py --dataset emnist     # EMNIST letters (needs: pip install tensorflow_datasets)
"""
import argparse
import string
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_data(name):
    if name == "mnist":
        (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.mnist.load_data()
        classes = [str(i) for i in range(10)]
        x_tr, x_te = x_tr[..., None], x_te[..., None]
    else:
        import tensorflow_datasets as tfds
        tr, te = tfds.as_numpy(tfds.load("emnist/letters", split=["train", "test"],
                                         batch_size=-1, as_supervised=True))
        x_tr, y_tr = tr
        x_te, y_te = te
        # EMNIST images are stored transposed; labels are 1..26
        x_tr, x_te = x_tr.transpose(0, 2, 1, 3), x_te.transpose(0, 2, 1, 3)
        y_tr, y_te = y_tr - 1, y_te - 1
        classes = list(string.ascii_uppercase)
    return (x_tr.astype("float32") / 255.0, y_tr), (x_te.astype("float32") / 255.0, y_te), classes


def build_model(n_classes):
    return models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        # augmentation (active only during training) -> better on real handwriting
        layers.RandomRotation(0.06),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomZoom(0.1),
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(n_classes, activation="softmax"),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mnist", "emnist"], default="mnist")
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()

    (x_train, y_train), (x_test, y_test), classes = load_data(args.dataset)
    print("Train:", x_train.shape, "Test:", x_test.shape, "Classes:", len(classes))

    model = build_model(len(classes))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    cbs = [callbacks.EarlyStopping(patience=4, restore_best_weights=True),
           callbacks.ReduceLROnPlateau(patience=2, factor=0.5)]
    model.fit(x_train, y_train, validation_split=0.1, epochs=args.epochs,
              batch_size=128, callbacks=cbs)

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy: {acc:.4f}")

    pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    print(classification_report(y_test, pred, target_names=classes))

    # Confusion matrix image
    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xticks(range(len(classes)), classes, fontsize=7)
    plt.yticks(range(len(classes)), classes, fontsize=7)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.colorbar()
    plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=120); plt.close()

    # Sample predictions image
    idx = np.random.choice(len(x_test), 15, replace=False)
    plt.figure(figsize=(10, 4))
    for i, k in enumerate(idx):
        plt.subplot(3, 5, i + 1)
        plt.imshow(x_test[k].squeeze(), cmap="gray"); plt.axis("off")
        ok = pred[k] == y_test[k]
        plt.title(f"{classes[pred[k]]} ({classes[y_test[k]]})", color="green" if ok else "red", fontsize=9)
    plt.suptitle("Predicted (True)")
    plt.tight_layout(); plt.savefig("sample_predictions.png", dpi=120); plt.close()

    model.save("handwritten_model.keras")
    np.save("classes.npy", np.array(classes))
    print("Saved: handwritten_model.keras, classes.npy, confusion_matrix.png, sample_predictions.png")


if __name__ == "__main__":
    main()
