import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os

"""
Train a CNN using COMBINED MNIST + EMNIST Digits dataset.
Total dataset size becomes ~340k digit samples.

Output model: mnist_digit_model.h5
"""

BATCH_SIZE = 128
EPOCHS = 15 # Reduced slightly as 340k images converge quickly

# -------------------------
# PREPROCESSING
# -------------------------

def preprocess(image, label):
    """
    Convert to float32 [0,1].
    Assumes input image is already (28, 28, 1).
    """
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# -------------------------
# MODEL
# -------------------------

def build_model():
    # Augmentation to help generalization
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.10),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="augmentation"
    )

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28, 1)),
            data_augmentation,
            
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding='same'),
            tf.keras.layers.MaxPooling2D(2),
            
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding='same'),
            tf.keras.layers.MaxPooling2D(2),
            
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
            
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(10, activation="softmax")
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

# -------------------------
# MAIN TRAINING PIPELINE
# -------------------------

def main():
    print("Loading datasets...")

    # -----------------------------------------------------
    # 1️⃣ Load MNIST (Standard Keras)
    # -----------------------------------------------------
    # Correctly unpack the tuples
    (m_train_img, m_train_lbl), (m_test_img, m_test_lbl) = tf.keras.datasets.mnist.load_data()
    
    # Expand dims to match (N, 28, 28, 1) for consistency
    m_train_img = np.expand_dims(m_train_img, axis=-1)
    m_test_img = np.expand_dims(m_test_img, axis=-1)
    
    print(f"MNIST Train shape: {m_train_img.shape}")

    # -----------------------------------------------------
    # 2️⃣ Load EMNIST Digits (TensorFlow Datasets)
    # -----------------------------------------------------
    # Note: EMNIST from TFDS is typically (28, 28, 1)
    ds_train, ds_test = tfds.load(
        "emnist/digits",
        split=["train", "test"],
        shuffle_files=True,
        as_supervised=True,
        batch_size=-1 # Load full dataset into memory as NumPy
    )

    e_train_img, e_train_lbl = tfds.as_numpy(ds_train)
    e_test_img, e_test_lbl = tfds.as_numpy(ds_test)

    print(f"EMNIST Train shape: {e_train_img.shape}")

    # -----------------------------------------------------
    # 3️⃣ Combine MNIST + EMNIST
    # -----------------------------------------------------
    # Concatenate along the first axis (number of samples)
    X_train = np.concatenate([m_train_img, e_train_img], axis=0)
    y_train = np.concatenate([m_train_lbl, e_train_lbl], axis=0)

    X_test = np.concatenate([m_test_img, e_test_img], axis=0)
    y_test = np.concatenate([m_test_lbl, e_test_lbl], axis=0)

    print("-" * 30)
    print(f"COMBINED Train: {X_train.shape}")
    print(f"COMBINED Test:  {X_test.shape}")
    print("-" * 30)

    # -----------------------------------------------------
    # 4️⃣ Create tf.data.Dataset
    # -----------------------------------------------------
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_ds  = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    train_ds = (
        train_ds
        .shuffle(100_000)
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_ds = (
        test_ds
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    # -----------------------------------------------------
    # 5️⃣ Train Model
    # -----------------------------------------------------
    model = build_model()
    model.summary()

    # Early stopping to prevent over-training since dataset is large
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=3, 
        restore_best_weights=True
    )

    print("\nStarting training...")
    model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=test_ds,
        callbacks=[early_stop]
    )

    # -----------------------------------------------------
    # 6️⃣ Evaluate & Save
    # -----------------------------------------------------
    loss, acc = model.evaluate(test_ds, verbose=0)
    print(f"\nFinal Combined Dataset Accuracy: {acc:.4f}")

    model.save("mnist_digit_model.h5")
    print("Model saved as 'mnist_digit_model.h5'")

if __name__ == "__main__":
    main()