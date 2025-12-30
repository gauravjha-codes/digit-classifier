
# Handwritten Digit Classification (0–9)

This project implements a **Convolutional Neural Network (CNN)** to classify handwritten digits (0–9) using a **combined MNIST and EMNIST Digits dataset**. The model is built with **TensorFlow/Keras** and designed for robust generalization using data augmentation and early stopping.

---

## 📌 Project Overview

- Task: Handwritten digit classification (0–9)
- Model Type: Convolutional Neural Network (CNN)
- Framework: TensorFlow / Keras
- Dataset: MNIST + EMNIST (Digits)
- Output: Trained model saved as `mnist_digit_model.h5`

The MNIST and EMNIST datasets are combined to increase dataset diversity and improve model robustness.

---

## 🧠 Model Architecture

- Input: 28 × 28 grayscale images
- Data Augmentation:
  - Random rotation
  - Random translation
  - Random zoom
- Convolutional Layers:
  - Conv2D (32 filters) + MaxPooling
  - Conv2D (64 filters) + MaxPooling
  - Conv2D (128 filters)
- Fully Connected Layers:
  - Dense (128 units, ReLU)
  - Dropout (0.3)
  - Dense (10 units, Softmax)

---

## ⚙️ Training Details

- Optimizer: Adam (learning rate = 1e-3)
- Loss Function: Sparse Categorical Crossentropy
- Batch Size: 128
- Epochs: 15 (with Early Stopping)
- Evaluation Metric: Accuracy
- Early Stopping:
  - Monitors validation loss
  - Restores best weights to prevent overfitting

---

## 📊 Dataset

- **MNIST**: Standard handwritten digit dataset
- **EMNIST Digits**: Extended handwritten digit dataset
- Total samples after combination: ~340,000 images

All images are normalized to the range [0, 1] before training.

---

## 🚀 How to Run

### 1️⃣ Install Dependencies
```bash
pip install tensorflow tensorflow-datasets numpy
```

### 2️⃣ Train the Model
```bash
python train_model.py
```

### 3️⃣ Output
- Trained model will be saved as:
  ```
  mnist_digit_model.h5
  ```

---

## 📁 Project Structure

```
.
├── train_model.py
├── mnist_digit_model.h5
└── README.md
```

---

## 🎯 Key Learnings

- Implemented CNNs for image classification
- Used data augmentation to improve model robustness
- Built efficient tf.data pipelines
- Combined multiple datasets for better generalization
- Applied early stopping to avoid overfitting

---

## 📌 Future Improvements

- Hyperparameter tuning
- Model quantization for deployment
- Web or mobile inference interface
- Confusion matrix and class-wise evaluation

---

## 📄 License

This project is for educational and learning purposes.
