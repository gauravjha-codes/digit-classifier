from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf

"""
Simple full-stack app to classify handwritten digit images (0–9).

Steps to run:
1. Make sure `mnist_digit_model.h5` exists (run train_model.py once).
2. Install dependencies:
       pip install flask tensorflow pillow
3. Start the server:
       python app.py
4. Open in browser:
       http://127.0.0.1:5000
"""

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "mnist_digit_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Convert image to MNIST-like 28x28:
    - grayscale
    - invert if needed (white digit on black background)
    - binarize
    - crop tight around the digit
    - resize digit to ~20x20
    - pad to 28x28 and center

    Returns array of shape (1, 28, 28, 1) with float32 in [0, 1].
    """
    # 1. Convert to grayscale
    img = image.convert("L")

    # 2. Resize to a larger square first for consistency
    img = img.resize((128, 128))

    arr = np.array(img).astype("float32") / 255.0

    # 3. If background is bright, invert so digit is white on black
    if arr.mean() > 0.5:
        arr = 1.0 - arr

    # 4. Binarize (threshold)
    thresh = 0.2
    bin_img = (arr > thresh).astype("float32")

    # 5. Find bounding box of the digit (non-zero area)
    coords = np.argwhere(bin_img > 0)
    if coords.size == 0:
        # No strokes detected -> return blank image
        digit_28 = np.zeros((28, 28), dtype="float32")
    else:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1  # end index is exclusive
        digit = bin_img[y0:y1, x0:x1]

        # 6. Resize digit to 20x20 while keeping aspect ratio
        digit_img = Image.fromarray((digit * 255).astype("uint8"))

        # Pillow compatibility for different versions
        if hasattr(Image, "Resampling"):
            resample_mode = Image.Resampling.LANCZOS
        else:
            resample_mode = Image.LANCZOS

        digit_img = digit_img.resize((20, 20), resample=resample_mode)

        # 7. Put 20x20 digit into 28x28 canvas (centered)
        digit_28 = np.zeros((28, 28), dtype="float32")
        y_offset = (28 - 20) // 2
        x_offset = (28 - 20) // 2
        digit_28[y_offset:y_offset + 20, x_offset:x_offset + 20] = (
            np.array(digit_img).astype("float32") / 255.0
        )

    # 8. Final shape: (1, 28, 28, 1)
    digit_28 = digit_28.reshape(1, 28, 28, 1)
    return digit_28


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        image = Image.open(file.stream)
    except Exception as e:
        return jsonify({"error": f"Cannot open image: {e}"}), 400

    # Preprocess and predict
    arr = preprocess_image(image)
    probs = model.predict(arr)[0]
    digit = int(np.argmax(probs))
    confidence = float(np.max(probs))

    response = {
        "digit": digit,
        "confidence": confidence,
    }

    # Optional: flag low-confidence predictions
    if confidence < 0.6:
        response["warning"] = "Model is not very confident about this prediction."

    return jsonify(response)


if __name__ == "__main__":
    # Debug mode for development
    app.run(debug=True)
