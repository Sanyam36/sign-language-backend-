from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import json
import cv2
import base64
import os

app = Flask(__name__)
CORS(app)

# ── Load Model ──────────────────────────────────────
print("🔄 Loading model...")
model = tf.keras.models.load_model("model.h5")

# ── Load Class Labels ────────────────────────────────
with open("class_labels.json", "r") as f:
    class_indices = json.load(f)

# Reverse: {0: 'A', 1: 'B', ...}
labels = {v: k for k, v in class_indices.items()}
print(f"✅ Model loaded! Classes: {list(labels.values())}")

# ── Preprocess Image ─────────────────────────────────
def preprocess(img_array):
    img = cv2.resize(img_array, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ── Routes ───────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Sign Language API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if "image" not in data:
            return jsonify({"error": "No image provided"}), 400

        # Decode base64 image
        img_data = base64.b64decode(data["image"].split(",")[1])
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Invalid image"}), 400

        # Preprocess & Predict
        processed = preprocess(img)
        predictions = model.predict(processed, verbose=0)
        
        predicted_idx = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0])) * 100
        predicted_label = labels[predicted_idx]

        # Top 3 predictions
        top3_idx = np.argsort(predictions[0])[-3:][::-1]
        top3 = [
            {"label": labels[int(i)], "confidence": round(float(predictions[0][i]) * 100, 2)}
            for i in top3_idx
        ]

        return jsonify({
            "prediction": predicted_label,
            "confidence": round(confidence, 2),
            "top3": top3
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("🚀 Starting Sign Language API...")
    print("📡 API running at: http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
