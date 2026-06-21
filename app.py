from flask import Flask, request, jsonify
import numpy as np
import joblib
from PIL import Image
from tensorflow.keras.models import load_model, Model
import os

app = Flask(__name__)

# =========================
# PATH FILE MODEL
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CNN_MODEL_PATH = os.path.join(BASE_DIR, "model_kayu_5kelas_best.h5")
RF_MODEL_PATH = os.path.join(BASE_DIR, "random_forest_umur.pkl")

# =========================
# LOAD MODEL
# =========================
cnn_model = load_model(CNN_MODEL_PATH)
rf_model = joblib.load(RF_MODEL_PATH)

# Feature extractor untuk Random Forest
feature_extractor = Model(
    inputs=cnn_model.input,
    outputs=cnn_model.layers[-2].output
)

# =========================
# LABEL JENIS KAYU
# =========================
jenis_kayu = [
    "Jati",
    "Meranti",
    "Seru",
    "Tembesu",
    "Unglen"
]

# =========================
# MAPPING UMUR BERDASARKAN HASIL RF
# =========================
# 0 = Muda
# 1 = Sedang
# 2 = Tua
umur_mapping = {
    0: {
        "kategori": "Muda",
        "rentang": "10-20 Tahun"
    },
    1: {
        "kategori": "Sedang",
        "rentang": "15-30 Tahun"
    },
    2: {
        "kategori": "Tua",
        "rentang": "35-50 Tahun"
    }
}

# =========================
# HOME / HEALTH CHECK
# =========================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "WoodVision AI Backend is running",
        "status": "success"
    })

# =========================
# PREDICT
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({
                "error": "File gambar tidak ditemukan"
            }), 400

        file = request.files["image"]

        # Buka gambar
        img = Image.open(file).convert("RGB")
        img = img.resize((224, 224))

        # Preprocessing
        img_array = np.array(img).astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # =========================
        # PREDIKSI JENIS KAYU DENGAN CNN
        # =========================
        pred = cnn_model.predict(img_array, verbose=0)
        kelas_idx = int(np.argmax(pred))
        confidence = float(np.max(pred))
        nama_kayu = jenis_kayu[kelas_idx]

        # =========================
        # PREDIKSI UMUR DENGAN RANDOM FOREST
        # =========================
        fitur = feature_extractor.predict(img_array, verbose=0)
        umur_idx = int(rf_model.predict(fitur)[0])

        info_umur = umur_mapping.get(
            umur_idx,
            {"kategori": "-", "rentang": "-"}
        )

        return jsonify({
            "jenis_kayu": nama_kayu,
            "confidence": f"{round(confidence * 100, 2)}%",
            "kategori_umur": info_umur["kategori"],
            "rentang_umur": info_umur["rentang"]
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

# =========================
# RUN LOCAL
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)