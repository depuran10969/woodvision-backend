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
# Urutan harus sama dengan output model CNN
# =========================
jenis_kayu = [
    "Jati",
    "Meranti",
    "Seru",
    "Tembesu",
    "Unglen"
]

# =========================
# LABEL KATEGORI UMUR DARI RANDOM FOREST
# 0 = Muda, 1 = Sedang, 2 = Tua
# =========================
kategori_umur_mapping = {
    0: "Muda",
    1: "Sedang",
    2: "Tua"
}

# =========================
# ESTIMASI UMUR FINAL BERDASARKAN JENIS KAYU
# Ini yang akan ditampilkan di aplikasi
# =========================
estimasi_umur_kayu = {
    "Jati": {
        "kategori_default": "Tua",
        "estimasi": "40-80 Tahun"
    },
    "Meranti": {
        "kategori_default": "Sedang",
        "estimasi": "15-30 Tahun"
    },
    "Seru": {
        "kategori_default": "Muda",
        "estimasi": "10-20 Tahun"
    },
    "Tembesu": {
        "kategori_default": "Tua",
        "estimasi": "35-50 Tahun"
    },
    "Unglen": {
        "kategori_default": "Tua",
        "estimasi": "30-50 Tahun"
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

        # =========================
        # PREPROCESS IMAGE
        # =========================
        img = Image.open(file).convert("RGB")
        img = img.resize((224, 224))

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
        # PREDIKSI KATEGORI UMUR DENGAN RANDOM FOREST
        # =========================
        fitur = feature_extractor.predict(img_array, verbose=0)
        umur_idx = int(rf_model.predict(fitur)[0])

        kategori_umur_rf = kategori_umur_mapping.get(umur_idx, "-")

        # =========================
        # ESTIMASI UMUR FINAL BERDASARKAN JENIS KAYU
        # =========================
        info_estimasi = estimasi_umur_kayu.get(
            nama_kayu,
            {
                "kategori_default": kategori_umur_rf,
                "estimasi": "-"
            }
        )

        # Kalau mau kategori umur tetap mengikuti RF:
        kategori_final = kategori_umur_rf

        # Kalau RF error / tidak cocok, fallback ke kategori default kayu
        if kategori_final == "-" or kategori_final == "":
            kategori_final = info_estimasi["kategori_default"]

        return jsonify({
            "jenis_kayu": nama_kayu,
            "confidence": f"{round(confidence * 100, 2)}%",
            "kategori_umur": kategori_final,
            "estimasi_umur": info_estimasi["estimasi"]
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