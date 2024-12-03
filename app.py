from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import base64

app = Flask(__name__)

# Memuat model
model = load_model('model/WaterEye-v0.h5')
model.make_predict_function()

## v0 Warna
dic = {0: 'black', 1: 'blue', 2: 'brown', 3: 'clean', 4: 'green', 5: 'red', 6: 'yellow'}

## v1,v2 Warna + Kejernihan
# dic = {
#     0: 'black-concentrated', 1: 'blue-concentrated', 2: 'brown-concentrated',
#     3: 'black-clear', 4: 'blue-clear', 5: 'brown-clear',
#     6: 'green-clear', 7: 'red-clear', 8: 'transparent-clear',
#     9: 'transparent-concentrated', 10: 'yellow-clear', 11: 'green-concentrated',
#     12: 'not_water', 13: 'red-concentrated', 14: 'yellow-concentrated'
# }

def predict_label(img_path):
    i = image.load_img(img_path, target_size=(224, 224))
    i = image.img_to_array(i) / 255.0
    i = i.reshape(1, 224, 224, 3)
    predictions = model.predict(i)
    color_water = predictions.argmax(axis=1)[0]
    accuracy = predictions[0][color_water]
    return dic[color_water], float(accuracy)

# API untuk prediksi
@app.route("/predict", methods=['POST'])
def predict():
    # Cek apakah data dalam format JSON
    if request.is_json:
        data = request.get_json()  # Mendapatkan data JSON
        if 'image_data' in data:  # Input dari kamera (Base64)
            image_data = data['image_data']
            image_data = image_data.split(',')[1]  # Hapus header Base64
            image_bytes = base64.b64decode(image_data)

            # Simpan sementara untuk prediksi
            img_path = "temp_image.png"
            with open(img_path, "wb") as f:
                f.write(image_bytes)
        else:
            return jsonify({"error": "No image data provided"}), 400
    elif 'file' in request.files:  # Input dari file unggahan
        img = request.files['file']
        img_path = "temp_image.png"
        img.save(img_path)
    else:
        return jsonify({"error": "No image provided"}), 400

    # Prediksi
    try:
        color_water, accuracy = predict_label(img_path)
        os.remove(img_path)  # Hapus file sementara
        return jsonify({"prediction": color_water, "confidence": f"{accuracy:.2%}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
