from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import base64
import io
import logging
from werkzeug.utils import secure_filename

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_path = os.getenv('MODEL_PATH', 'model/WaterEye-v3.h5')
model = load_model(model_path)
model.make_predict_function()

# LABELS / CLASS
dic = {0: 'black', 1: 'blue', 2: 'brown', 3: 'clear', 4: 'green', 5: 'not-water', 6: 'red', 7: 'yellow'}

def predict_label(img_bytes):
    # Load image, resize, and normalize
    img = image.load_img(io.BytesIO(img_bytes), target_size=(224, 224))
    img = image.img_to_array(img) / 255.0
    img = img.reshape(1, 224, 224, 3)

    predictions = model.predict(img)
    color_water = predictions.argmax(axis=1)[0]
    accuracy = predictions[0][color_water]
    return dic[color_water], float(accuracy)

@app.route("/health", methods=['GET'])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=['POST'])
def predict():
    try:
        img_bytes = None

        # Check for JSON input (base64 encoded image)
        if request.is_json:
            data = request.get_json()
            if 'image' in data:
                image_data = data['image']
                image_data = image_data.split(',')[1]  # Strip off base64 prefix
                img_bytes = base64.b64decode(image_data)
            else:
                return jsonify({"error": "No image data provided"}), 400
        # Check for file uploads
        elif 'file' in request.files:
            img = request.files['file']
            if img and allowed_file(img.filename):
                img_bytes = img.read()
            else:
                return jsonify({"error": "Invalid file format"}), 400
        else:
            return jsonify({"error": "No image provided"}), 400

        # Ensure img_bytes is not None before calling the prediction
        if img_bytes is None:
            return jsonify({"error": "No valid image data provided"}), 400

        # Predict label
        color_water, accuracy = predict_label(img_bytes)

        return jsonify({
            "prediction": color_water,
            "confidence": f"{accuracy:.2%}"
        })
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({"error": "An error occurred during prediction"}), 500

def allowed_file(filename):
    allowed_extensions = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=False)
