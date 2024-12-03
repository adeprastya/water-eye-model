from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import base64
import io
import logging
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Set up logging for production
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model
model = load_model(os.getenv('MODEL_PATH', 'model/WaterEye-v0.h5'))
model.make_predict_function()

# Define labels
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

def predict_label(img_bytes):
    # Load image resize, and normalize
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
        # Check for JSON input
        if request.is_json:
            data = request.get_json()
            if 'image_data' in data:
                image_data = data['image_data']
                image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
            else:
                return jsonify({"error": "No image data provided"}), 400
        elif 'file' in request.files:  # Handle file uploads
            img = request.files['file']
            if img and allowed_file(img.filename):
                img_bytes = img.read()
            else:
                return jsonify({"error": "Invalid file format"}), 400
        else:
            return jsonify({"error": "No image provided"}), 400

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
