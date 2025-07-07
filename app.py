from flask import Flask, request, jsonify
import os
from predict_pet import predict_for_api

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    img = request.files['image']
    img_path = os.path.join(UPLOAD_FOLDER, img.filename)
    img.save(img_path)

    try:
        result = predict_for_api(img_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)