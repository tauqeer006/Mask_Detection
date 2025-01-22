from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os

app = Flask(__name__)

model = load_model("mask_detection_model.h5")

@app.route('/')
def index():
    return render_template('index.html')  

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    image_path = os.path.join("uploads", file.filename)
    file.save(image_path)

    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224)) 
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)[0]

    class_names = {0: "With Mask", 1: "Without Mask", 2: "Improper Mask"}
    result = class_names[predicted_class]
    print("the predicted class is : ",result)

    os.remove(image_path)

    return jsonify({'prediction': result})

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)
