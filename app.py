from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image

app = Flask(__name__)

# Load the trained CNN model
model = load_model("best_model.h5")  # Make sure it's the final trained model

# Define model input size
IMG_SIZE = (128, 128)

# Class labels based on training
class_labels = ["NORMAL", "PNEUMONIA"]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    # Save uploaded file
    filepath = os.path.join('static', file.filename)
    file.save(filepath)

    # ✅ Preprocess: RGB, resize, normalize
    img = Image.open(filepath).convert("RGB").resize(IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict (softmax output)
    prediction = model.predict(img_array)
    print("Predicted probabilities:", prediction)

    predicted_class = int(np.argmax(prediction, axis=1)[0])
    label = class_labels[predicted_class]
    confidence = float(prediction[0][predicted_class]) * 100

    return render_template('result.html', prediction=label, confidence=confidence, img_path=filepath)

if __name__ == '__main__':
    app.run(debug=True)
