from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
import cv2
import os
import json
import requests  # For making API calls to Grok

app = Flask(__name__)

# Parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
CLASS_NAMES = ['normal', 'pneumonia', 'tuberculosis']

# Paths
project_dir = os.getcwd()
model_dir = os.path.join(project_dir, 'models')
model_path = os.path.join(model_dir, 'best_model.h5')
class_indices_file = os.path.join(project_dir, 'class_indices.json')

# Grok API Configuration
GROK_API_URL = "https://api.x.ai/v1/grok"  # Hypothetical endpoint
GROK_API_KEY = "your_grok_api_key_here"  # Replace with your API key

# Debug: Print paths to confirm
print(f"Project directory: {project_dir}")
print(f"Model directory: {model_dir}")
print(f"Model path: {model_path}")

# Debug: Check if the model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}. Please ensure 'best_model.h5' exists in {model_dir}")

# Load model and class indices
model = tf.keras.models.load_model(model_path)
print(f"Model input shape: {model.input_shape}")
with open(class_indices_file, 'r') as f:
    class_indices = json.load(f)

def preprocess_image(image_path):
    """Preprocess an uploaded image for prediction, matching the model's expected input shape."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Failed to load image")
    print(f"Shape after reading (grayscale): {img.shape}")

    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    print(f"Shape after resizing: {img.shape}")

    img = img / 255.0
    print(f"Shape after normalization: {img.shape}")

    img = np.expand_dims(img, axis=-1)
    print(f"Shape after adding channel dimension: {img.shape}")

    img = np.expand_dims(img, axis=0)
    print(f"Shape after adding batch dimension: {img.shape}")

    expected_shape = (1, 224, 224, 1)
    if img.shape != expected_shape:
        raise ValueError(f"Preprocessing failed: Expected shape {expected_shape}, but got {img.shape}")

    return img

def get_grok_explanation(prediction, confidences):
    """Use Grok API to generate an explanation for the prediction."""
    prompt = f"The chest X-ray classifier predicted {prediction} with the following confidences: {confidences}. Provide a detailed explanation of this prediction, including possible reasons for the classification and a disclaimer that this is not a medical diagnosis."
    
    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "grok-beta",
        "prompt": prompt,
        "max_tokens": 500
    }
    
    try:
        response = requests.post(GROK_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        explanation = response.json().get("choices", [{}])[0].get("text", "No explanation available.")
    except Exception as e:
        explanation = f"Error fetching explanation from Grok API: {str(e)}"
    
    return explanation

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='No file uploaded')
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='No file selected')
    
    upload_dir = os.path.join(project_dir, 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, 'uploaded_image.png')
    file.save(file_path)
    
    try:
        img = preprocess_image(file_path)
        
        print(f"Shape before prediction: {img.shape}")
        expected_shape = (1, 224, 224, 1)
        if img.shape != expected_shape:
            raise ValueError(f"Unexpected shape before prediction: Expected {expected_shape}, but got {img.shape}")
        
        img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
        print(f"Shape after converting to tensor: {img_tensor.shape}")
        
        prediction = model.predict(img_tensor)
        print(f"Prediction shape: {prediction.shape}")
        
        predicted_class_idx = np.argmax(prediction[0])
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = prediction[0][predicted_class_idx] * 100
        
        confidences = {CLASS_NAMES[i]: f'{prediction[0][i] * 100:.2f}' for i in range(len(CLASS_NAMES))}
        
        # Get explanation from Grok API
        explanation = get_grok_explanation(predicted_class, confidences)
        
        return render_template('result.html', 
                             prediction=predicted_class, 
                             confidence=f'{confidence:.2f}',
                             confidences=confidences,
                             explanation=explanation,
                             image_path=file_path)
    except Exception as e:
        return render_template('index.html', error=f'Error processing image: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)