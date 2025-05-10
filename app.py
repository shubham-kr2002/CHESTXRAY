from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
import cv2
import os
import json

app = Flask(__name__)

# Parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
CLASS_NAMES = ['normal', 'pneumonia', 'tuberculosis']

# Paths: Use the current working directory as the project directory
project_dir = os.getcwd()  # This will be C:\Users\shubh\chest_xray_ai_project
model_dir = os.path.join(project_dir, 'models')
model_path = os.path.join(model_dir, 'best_model.h5')
class_indices_file = os.path.join(project_dir, 'class_indices.json')

# Debug: Print paths to confirm
print(f"Project directory: {project_dir}")
print(f"Model directory: {model_dir}")
print(f"Model path: {model_path}")

# Debug: Check if the model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}. Please ensure 'best_model.h5' exists in {model_dir}")

# Load model and class indices
model = tf.keras.models.load_model(model_path)
print(f"Model input shape: {model.input_shape}")  # Debug: Confirm model's expected input shape
with open(class_indices_file, 'r') as f:
    class_indices = json.load(f)

def preprocess_image(image_path):
    """Preprocess an uploaded image for prediction, matching the model's expected input shape."""
    # Step 1: Read the image as grayscale (to match training and model expectation)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Failed to load image")
    print(f"Shape after reading (grayscale): {img.shape}")

    # Step 2: Resize the image
    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))  # Shape: (224, 224)
    print(f"Shape after resizing: {img.shape}")

    # Step 3: Normalize to [0, 1]
    img = img / 255.0  # Shape: (224, 224)
    print(f"Shape after normalization: {img.shape}")

    # Step 4: Add channel dimension (model expects 1 channel)
    img = np.expand_dims(img, axis=-1)  # Shape: (224, 224, 1)
    print(f"Shape after adding channel dimension: {img.shape}")

    # Step 5: Add batch dimension
    img = np.expand_dims(img, axis=0)  # Shape: (1, 224, 224, 1)
    print(f"Shape after adding batch dimension: {img.shape}")

    # Step 6: Final shape validation
    expected_shape = (1, 224, 224, 1)
    if img.shape != expected_shape:
        raise ValueError(f"Preprocessing failed: Expected shape {expected_shape}, but got {img.shape}")

    return img

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
    
    # Save uploaded file
    upload_dir = os.path.join(project_dir, 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, 'uploaded_image.png')
    file.save(file_path)
    
    try:
        # Preprocess and predict
        img = preprocess_image(file_path)
        
        # Debug: Verify shape before prediction
        print(f"Shape before prediction: {img.shape}")
        expected_shape = (1, 224, 224, 1)
        if img.shape != expected_shape:
            raise ValueError(f"Unexpected shape before prediction: Expected {expected_shape}, but got {img.shape}")
        
        # Convert to TensorFlow tensor and verify shape
        img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
        print(f"Shape after converting to tensor: {img_tensor.shape}")
        
        prediction = model.predict(img_tensor)
        print(f"Prediction shape: {prediction.shape}")
        
        predicted_class_idx = np.argmax(prediction[0])
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = prediction[0][predicted_class_idx] * 100
        
        # Get all confidences
        confidences = {CLASS_NAMES[i]: f'{prediction[0][i] * 100:.2f}' for i in range(len(CLASS_NAMES))}
        
        return render_template('result.html', 
                             prediction=predicted_class, 
                             confidence=f'{confidence:.2f}',
                             confidences=confidences,
                             image_path=file_path)
    except Exception as e:
        return render_template('index.html', error=f'Error processing image: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)