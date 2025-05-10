import tensorflow as tf
import os

# Path to the model
project_dir = './chest_xray_ai_project'
model_path = f'{project_dir}/models/best_model.h5'  # Update to 'model.h5' if renamed

# Debug: Print absolute path
absolute_model_path = os.path.abspath(model_path)
print(f"Attempting to load model from: {absolute_model_path}")

# Check if file exists
if not os.path.exists(absolute_model_path):
    raise FileNotFoundError(f"Model file not found at {absolute_model_path}")

# Load the model
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully!")