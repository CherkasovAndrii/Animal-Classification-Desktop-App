import json

from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Define the custom function for RMSE
def rmse(y_true, y_pred):
    return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true)))

# Register the custom function
tf.keras.utils.get_custom_objects().update({'rmse': rmse})

# Load your predefined models
models = {
    'one': load_model('animal_upd_250_cats_added_3.keras', custom_objects={'rmse': rmse}),
    'two': load_model('updated_vgg16.keras', custom_objects={'rmse': rmse}),
    'three': load_model('animal_upd_250_cats_added_3.keras', custom_objects={'rmse': rmse})
}

# Path where uploaded models will be stored
upload_dir = 'uploaded_models'

# Function to preprocess the image
def preprocess_image(img_path, model):
    # Get the expected input shape from the model
    input_shape = model.input_shape[1:3]  # This will give you (height, width)
    
    img = image.load_img(img_path, target_size=input_shape)  # Use the model's input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the image if required by your model
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    try:
        img_file = request.files['file']
        model_choice = request.form.get('model', 'one')
        model = models.get(model_choice, models['one'])  # Fallback to default model
        
        if model is None:
            return jsonify({'error': 'Model not found.'}), 400
        
        # Load class indices for the specific model
        class_indices_file = f'{model_choice}_class_indices.json'  # Assuming naming convention
        with open(class_indices_file, 'r') as f:
            class_indices = json.load(f)
        
        img_path = os.path.join('uploads', img_file.filename)
        img_file.save(img_path)
        
        img_array = preprocess_image(img_path, model)
        
        preds = model.predict(img_array)
        top_preds = preds[0].argsort()[-5:][::-1]
        
        response = [{'label': class_indices[i], 'score': float(preds[0][i])} for i in top_preds]
        
        return jsonify(response)
    except Exception as e:
        app.logger.error(f"Error processing the request: {e}")
        return jsonify({'error': str(e)}), 500
    
    
@app.route('/upload_model', methods=['POST'])
def upload_model():
    global models  # Reference the global models dictionary
    
    if 'model' not in request.files:
        return 'No model file part', 400
    
    model_file = request.files['model']
    
    if model_file.filename == '':
        return 'No selected model file', 400
    
    # Ensure the directory exists
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    
    # Save the uploaded model
    model_path = os.path.join(upload_dir, model_file.filename)
    model_file.save(model_path)

    # Load the newly uploaded model
    try:
        new_model_name = os.path.splitext(model_file.filename)[0]  # Use the filename (without extension) as the key
        models[new_model_name] = load_model(model_path, custom_objects={'rmse': rmse})
        return f'Model uploaded and loaded successfully as "{new_model_name}".', 200
    except Exception as e:
        app.logger.error(f"Error loading model: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(port=5000)
