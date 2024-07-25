from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Define the custom function
def rmse(y_true, y_pred):
    return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true)))

# Register the custom function
tf.keras.utils.get_custom_objects().update({'rmse': rmse})

# Load your custom model with the custom function
model = load_model('animal_upd_250_cats_added_3.keras', custom_objects={'rmse': rmse})

# Define a function to preprocess the image as needed by your model
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(64, 64))  # Adjust the target size as per your model's input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the image if required by your model
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    try:
        img_file = request.files['file']
        img_path = os.path.join('uploads', img_file.filename)
        img_file.save(img_path)
        
        img_array = preprocess_image(img_path)
        
        preds = model.predict(img_array)
        # Assuming your model outputs probabilities for each class
        # Modify this according to your model's output
        class_indices = ['bear', 'cat', 'crow', 'elephant', 'rat']  # List of class names
        top_preds = preds[0].argsort()[-5:][::-1]
        
        response = [{'label': class_indices[i], 'score': float(preds[0][i])} for i in top_preds]
        
        return jsonify(response)
    except Exception as e:
        app.logger.error(f"Error processing the request: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(port=5000)
