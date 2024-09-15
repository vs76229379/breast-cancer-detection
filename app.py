import os
import numpy as np
from flask import Flask, request, render_template, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename
import tensorflow as tf
import gc

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Ensure TensorFlow uses CPU only

app = Flask(__name__)

# Directory to save uploaded files
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'save')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load your pre-trained model
model_path = os.path.join(os.getcwd(), 'final_mkc_model.h5')
model = load_model(model_path)

# Class labels
class_labels = ['cancer', 'noncancer', 'others']

def model_predict(img_path, model):
    try:
        # Load and preprocess the image
        img = load_img(img_path, target_size=(224, 224))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0

        # Make prediction
        prediction = model.predict(img)
        return prediction
    except Exception as e:
        print(f"Error in model prediction: {e}")
        return None
    finally:
        tf.keras.backend.clear_session()  # Clear TensorFlow session
        gc.collect()  # Force garbage collection

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/BreastCancerInfo', methods=['GET'])
def breast_cancer_info():
    return render_template('BreastCancerInfo.html')

@app.route('/contributors', methods=['GET'])
def contributors():
    return render_template('contributors.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            prediction = model_predict(file_path, model)
            if prediction is None:
                return "Error processing the image"

            result_index = np.argmax(prediction, axis=1)[0]
            result_label = class_labels[result_index]

            result_text = f"Prediction: {result_label}"

            if os.path.exists(file_path):
                os.remove(file_path)

            return render_template('result.html', result_text=result_text, filename=filename)
        except Exception as e:
            print(f"Error during file processing or prediction: {e}")
            return "An error occurred during prediction"

    return "Method not allowed", 405

if __name__ == '__main__':
     app.run(debug=True)
