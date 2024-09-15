import os
import numpy as np
from flask import Flask, request, render_template, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename
from tensorflow.keras import backend as K

# Suppress TensorFlow logging and disable GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress most TensorFlow logging
os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION'] = 'true'  # Ensure GPU memory is managed properly
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU devices

app = Flask(__name__)

# Directory to save uploaded files
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'save')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load your pre-trained model
model_path = os.path.join(os.getcwd(), 'final_mkc_model.h5')  # Update with your model file path
model = load_model(model_path)

# Class labels
class_labels = ['cancer', 'noncancer', 'others']

def model_predict(img_path, model):
    try:
        # Load and preprocess the image
        img = load_img(img_path, target_size=(224, 224))  # Resize image to 224x224
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0  # Normalize the image to the range [0, 1]

        # Make prediction
        prediction = model.predict(img)
        return prediction
    except Exception as e:
        print(f"Error in model prediction: {e}")
        return None

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/BreastCancerInfo', methods=['GET'])
def breast_cancer_info():
    # Breast Cancer Info page
    return render_template('BreastCancerInfo.html')

@app.route('/contributors', methods=['GET'])
def contributors():
    # Contributors page
    return render_template('contributors.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file:
            try:
                # Save the uploaded file
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                # Make prediction
                prediction = model_predict(file_path, model)
                if prediction is None:
                    return "Error processing the image"

                # Process the prediction and return the result
                result_index = np.argmax(prediction, axis=1)[0]  # Get the index of the predicted class
                result_label = class_labels[result_index]  # Get the class label

                result_text = f"Prediction: {result_label}"

                # Clear TensorFlow session to free up memory
                K.clear_session()

                # Remove file after prediction
                if os.path.exists(file_path):
                    os.remove(file_path)

                return render_template('result.html', result_text=result_text, filename=filename)
            except Exception as e:
                print(f"Error during file processing or prediction: {e}")
                return "An error occurred during prediction"

    return "Method not allowed", 405

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Default to 5000 if PORT is not set
    app.run(host='0.0.0.0', port=port, debug=False)
