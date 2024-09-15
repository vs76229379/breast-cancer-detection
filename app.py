import os
import numpy as np
from flask import Flask, request, render_template, send_from_directory
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

# Load the TFLite model
model_path = os.path.join(os.getcwd(), 'model.tflite')
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class labels
class_labels = ['cancer', 'noncancer', 'others']

def model_predict(img_path, interpreter):
    try:
        # Load and preprocess the image
        print(f"Loading image from path: {img_path}")
        img = load_img(img_path, target_size=(224, 224))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0  # Normalize image to 0-1 range

        # Debugging the image shape and dtype
        print(f"Image shape: {img.shape}, dtype: {img.dtype}")

        # Ensure the input tensor is of the correct dtype
        img = img.astype(np.float32)

        # Set input tensor and check its shape and details
        interpreter.set_tensor(input_details[0]['index'], img)
        input_shape = input_details[0]['shape']
        input_dtype = input_details[0]['dtype']
        print(f"Expected input shape: {input_shape}, dtype: {input_dtype}")

        # Invoke the interpreter
        interpreter.invoke()

        # Get output tensor and debug output
        prediction = interpreter.get_tensor(output_details[0]['index'])
        print(f"Prediction output: {prediction}")
        
        return prediction
    except Exception as e:
        print(f"Error in model prediction: {e}")
        return None
    finally:
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

            print(f"Saved file to: {file_path}")

            # Call prediction function
            prediction = model_predict(file_path, interpreter)
            if prediction is None:
                print("Error: Prediction returned None.")
                return "Error processing the image"

            # Retrieve the predicted label
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
