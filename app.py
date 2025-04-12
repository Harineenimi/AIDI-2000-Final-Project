from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.image import load_img, img_to_array  # type: ignore
from tensorflow.keras.applications.densenet import preprocess_input  # type: ignore
import numpy as np
import os
import traceback
import tensorflow as tf  # TensorFlow Lite is included here

app = Flask(__name__)
STATIC_UPLOADS = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = STATIC_UPLOADS

# Ensure uploads folder exists
os.makedirs(STATIC_UPLOADS, exist_ok=True)

# Class labels
class_names = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

# Load TFLite model only once
interpreter = tf.lite.Interpreter(model_path="model_bccd_compressed.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.route('/')
def index():
    return render_template('index.html', prediction=None, image_path=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Clean uploads directory
        if os.path.exists(app.config['UPLOAD_FOLDER']):
            for f in os.listdir(app.config['UPLOAD_FOLDER']):
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], f)
                if os.path.isfile(file_path):
                    os.remove(file_path)

        if 'file' not in request.files:
            return 'No file uploaded!', 400

        file = request.files['file']
        if file.filename == '':
            return 'No selected file!', 400

        # Save uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Preprocess image
        img = load_img(filepath, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)  # If model expects DenseNet-preprocessed input

        # Convert to float32 if required by the TFLite model
        if input_details[0]['dtype'] == np.float32:
            img_array = img_array.astype(np.float32)

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], img_array)

        # Run inference
        interpreter.invoke()

        # Get prediction
        output_data = interpreter.get_tensor(output_details[0]['index'])
        class_index = np.argmax(output_data, axis=1)[0]
        result = class_names[class_index]

        return render_template('index.html', prediction=result, image_path='uploads/' + file.filename)

    except Exception as e:
        print("❌ Prediction error:", e)
        traceback.print_exc()
        return f"Internal Server Error: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)
