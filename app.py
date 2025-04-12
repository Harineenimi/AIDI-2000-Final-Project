from flask import Flask, render_template, request
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array  # type: ignore
from tensorflow.keras.applications.densenet import preprocess_input  # type: ignore
import numpy as np
import os
import shutil

app = Flask(__name__)
STATIC_UPLOADS = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = STATIC_UPLOADS

# Load trained DenseNet model (suppress layer name restrictions)
model = load_model('model_bccd.h5', compile=False)

# Class labels
class_names = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

@app.route('/')
def index():
    return render_template('index.html', prediction=None, image_path=None)

@app.route('/predict', methods=['POST'])
def predict():
    # Clean uploads directory
    for f in os.listdir(app.config['UPLOAD_FOLDER']):
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], f))

    if 'file' not in request.files:
        return 'No file uploaded!', 400

    file = request.files['file']
    if file.filename == '':
        return 'No selected file!', 400

    # Save the uploaded file to static/uploads/
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Preprocess the image
    img = load_img(filepath, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make prediction
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction, axis=1)[0]
    result = class_names[class_index]

    return render_template('index.html', prediction=result, image_path='uploads/' + file.filename)

if __name__ == '__main__':
    os.makedirs(STATIC_UPLOADS, exist_ok=True)
    app.run(debug=True)
