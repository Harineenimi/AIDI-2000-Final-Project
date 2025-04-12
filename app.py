from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.image import load_img, img_to_array  # type: ignore
from tensorflow.keras.applications.densenet import preprocess_input  # type: ignore
import numpy as np
import os
import traceback

app = Flask(__name__)
STATIC_UPLOADS = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = STATIC_UPLOADS

# Ensure uploads folder exists
os.makedirs(STATIC_UPLOADS, exist_ok=True)

# Class labels
class_names = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

@app.route('/')
def index():
    return render_template('index.html', prediction=None, image_path=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Load the model inside the route to reduce memory use on startup
        from tensorflow.keras.models import load_model
        model = load_model('model_bccd.h5', compile=False)

        # Clean uploads directory
        if os.path.exists(app.config['UPLOAD_FOLDER']):
            for f in os.listdir(app.config['UPLOAD_FOLDER']):
                os.remove(os.path.join(app.config['UPLOAD_FOLDER'], f))

        if 'file' not in request.files:
            return 'No file uploaded!', 400

        file = request.files['file']
        if file.filename == '':
            return 'No selected file!', 400

        # Save uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Preprocess the image
        img = load_img(filepath, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Predict
        prediction = model.predict(img_array)
        class_index = np.argmax(prediction, axis=1)[0]
        result = class_names[class_index]

        return render_template('index.html', prediction=result, image_path='uploads/' + file.filename)

    except Exception as e:
        print("❌ Prediction error:", e)
        traceback.print_exc()
        return f"Internal Server Error: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)
