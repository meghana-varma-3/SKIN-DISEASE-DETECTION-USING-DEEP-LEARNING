from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

app = Flask(__name__)

MODEL_PATH = r"C:\Users\megha\Desktop\.h5 files\trained_model.h5"
model = load_model(MODEL_PATH)

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(75, 100))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255.
    preds = model.predict(x)
    class_names = ['NULL', 'Basal Cell Carcinoma', 'Benign Keratosis-like Lesions ', 'Psoriasis', 'Warts',]
    predicted_class_index = np.argmax(preds)
    predicted_class_name = class_names[predicted_class_index]
    return predicted_class_name 

@app.route('/', methods=['GET'])
def index():
    return render_template('UIndex.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        preds1 = model_predict(file_path, model)
        return preds1
    return None

if __name__ == '__main__':
    app.run(debug=True)
