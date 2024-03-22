
# from flask import Flask, request, jsonify
# import tensorflow as tf
# from PIL import Image
# import numpy as np
# import io
#
# app = Flask(__name__)
#
# # Function to load the model
# def load_model():
#     model = tf.keras.models.load_model('image_classification_model.h5')
#     return model
#
# model = load_model()
#
# # Function to preprocess the image
# def preprocess_image(image, target_size):
#     if image.mode != "RGB":
#         image = image.convert("RGB")
#     image = image.resize(target_size)
#     image = np.expand_dims(image, axis=0)
#     return image
#
# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file provided'}), 400
#
#     file = request.files['file']
#     image = Image.open(io.BytesIO(file.read()))
#     processed_image = preprocess_image(image, target_size=(32, 32))
#
#     prediction = model.predict(processed_image).tolist()
#
#     return jsonify({'prediction': prediction})
#
# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# Class names, assuming CIFAR-10 for example
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Function to load the model
def load_model():
    model = tf.keras.models.load_model('image_classification_model.h5')
    return model

model = load_model()

# Function to preprocess the image
def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/')
def index():
    # Ensure the HTML file is named appropriately in the templates folder
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    image = Image.open(io.BytesIO(file.read()))
    processed_image = preprocess_image(image, target_size=(32, 32))

    prediction = model.predict(processed_image)
    # Convert the prediction to the class name
    predicted_class = class_names[np.argmax(prediction)]

    return jsonify({'prediction': predicted_class})

if __name__ == "__main__":
    app.run(debug=True)

