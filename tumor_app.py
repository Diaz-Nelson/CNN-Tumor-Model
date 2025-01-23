from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load your trained model (replace with the correct model path)
model = load_model('tumor_model.h5')

# Define your labels (if applicable)
labels = ['class_0', 'class_1', 'class_2', 'class_3']  # Replace with actual class names

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route for handling file upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    if file:
        # Load the image and process it
        img = image.load_img(file, target_size=(150, 150))
        img_array = image.img_to_array(img) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make prediction
        prediction = model.predict(img_array)
        predicted_label = np.argmax(prediction, axis=1)  # Get the predicted class
        predicted_label_name = labels[predicted_label[0]]

        return f"Predicted: {predicted_label_name} with a confidence score of {round(prediction[0][predicted_label[0]] * 100, 2)}%"

if __name__ == '__main__':
    app.run(debug=True)
