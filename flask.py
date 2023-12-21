# Import necessary libraries
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Initialize Flask application
app = Flask(__name__)

# Load the trained model
model = load_model('my_image_classifier.h5')

# Define endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img = image.load_img(file, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Rescale to match training data preprocessing
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        
        # Assuming classes: 0 - human, 1 - horses, 2 - flowers, 3 - dogs, 4 - cats
        class_labels = ['human', 'horses', 'flowers', 'dogs', 'cats']
        result = {'class': class_labels[predicted_class]}
        
        return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
