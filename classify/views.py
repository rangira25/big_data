import os
from django.shortcuts import render
from tensorflow.keras.models import load_model # type: ignore
from PIL import Image
import numpy as np

# Conditionally load the model only when the app is running, not during migrations
model = None
if os.environ.get('RUN_MAIN', None) == 'true':  # RUN_MAIN ensures model loads only in the main process
    model_path = os.path.join(os.path.dirname(__file__), "mnist_model.h5")  # Path in classify app
    model = load_model(model_path)

def classify_digit(request):
    """
    View to handle digit classification.
    Accepts an image file upload, preprocesses it, and uses the model to predict the digit.
    """
    if request.method == "POST":
        image = request.FILES['digit_image']  # Get the uploaded file
        img = Image.open(image).convert('L')  # Convert to grayscale
        img = img.resize((28, 28))  # Resize to 28x28 pixels
        img_array = np.array(img) / 255.0  # Normalize pixel values
        img_array = img_array.reshape(1, 28, 28, 1)  # Add batch dimension for model input

        # Predict using the model
        prediction = np.argmax(model.predict(img_array))
        return render(request, 'result.html', {'digit': prediction})  # Pass result to template

    return render(request, 'upload.html')  # Render upload page for GET requests
