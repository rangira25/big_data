import os
from django.http import JsonResponse
from django.shortcuts import render
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import base64
from io import BytesIO

# Load the basic model and try loading the advanced model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS = {"basic": load_model(os.path.join(BASE_DIR, "mnist_model.h5"))}

# Optional: Attempt to load the advanced model if available
try:
    MODELS["advanced"] = load_model(os.path.join(BASE_DIR, "mnist_model_advanced.h5"))
except FileNotFoundError:
    print("Advanced model not found. Defaulting to the basic model.")

def home(request):
    """Render the home page."""
    return render(request, 'home.html')

def predict_digit(request):
    """Handle digit classification."""
    if request.method == "POST":
        try:
            # Get the selected model (default to basic if not specified)
            selected_model = request.POST.get('model_choice', 'basic')
            model = MODELS.get(selected_model, MODELS['basic'])

            # Check if the user uploaded an image
            if 'digit_image' in request.FILES:
                # Process uploaded file (image)
                image = request.FILES['digit_image']
                img = Image.open(image).convert('L').resize((28, 28))
            else:
                # Process base64 encoded canvas image (for drawing)
                data_url = request.POST.get('digit_image')
                _, encoded = data_url.split(',', 1)  # Get base64 portion
                img_data = base64.b64decode(encoded)
                img = Image.open(BytesIO(img_data)).convert('L').resize((28, 28))

            # Preprocess image
            img_array = np.array(img) / 255.0  # Normalize pixel values
            img_array = img_array.reshape(1, 28, 28, 1)  # Add batch dimension

            # Make prediction using the model
            predictions = model.predict(img_array)
            digit = int(np.argmax(predictions))  # Get the predicted digit
            probabilities = [float(f"{p:.4f}") for p in predictions[0]]

            return JsonResponse({
                'digit': digit,
                'probabilities': probabilities
            }, status=200)

        except Exception as e:
            print(f"Error: {e}")
            return JsonResponse({'error': 'Error processing the image'}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=405)
