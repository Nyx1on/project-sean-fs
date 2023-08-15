import os
from django.shortcuts import render, HttpResponse
from django.http import JsonResponse
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define the file paths
# Get the base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Specify the relative path to the model and tokenizer files
MODEL_FILE_PATH = os.path.join(BASE_DIR, '..','model', 'sentimental_analysis_model.h5')
TOKENIZER_FILE_PATH = os.path.join(BASE_DIR,'..', 'model', 'tokenizer.pickle')

# Load the saved model
model = load_model(MODEL_FILE_PATH)

# Load the tokenizer used to preprocess the data
with open(TOKENIZER_FILE_PATH, 'rb') as f:
    tokenizer = pickle.load(f)

def index(request):
    return JsonResponse({"ServerStatus": "OK"})


def predict(request):
    if request.method == 'GET':
        # Get the input review from the GET request
        input_review = request.GET.get('input_review')

        if input_review is not None:
            # Preprocess the input review
            input_seq = tokenizer.texts_to_sequences([input_review])
            input_padded = pad_sequences(
                input_seq, maxlen=1000, padding='post', truncating='post')
            
            # Make prediction using the trained model
            prediction = model.predict(input_padded)[0][0]
        else:
            return JsonResponse({"Predictions": "Invalid input"})

        if prediction > 0.5:
            sentiment = "positive"
        else:
            sentiment = "negative"

        prediction_number = str(prediction)

        return JsonResponse({
            "Predictions": sentiment,
            "PredictionNumber": prediction_number
        })
        
    else:
        return JsonResponse({"Predictions": "Invalid method"})
