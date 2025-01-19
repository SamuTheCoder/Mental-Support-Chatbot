from django.shortcuts import render
import torch
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .data_processing import bag_of_words

from .model import NeuralNet
import random
import os

# Create your views here.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KB_PATH = os.path.join(BASE_DIR, 'KB.json')
with open(KB_PATH, 'r') as json_data:
    intents = json.load(json_data)

FILE = os.path.join(BASE_DIR, 'data.pth')
data = torch.load(FILE, map_location=device, weights_only=True)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

def index(request):
    return HttpResponse("Hello, world. You're at the chatbot index.")

@csrf_exempt
def chatbot_response(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            sentence = data.get('message', '')

            # Preprocess the sentence
            X = bag_of_words(sentence, all_words)
            X = X.reshape(1, X.shape[0])
            X = torch.from_numpy(X).to(device)

            output = model(X)
            _, predicted = torch.max(output, dim=1)

            tag = tags[predicted.item()]

            #Calculate confidence
            probs = torch.softmax(output, dim=1)
            prob = probs[0][predicted.item()]
            if prob.item() > 0.75:
                for intent in intents['intents']:
                    if tag == intent["tag"]:
                        response = random.choice(intent['responses'])
                        return JsonResponse({"response": response}) 
            else:
                return JsonResponse({"response": "I do not understand..."}, status=500)
        except Exception as e:
            return JsonResponse({"error": str(e)})
        
    return JsonResponse({"error": "Invalid request method"}, status=400)






