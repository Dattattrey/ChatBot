import random
import json
import torch
import pywhatkit as pwt
from model import NeuralNet
import nltk_utils
from subprocess import call
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# Creating class for importing the image file
class CallImage(object):

    # Defining path of the image file (relative to current script)
    def __init__(self, path=None):
        if path is None:
            base_dir = os.path.dirname(__file__)
            self.path = os.path.join(base_dir, "image.py")
        else:
            self.path = path

    def call_image(self):
        call(["python", self.path])


bot_name = "Shein"
print("Let's Chat! (type 'quit'/'exit' to exit)")
while True:
    sentence = input("You: ")

    if sentence.lower() in ["google", "search on google"]:
        sentence = input("What do you want to search: ")
        pwt.search(sentence)
        continue

    # If the user types 'download', image file runs
    if sentence.lower() == "download":
        print("Image Downloader opening...")
        c = CallImage()
        c.call_image()
        continue

    if sentence.lower() in ["youtube", "search on youtube"]:
        sentence = input("What do you want to see: ")
        print("Playing...")
        pwt.playonyt(sentence)
        continue

    if sentence.lower() in ["quit", "exit"]:
        break

    # NLP processing starts
    sentence = nltk_utils.tokenize(sentence)
    X = nltk_utils.bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.90:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand...")