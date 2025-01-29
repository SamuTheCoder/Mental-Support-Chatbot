import numpy as np
import json

import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from data_processing import tokenize, stem, bag_of_words
from model import NeuralNet

import csv

# Get data from json file
with open("KB.json", "r") as f:
    intents = json.load(f)

# Data has intents, with tags, patterns and responses

all_words = []
tags = []
xy = []

for intent in intents["intents"]:
    tag = intent["tag"]
    tags.append(tag)
    for pattern in intent["patterns"]:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

# Stem and lower each word
all_words = [stem(word) for word in all_words]

# Remove duplicates
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print("all_words:", all_words)
print("tags:", tags)
print("xy:", xy)

# Create training data

X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    pattern_sentence = " ".join(pattern_sentence)
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

#Hyper-parameters
num_epochs = 250
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # Dataset[idx]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples
    
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model

#Loss accross all batches in an epoch
running_loss = 0.0

patience = 5 
best_loss = float('inf')
trigger_times = 0

with open("training.csv", mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Loss"])

    for epoch in range(num_epochs):
        running_loss = 0.0
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(words)
            # Loss for current batch
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss/len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        writer.writerow([epoch+1, epoch_loss])

        # Early stopping logic
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            trigger_times = 0  # Reset patience counter
        else:
            trigger_times += 1
            print(f"No improvement for {trigger_times} epochs.")

            if trigger_times >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}. Best loss: {best_loss:.4f}")
                break

print("Training complete.")
print(f"Final loss: {running_loss/len(train_loader):.4f}")

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'Training complete. File saved to {FILE}')

print('Plotting loss graph...')

epochs = []
losses = []
with open("training.csv", mode='r') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        epochs.append(int(row[0]))
        losses.append(float(row[1]))

plt.plot(epochs, losses, label='Training loss', marker='o', color='r', linestyle='-')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.grid(True)
plt.savefig('loss.png')
print('Loss graph saved to loss.png')

