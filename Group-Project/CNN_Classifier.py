
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from read_json import read_json_folder

from gensim.parsing.preprocessing import remove_stopwords
import gensim.downloader as api

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler,SequentialSampler
import torch.nn.functional as F

from torchtext.data.utils import get_tokenizer

from datasets import Dataset, load_dataset

import nltk
import os
from sklearn.model_selection import train_test_split
nltk.download('punkt')
nltk.download('stopwords')

#------------------------------------------------------------------------------------
#
# Device set up and import json data
#
#------------------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device available for running: ")
print(device)


print("Reading data...")
df, json_data_list = read_json_folder('../data/jsons')
df['full_content'] = df['title'] + ' ' + df['content']
df = df.drop(['topic', 'source', 'url', 'date', 'authors','title', 'content',
              'content_original', 'source_url', 'bias_text','ID'], axis=1)


#------------------------------------------------------------------------------------
#
# Text preprocessing and data loading
#
#------------------------------------------------------------------------------------
class TextDataset(Dataset):
    def __init__(self, df, text_col, label_col, max_doc_length, tokenizer, word2vec_model):
        self.df = df
        self.text_col = text_col
        self.label_col = label_col
        self.max_doc_length = max_doc_length
        self.tokenizer = tokenizer
        self.word2vec_model = word2vec_model

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = self.df[self.text_col].iloc[index]
        label = self.df[self.label_col].iloc[index]

        tokens = self.tokenizer(text)

        embeddings = []
        for token in tokens:
            if token in self.word2vec_model.key_to_index:
                embedding = self.word2vec_model[token]
                embedding = torch.from_numpy(embedding)
            else:
                embedding = torch.zeros(self.word2vec_model.vector_size)
            embeddings.append(embedding)

        # Pad or truncate the embeddings to the desired length
        embeddings = embeddings[:self.max_doc_length]
        embeddings = embeddings + [torch.zeros(self.word2vec_model.vector_size)] * (
                    self.max_doc_length - len(embeddings))

        # Stack the embeddings into a tensor
        text_tensor = torch.stack(embeddings)

        # Convert label to tensor
        label_tensor = torch.tensor(label, dtype=torch.long)

        return text_tensor, label_tensor

def tokenizer(text):
    no_stop = remove_stopwords(text)
    tokenizer = get_tokenizer("basic_english")
    tokens = tokenizer(no_stop)
    return tokens

word2vec_model = api.load("word2vec-google-news-300")

text_column = 'full_content'
label_column = 'bias'
max_doc_length = 500
batch_size = 32

print('Loading and tokenizing data...')
dataset = TextDataset(df, text_column, label_column, max_doc_length, tokenizer, word2vec_model)
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

#------------------------------------------------------------------------------------
#
# Build CNN model
#
#------------------------------------------------------------------------------------

class CNN(nn.Module):
    def __init__(self, embedding_dim, num_filters, filter_sizes, num_classes, dropout):
        super(CNN, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.num_classes = num_classes
        self.dropout = dropout

        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters,(fsz,embedding_dim)) for fsz in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)


    def forward(self, x):

        x = x.unsqueeze(1)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]

        x = torch.cat(x, 1)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.fc(x)

        return x



embedding_dim = word2vec_model.vector_size
num_filters = 100
filter_sizes = [3, 4, 5]
num_classes = 3
dropout = 0.5
epochs = 50
lr = 0.001

print("Building model...")
model = CNN(embedding_dim, num_filters, filter_sizes, num_classes, dropout)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr)


#------------------------------------------------------------------------------------
#
# Train model
#
#------------------------------------------------------------------------------------

print("Training model...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for batch_data in train_dataloader:
        inputs, labels = batch_data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()
        running_loss += loss.item()

    # Print the average loss for the epoch
    epoch_loss = running_loss / len(train_dataloader)
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}")

#------------------------------------------------------------------------------------
#
# Evaluate model
#
#------------------------------------------------------------------------------------

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_data in test_dataloader:
            inputs, labels = batch_data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Print the accuracy for the testing set
    accuracy = correct / total
    print(f"Testing Accuracy: {accuracy:.4f}")
