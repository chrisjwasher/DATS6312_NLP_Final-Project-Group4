
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from gensim.parsing.preprocessing import remove_stopwords
import gensim.downloader as api

from torchtext.data.utils import get_tokenizer

#------------------------------------------------------------------------------------
#
# Device set up and import json data
#
#------------------------------------------------------------------------------------


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device available for running: ")
print(device)


print("Reading data...")
def read_json_folder(folder_path):
    """
    Read JSON files from a folder and return a list of dictionaries.
    Args:
        folder_path (str): Path to the folder containing JSON files.
    Returns:
        list: A list of dictionaries containing data from each JSON file.
    """
    json_data_list = []

    # Check if the folder path exists
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return json_data_list

    # Iterate over files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # Check if the file is a JSON file
        if filename.endswith('.json'):
            with open(file_path, 'r') as f:
                # Load JSON data from the file
                try:
                    json_data = json.load(f)
                    json_data_list.append(json_data)
                except json.JSONDecodeError:
                    print(f"Error reading JSON from file: {file_path}")
                    continue

    df = pd.DataFrame.from_dict(json_data_list)

    return df, json_data_list


df, json_data_list = read_json_folder('data/jsons')
df['full_content'] = df['title'] + ' ' + df['content']
df = df.drop(['topic', 'source', 'url', 'date', 'authors','title', 'content',
              'content_original', 'source_url', 'bias_text','ID'], axis=1)

'''
#------------------------------------------------------------------------------------
#
# Text preprocessing and data loading
#
#------------------------------------------------------------------------------------
class TextDataset(Dataset):
    def __init__(self, documents, labels, tokenizer, word2vec_model, max_length):
        self.documents = documents
        self.labels = labels
        self.tokenizer = tokenizer
        self.word2vec_model = word2vec_model
        self.max_length = max_length

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        document = self.documents[idx]
        label = self.labels[idx]

        tokens = self.tokenizer(document)

        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + ['<pad>'] * (self.max_length - len(tokens))

        token_indices = []
        for token in tokens:
            if token in self.word2vec_model.vocab:
                token_indices.append(self.word2vec_model.key_to_index[token])
            else:
                token_indices.append(0)

        return torch.tensor(token_indices), torch.tensor(label)

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
train_documents, temp_documents, train_labels, temp_labels = train_test_split(df[text_column], df[label_column], test_size=0.2, random_state=42)
test_documents, val_documents, test_labels, val_labels = train_test_split(temp_documents, temp_labels, test_size=0.2, random_state=42)


train_data = TextDataset(train_documents.reset_index(drop=True), train_labels.reset_index(drop=True), tokenizer, word2vec_model, max_doc_length)
val_data = TextDataset(val_documents.reset_index(drop=True), val_labels.reset_index(drop=True), tokenizer, word2vec_model, max_doc_length)
test_data = TextDataset(test_documents.reset_index(drop=True), test_labels.reset_index(drop=True), tokenizer, word2vec_model, max_doc_length)


train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


#------------------------------------------------------------------------------------
#
# Build LSTM model
#
#------------------------------------------------------------------------------------
print('Building model...')
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, word2vec_model):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(word2vec_model.vectors))
        self.embedding.weight.requires_grad = False  # Freeze the embedding layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, lengths):
        embedded = self.embedding(x)

        # Pack the padded sequences
        packed_embedded = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        last_hidden = hidden[-1]
        logits = self.fc(last_hidden)

        return logits



#------------------------------------------------------------------------------------
#
# Initialize LSTM model
#
#------------------------------------------------------------------------------------

vocab_size = len(word2vec_model.key_to_index)
embedding_dim = 300
hidden_dim = 128
output_dim = 3


model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, word2vec_model)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())


#------------------------------------------------------------------------------------
#
# Build LSTM model
#
#------------------------------------------------------------------------------------
print('Training model...')
# Train the model
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    for documents, labels in train_dataloader:
        # Get sequence lengths
        lengths = torch.sum(documents != tokenizer.vocab['<pad>'], dim=1)

        # Forward pass
        logits = model(documents, lengths)
        loss = criterion(logits, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate on validation set
    model.eval()
    with torch.no_grad():
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for documents, labels in val_dataloader:
            # Get sequence lengths
            lengths = torch.sum(documents != tokenizer.vocab['<pad>'], dim=1)

            # Forward pass
            logits = model(documents, lengths)
            loss = criterion(logits, labels)

            # Compute validation metrics
            _, predicted = torch.max(logits, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            total_loss += loss.item() * labels.size(0)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.4f}")


model.eval()
with torch.no_grad():
    true_labels = []
    predicted_labels = []

    for documents, labels in test_dataloader:
        # Get sequence lengths
        lengths = torch.sum(documents != tokenizer.vocab['<pad>'], dim=1)

        # Forward pass
        logits = model(documents, lengths)

        # Compute predicted labels
        _, predicted = torch.max(logits, 1)

        true_labels.extend(labels.tolist())
        predicted_labels.extend(predicted.tolist())

    # Compute evaluation metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    cm = confusion_matrix(true_labels, predicted_labels)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

# Save accuracy and F1 score to CSV file
metrics_data = {'Metric': ['Accuracy', 'F1 Score'], 'Value': [accuracy, f1]}
metrics_df = pd.DataFrame(metrics_data)
metrics_df.to_csv('metrics.csv', index=False)

# Save confusion matrix to PNG file
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()

'''


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Define your dataset class
class TextDataset(Dataset):
    def __init__(self, documents, labels, tokenizer, word2vec_model, max_length):
        self.documents = documents
        self.labels = labels
        self.tokenizer = tokenizer
        self.word2vec_model = word2vec_model
        self.max_length = max_length

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        document = self.documents[idx]
        label = self.labels[idx]

        tokens = self.tokenizer(document)

        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + ['<pad>'] * (self.max_length - len(tokens))

        token_indices = []
        for token in tokens:
            if token in self.word2vec_model.key_to_index:
                token_indices.append(self.word2vec_model.key_to_index[token])
            else:
                token_indices.append(0)  # Use index 0 for unknown tokens

        return torch.tensor(token_indices), torch.tensor(label)

# Define your LSTM model
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, word2vec_model):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(word2vec_model.vectors))
        self.embedding.weight.requires_grad = False  # Freeze the embedding layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, lengths):
        # Embed the token indices
        embedded = self.embedding(x)

        # Pack the padded sequences
        packed_embedded = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)

        # Pass through LSTM
        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        # Unpack the padded sequences
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Take the last hidden state as the representation
        last_hidden = hidden[-1]

        # Pass through the final fully connected layer
        logits = self.fc(last_hidden)

        return logits

# Prepare your data
documents = df['full_content'].tolist()  # List of documents
labels = df['bias'].tolist()  # List of corresponding labels (0, 1, 2)

# Split the data into training, validation, and test sets
train_documents, test_documents, train_labels, test_labels = train_test_split(documents, labels, test_size=0.2, random_state=42)
train_documents, val_documents, train_labels, val_labels = train_test_split(train_documents, train_labels, test_size=0.2, random_state=42)

# Define tokenizer and load the pre-trained Word2Vec model
def tokenizer(text):
    no_stop = remove_stopwords(text)
    tokenizer = get_tokenizer("basic_english")
    tokens = tokenizer(no_stop)
    return tokens

word2vec_model = api.load("word2vec-google-news-300")

# Define maximum document length
max_doc_length = 300  # Set an appropriate value based on your data

# Create dataset instances
train_data = TextDataset(train_documents, train_labels, tokenizer, word2vec_model, max_doc_length)
val_data = TextDataset(val_documents, val_labels, tokenizer, word2vec_model, max_doc_length)
test_data = TextDataset(test_documents, test_labels, tokenizer, word2vec_model, max_doc_length)

# Create data loaders
train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=32)
test_dataloader = DataLoader(test_data, batch_size=32)

# Initialize the model
vocab_size = len(word2vec_model.key_to_index)
embedding_dim = 300  # Assuming the word2vec-google-news-300 model has 300-dimensional embeddings
hidden_dim = 128
output_dim = 3

model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, word2vec_model)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Train the model
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    for documents, labels in train_dataloader:
        # Get sequence lengths
        lengths = torch.sum(documents != 0, dim=1)

        # Forward pass
        logits = model(documents, lengths)
        loss = criterion(logits, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate on validation set
    model.eval()
    with torch.no_grad():
        val_loss = 0
        val_accuracy = 0
        val_f1 = 0
        val_samples = 0

        for documents, labels in val_dataloader:
            # Get sequence lengths
            lengths = torch.sum(documents != 0, dim=1)

            # Forward pass
            logits = model(documents, lengths)
            loss = criterion(logits, labels)

            # Compute validation metrics
            predictions = torch.argmax(logits, dim=1)
            val_loss += loss.item() * labels.size(0)
            val_accuracy += accuracy_score(labels.cpu(), predictions.cpu()) * labels.size(0)
            val_f1 += f1_score(labels.cpu(), predictions.cpu(), average='weighted') * labels.size(0)
            val_samples += labels.size(0)

        val_loss /= val_samples
        val_accuracy /= val_samples
        val_f1 /= val_samples

        print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation F1 Score: {val_f1:.4f}")

# Evaluate on test set
model.eval()
with torch.no_grad():
    test_loss = 0
    test_accuracy = 0
    test_f1 = 0
    test_samples = 0
    true_labels = []
    predicted_labels = []

    for documents, labels in test_dataloader:
        # Get sequence lengths
        lengths = torch.sum(documents != 0, dim=1)

        # Forward pass
        logits = model(documents, lengths)
        loss = criterion(logits, labels)

        # Compute test metrics
        predictions = torch.argmax(logits, dim=1)
        test_loss += loss.item() * labels.size(0)
        test_accuracy += accuracy_score(labels.cpu(), predictions.cpu()) * labels.size(0)
        test_f1 += f1_score(labels.cpu(), predictions.cpu(), average='weighted') * labels.size(0)
        test_samples += labels.size(0)
        true_labels.extend(labels.cpu().tolist())
        predicted_labels.extend(predictions.cpu().tolist())

    test_loss /= test_samples
    test_accuracy /= test_samples
    test_f1 /= test_samples

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test F1 Score: {test_f1:.4f}")

    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    # Save accuracy and F1 score to CSV file
    metrics_data = {'Metric': ['Accuracy', 'F1 Score'], 'Value': [test_accuracy, test_f1]}
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv('metrics.csv', index=False)

    # Save confusion matrix to PNG file
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()