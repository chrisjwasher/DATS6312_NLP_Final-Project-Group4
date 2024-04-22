import time
import os
import csv
import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
from nltk.corpus import stopwords
from collections import Counter
import re
import gensim.downloader as api
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


import nltk
from nltk.tokenize.casual import casual_tokenize
from nltk.stem import WordNetLemmatizer


import re

is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

#%%

# ******************
#     Read data
# ******************
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

df, json_data_list = read_json_folder('Group Project/data/jsons')


#%%
# ******************
#   Preprocessing
# ******************

nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('english')

def preprocess(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r"[^\w\s]", ' ', text)
    # Remove stopwords
    word_tokens = casual_tokenize(text)
    filtered_text = [w for w in word_tokens if w not in stop_words]
    text = ' '.join(filtered_text)
    # Lemmatize text
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(w) for w in casual_tokenize(text)]
    return ' '.join(lemmatized)


df['content_clean'] = df['content'].apply(preprocess)
#%%

def split(df):
    X = df.drop(columns=['topic', 'source', 'url',
                         'title', 'date', 'authors', 'content_original',
                         'source_url', 'bias_text', 'ID', 'bias'])
    y = df['bias']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=False)
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = split(df)

#%%
# ******************
#   padding and packing the articles to max length 300
# ******************
# Calculate vocabulary size from training data
vocab = set()
for sentence in X_train:
    vocab.update(sentence.split())

def padding_with_pack(onehot_seqs, max_seq_len=300):
    seq_lengths = torch.tensor([min(len(seq), max_seq_len) for seq in onehot_seqs])
    padded_seqs = torch.zeros((len(onehot_seqs), max_seq_len), dtype = torch.long)

    for idx, (seq, seqlen) in enumerate(zip(onehot_seqs, seq_lengths)):
        if len(seq) > max_seq_len:
        #truncate if sequence is longer
            padded_seqs[idx, :] = torch.tensor(seq[:max_seq_len])
        else:
            #pad with zeros if sequence is shorter
            padded_seqs[idx, :seqlen] = torch.tensor(seq)

    # sort sequence es by length in descending order
    seq_lengths, perm_idx = seq_lengths.sort(0, descending =True)
    padded_seqs = padded_seqs[perm_idx]

    #pack padded sequences
    packed_seq = pack_padded_sequence(padded_seqs, seq_lengths, batch_first=True)
    #print("packed sequence:", packed_seq)


    return packed_seq


x_train_pad = padding_with_pack(X_train,300)
x_test_pad = padding_with_pack(X_test,300)

# unpacked the pad sequences
x_train_pad, train_seq_lengths = pad_packed_sequence(x_train_pad, batch_first = True)
x_test_pad, test_seq_lengths = pad_packed_sequence(x_test_pad, batch_first = True)

# convert the training and testing input data to a PyTorch tensor, casting it to long and then float
train_data = TensorDataset(x_train_pad.long().float(), torch.tensor(y_train))
valid_data = TensorDataset(x_test_pad.long().float(), torch.tensor(y_test))

batch_size = 10

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last = True)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size, drop_last = True)

#%%
# ******************
#   modeling
# ******************
# Load Word2Vec embeddings
word2vec_model = api.load("word2vec-google-news-300")
class BiLSTM(nn.Module):
    def __init__(self, num_layers, vocab_size, hidden_dim, output_dim, drop_prob=0.5):
        super(BiLSTM, self).__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.num_layers = num_layers
        self.vocab_size = vocab_size
        # initiate embedding layer with Word2Vec embeddings
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(word2vec_model.vectors))
        # LSTM layer
        self.lstm = nn.LSTM(input_size=word2vec_model.vector_size, hidden_size=self.hidden_dim, num_layers=num_layers, batch_first=True, bidirectional = True)
        # dropout layer
        self.dropout = nn.Dropout(drop_prob)
        # fully connected layer
        self.fc = nn.Linear(self.hidden_dim*2, output_dim)
        # sigmoid activation
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size(0)
        x = x.long()  # Cast input tensor to torch.long
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        out = self.dropout(lstm_out)
        out = self.fc(out)
        sig_out = self.sig(out)
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]
        return sig_out, hidden

    def init_hidden(self, batch_size):
        h0 = torch.zeros((self.no_layers, batch_size, self.hidden_dim)).to(device)
        c0 = torch.zeros((self.no_layers, batch_size, self.hidden_dim)).to(device)
        hidden = (h0, c0)
        return hidden

num_layers = 2
## added 1 for the padding token or for the unknown tokens
vocab_size = len(vocab) + 1
embedding_dim = 50
# number of classes
output_dim = 3
hidden_dim = 256
drop_prob = 0.5


##three classes of labels. Cross-entropy loss measures the difference between two probability distributions:
# the predicted probability distribution output by the model, and the true distribution of the labels.

# instantiate the model
# Instantiate the model
model = BiLSTM(num_layers, vocab_size, hidden_dim, output_dim)
model.to(device)
lr = 0.0001

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


 #calculate accuracy for classification
def acc(pred, label):
    _, predicted = torch.max(pred, 1)
    correct = (predicted == label).sum().item()
    return correct

# Training loop
clip = 5
epochs = 50

valid_loss_min = np.Inf
epoch_tr_loss, epoch_vl_loss = [], []
epoch_tr_acc, epoch_vl_acc = [], []


for epoch in range(epochs):
    train_losses = []
    train_acc = 0.0
    model.train()
    h = model.init_hidden(batch_size)
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        h = tuple([each.data for each in h])
        optimizer.zero_grad()
        output, h = model(inputs, h)
        loss = criterion(output, labels)
        loss.backward()
        train_losses.append(loss.item())
        accuracy = acc(output, labels)
        train_acc += accuracy
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

    val_h = model.init_hidden(batch_size)
    val_losses = []
    val_acc = 0.0
    model.eval()
    for inputs, labels in valid_loader:
        val_h = tuple([each.data for each in val_h])
        inputs, labels = inputs.to(device), labels.to(device)
        output, val_h = model(inputs, val_h)
        val_loss = criterion(output, labels)
        val_losses.append(val_loss.item())
        accuracy = acc(output, labels)
        val_acc += accuracy

    epoch_train_loss = np.mean(train_losses)
    epoch_val_loss = np.mean(val_losses)
    epoch_train_acc = train_acc / len(train_loader.dataset)
    epoch_val_acc = val_acc / len(valid_loader.dataset)
    epoch_tr_loss.append(epoch_train_loss)
    epoch_vl_loss.append(epoch_val_loss)
    epoch_tr_acc.append(epoch_train_acc)
    epoch_vl_acc.append(epoch_val_acc)
    print(f'Epoch {epoch + 1}')
    print(f'train_loss : {epoch_train_loss} val_loss : {epoch_val_loss}')
    print(f'train_accuracy : {epoch_train_acc * 100} val_accuracy : {epoch_val_acc * 100}')