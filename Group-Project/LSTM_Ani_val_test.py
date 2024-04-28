import time
import os
import csv
import nltk
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

#nltk.download('wordnet')
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

df, json_data_list = read_json_folder('../data/jsons')

X,y = df['content'].values,df['bias'].values
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2, stratify=y)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, stratify=y_train)
print(f'shape of train data is {x_train.shape}')
print(f'shape of validation data is {x_val.shape}')
print(f'Shape of test data is {x_test.shape}')

#**********
#preprocessing
#*************



#nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('english')

def preprocess_string(s):
    s = re.sub(r"[^\w\s]", '', s)
    s = re.sub(r"\s+", '', s)
    return s


def tockenize(x_train, y_train, x_val, y_val, x_test, y_test):
    word_list = []
    stop_words = set(stopwords.words('english'))
    for sent in x_train:
        for word in sent.lower().split():
            word = preprocess_string(word)
            if word not in stop_words and word != '':
                word_list.append(word)

    corpus = Counter(word_list)
    corpus_ = sorted(corpus, key=corpus.get, reverse=True)[:1000]
    onehot_dict = {w: i + 1 for i, w in enumerate(corpus_)}

    final_list_train, final_list_val, final_list_test = [], [], []
    for sent in x_train:
        final_list_train.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split()
                                 if preprocess_string(word) in onehot_dict.keys()])
    for sent in x_val:
        final_list_val.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split()
                                if preprocess_string(word) in onehot_dict.keys()])
    for sent in x_test:
        final_list_test.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split()
                                if preprocess_string(word) in onehot_dict.keys()])

    encoded_train = y_train
    encoded_val = y_val
    encoded_test = y_test
    return final_list_train, encoded_train, final_list_test, encoded_test, final_list_val, encoded_val, onehot_dict



x_train,y_train,x_val, y_val, x_test,y_test,vocab = tockenize(x_train,y_train, x_val, y_val, x_test,y_test)
print("pre_process and tokenization complete")

### added pad_pack_sequence to the LSTM model
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
print("padding complete")

x_train_pad = padding_with_pack(x_train,300)
x_val_pad = padding_with_pack(x_val, 300)
x_test_pad = padding_with_pack(x_test,300)

# unpacked the pad sequences
x_train_pad, train_seq_lengths = pad_packed_sequence(x_train_pad, batch_first = True)
x_val_pad, val_seq_lengths = pad_packed_sequence(x_val_pad, batch_first = True)
x_test_pad, test_seq_lengths = pad_packed_sequence(x_test_pad, batch_first = True)

# convert the training and testing input data to a PyTorch tensor, casting it to long and then float
train_data = TensorDataset(x_train_pad.long().float(), torch.tensor(y_train, dtype=torch.long))
valid_data = TensorDataset(x_val_pad.long().float(), torch.tensor(y_val, dtype=torch.long))
test_data = TensorDataset(x_test_pad.long().float(), torch.tensor(y_test, dtype=torch.long))



batch_size=32

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last = True)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size, drop_last = True)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size, drop_last = True)

# loaded glove embeddings for Task 2
##### Task 2  #####
def load_glove_embeddings(filepath):

    embeddings_index = { }

    with open(filepath, 'r', encoding = "utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype = "float32")
            embeddings_index[word] = coefs

    return embeddings_index

glove_file_path = 'glove.6B.50d.txt'
glove_embeddings_index = load_glove_embeddings(glove_file_path)

# Initialize embedding matrix with GloVe embeddings
def initialize_embedding_matrix(embeddings_index, vocab):
    embedding_dim = len(next(iter(embeddings_index.values())))
    # add 1 to replace the unknown tokens
    vocab_size = len(vocab)+1
    # add zeros for the unknown token coefficients
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    for word, idx in vocab.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[idx] = embedding_vector

    # Convert the embedding matrix to torch tensor with data type torch.float32
    embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)
    #print("Embedding Matrix:", embedding_matrix[:2])
    return embedding_matrix

# created the embedding matrix with the indices and the words
embedding_matrix = initialize_embedding_matrix(glove_embeddings_index, vocab)

class ClassificationLSTM(nn.Module):
    def __init__(self, no_layers, vocab_size, hidden_dim, embedding_dim, drop_prob=0.5):
        super(ClassificationLSTM, self).__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.no_layers = no_layers
        self.vocab_size = vocab_size
        # initiate embedding layer with GloVe embeddings
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix), freeze = True)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=self.hidden_dim, num_layers=no_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, hidden):
        batch_size = x.size(0)
        # cast input tensor to torch.long
        x = x.long()
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        out = self.dropout(lstm_out)
        out = self.fc(out)
        # converted the output into float to address and error
        out = out.float()
        out = self.softmax(out)
        out = out.view(batch_size, -1)
        #sig_out = sig_out[:, -1]
        return out, hidden

    def init_hidden(self, batch_size):
        h0 = torch.zeros((self.no_layers, batch_size, self.hidden_dim)).to(device)
        c0 = torch.zeros((self.no_layers, batch_size, self.hidden_dim)).to(device)
        hidden = (h0, c0)
        return hidden

no_layers = 2
vocab_size = len(vocab) +1
embedding_dim = 50
output_dim = 3
hidden_dim = 256


model = ClassificationLSTM(no_layers,vocab_size,hidden_dim,embedding_dim,output_dim)
model.to(device)
lr=2e-5
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#def acc(pred,label):
   # pred = torch.round(pred.squeeze())
   # return torch.sum(pred == label.squeeze()).item()

def acc(pred, label):
    _, pred_classes = torch.max(pred, 1)  # Get the predicted classes

    correct = (pred_classes == label).sum().item()  # Count the number of correct predictions
    total = label.size(0)  # Get the total number of samples
    accuracy = correct / total  # Calculate accuracy
    return accuracy

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
        model.zero_grad()
        output, h = model(inputs, h)
        labels = labels.long()
        loss = criterion(output.squeeze(), labels)
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
        val_loss = criterion(output.squeeze(), labels.long())
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


# Test the model on the test set
test_losses = []
test_acc = 0.0
model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        output, _ = model(inputs, h)
        labels = labels.long()
        test_loss = criterion(output.squeeze(), labels)
        test_losses.append(test_loss.item())
        accuracy = acc(output, labels)
        test_acc += accuracy

test_loss = np.mean(test_losses)
test_acc = test_acc / len(test_loader.dataset)
print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc * 100:.2f}%')