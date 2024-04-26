
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from gensim.parsing.preprocessing import remove_stopwords
from gensim.utils import simple_preprocess
from gensim import corpora
import gensim.downloader as api


import torch
import torch.nn as nn
from torch.optim import AdamW
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import TensorDataset, DataLoader, RandomSampler,SequentialSampler


from datasets import Dataset, load_dataset
import evaluate

import random
import numpy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
import gensim
from sklearn.model_selection import train_test_split
nltk.download('punkt')
nltk.download('stopwords')
#from EDA_script import read_json_folder


# -------------------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device available for running: ")
print(device)

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


def token_text(text):
    no_stop = remove_stopwords(text)
    processed_text = simple_preprocess(no_stop)
    return processed_text

def build_vocabulary(token_text):
    '''
    Build vocabulary from list of tokenized texts and find maximum document length.

    Args:
        preprocessed_text: List[str]

    Returns:
        word2idx (Dict): Vocabulary built from the corpus
        max_len (int): Maximum doc length

    '''

    max_len = 0
    word2idx = {}

    # Add <pad> and <unk> tokens to the vocabulary
    word2idx['<pad>'] = 0
    word2idx['<unk>'] = 1

    idx = 2
    for doc in token_text:
        for token in doc:
            if token not in word2idx:
                word2idx[token] = idx
                idx += 1

        max_len = max(max_len, len(doc))

    return word2idx, max_len


def encode_text(tokenized_text, word2idx, max_len):
    '''
    Encode the tokens to their index in the vocabulary.

    Args:
        tokenized_text: List[List[Str]]
        word2idx: Dict
        max_len: Int

    Returns:
        input_ids (np.array): Array of token indexes in the vocabulary with
        shape (N, max_len). It will the input of our CNN model.
    '''
    input_ids = []
    for tokenized_doc in tokenized_text:
        tokenized_doc += ['<pad>'] * (max_len - len(tokenized_doc))

        input_id = [word2idx.get(token) for token in tokenized_doc]
        input_ids.append(input_id)

    return np.array(input_ids)

def load_pretrained_vectors():
    model = api.load("word2vec-google-news-300"

df['tokenized_text'] = df['full_content'].apply(token_text)

X_train,X_test,y_train,y_test = train_test_split(df['preprocessed_text'],
                                                 df['bias'],
                                                 test_size=.2,
                                                 stratify=df['bias'])



review_dict = corpora.Dictionary([['pad']])
review_dict.add_documents(X_train)


def data_loader(train_inputs, test_inputs, train_labels, test_labels, batch_size=50):
    '''
    Convert training and validation sets into torch.Tensors and load them to a DataLoader iterator.
    '''

    #convert to torch.Tensor
    train_inputs, test_inputs, train_labels, test_labels = (tuple(torch.tensor(data) for data in
               [train_inputs, test_inputs, train_labels, test_labels]))

    #Dataloader for train
    train_data = TensorDataset(train_inputs, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    #Dataloader for test
    test_data = TensorDataset(test_inputs, test_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    return train_dataloader, test_dataloader

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(100, 50)  # Input size is 100, which is the size of Word2Vec vectors
        self.layer2 = nn.Linear(50, 10)
        self.output = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.sigmoid(self.output(x))
        return x


LR = 1e-2
N_EPOCHS =100
EMBEDDING_DIM = 2


model = MLP()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

for epoch in range(N_EPOCHS):  # runs for 100 epochs
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train).reshape(-1)
    loss = criterion(outputs, y_train)

    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        predictions = model(X_test).reshape(-1)
        val_loss = criterion(predictions, y_test)

    print(f'Epoch [{epoch + 1}/{N_EPOCHS}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')


model.eval()
with torch.no_grad():
    y_pred = model(X_test).reshape(-1)
    y_pred_classes = (y_pred > 0.5).float()
    accuracy = (y_pred_classes == y_test).float().mean()
    print(f'Accuracy: {accuracy.item():.4f}')


'''
words = []
for text in corpus:
    for word in text.split(' '):
        words.append(word)
words = set(words)

word2int = {}

for i, word in enumerate(words):
    word2int[word] = i

sentences = []
for sentence in corpus:
    sentences.append(sentence.split())
print(sentences)


def get_data(sentences, WINDOW_SIZE):
    data = []

    for sentence in sentences:
        for idx, word in enumerate(sentence):
            for neighbor in sentence[max(idx - WINDOW_SIZE, 0): min(idx + WINDOW_SIZE, len(sentence)) + 1]:
                if neighbor != word:
                    data.append([word, neighbor])
    return data


data2 = get_data(sentences, WINDOW_SIZE=2)
data3 = get_data(sentences, WINDOW_SIZE=3)


df = pd.DataFrame(data2, columns = ['input', 'label'])


ONE_HOT_DIM = len(words)
def to_one_hot_encoding(data_point_index):
    one_hot_encoding = np.zeros(ONE_HOT_DIM)
    one_hot_encoding[data_point_index] = 1
    return one_hot_encoding

X = []
Y = []
for x, y in zip(df['input'], df['label']):
    X.append(to_one_hot_encoding(word2int[ x ]))
    Y.append(to_one_hot_encoding(word2int[ y ]))
X_train = np.asarray(X)
Y_train = np.asarray(Y)


LR = 1e-2
N_EPOCHS =2000
PRINT_LOSS_EVERY = 1000
EMBEDDING_DIM = 2
class MLP(nn.Module):
    def __init__(self, hidden_dim):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(12, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 12)
        self.act1 = torch.nn.Softmax(dim=1)
    def forward(self, x):
        out_em = self.linear1(x)
        output = self.linear2(out_em)
        output = self.act1(output)
        return out_em, output


p = torch.Tensor(X_train)
p.requires_grad = True
t = torch.Tensor(Y_train)


model = MLP(EMBEDDING_DIM)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

for epoch in range(N_EPOCHS):
    optimizer.zero_grad()
    _, t_pred = model(p)
    loss = criterion(t, t_pred)
    loss.backward()
    optimizer.step()
    if epoch % PRINT_LOSS_EVERY == 0:
        print("Epoch {} | Loss {:.5f}".format(epoch, loss.item()))

vectors = model.linear1._parameters['weight'].cpu().detach().numpy().transpose()
print(vectors)

'''