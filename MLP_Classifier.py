
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import random
import numpy

# -------------------------------------------------------------------------------------

corpus = []
def remove_stop_words(corpus):
    stop_words = ['is', 'a', 'will', 'be']
    results = []

    for text in corpus:
        tmp = text.split()
        for stop in stop_words:
            if stop in tmp:
                tmp.remove(stop)
        results.append(' '.join(tmp))

    return results


corpus = remove_stop_words(corpus)

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
