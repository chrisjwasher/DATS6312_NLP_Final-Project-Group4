import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler,SequentialSampler, Dataset
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel

from torchtext.data.utils import get_tokenizer


import nltk
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, f1_score, confusion_matrix
nltk.download('punkt')
nltk.download('stopwords')



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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device available for running: ")
print(device)


print("Reading data...")
df, json_data_list = read_json_folder('data/jsons')
df['full_content'] = df['title'] + ' ' + df['content']
df = df.drop(['topic', 'source', 'url', 'date', 'authors','title', 'content',
              'content_original', 'source_url', 'bias_text','ID'], axis=1)


#------------------------------------------------------------------------------------
#
# Text preprocessing and data loading
#
#------------------------------------------------------------------------------------


class TextDataset(Dataset):
    def __init__(self, dataframe, text_col, label_col, tokenizer, max_length):
        self.dataframe = dataframe
        self.text_col = text_col
        self.label_col = label_col
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        text = self.dataframe[self.text_col].iloc[index]
        label = self.dataframe[self.label_col].iloc[index]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return{
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

text_col = 'full_content'
label_col = 'bias'
max_length = 512


tokenizer = RobertaTokenizer.from_pretrained('roberta-base')


dataset = TextDataset(df, text_col, label_col, tokenizer, max_length)
train_data, temp_data = train_test_split(dataset, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

batch_size = 16
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)




#------------------------------------------------------------------------------------
#
# Define Roberta Model
#
#------------------------------------------------------------------------------------

class RoBERTaClassifier(nn.Module):
    def __init__(self, num_classes, dropout=.1):
        super(RoBERTaClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.roberta.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits



#------------------------------------------------------------------------------------
#
# Train Loop
#
#------------------------------------------------------------------------------------


num_classes = 3
model = RoBERTaClassifier(num_classes).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
num_epochs = 7

print("training loop...")
for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    for batch in train_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_dataloader)

# ------------------------------------------------------------------------------------
#
# Validation Loop
#
# ------------------------------------------------------------------------------------


    # Validation
    model.eval()
    val_loss = 0.0
    val_accuracy = 0.0
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            val_accuracy += (predicted == labels).sum().item()

    val_loss /= len(test_dataloader)
    val_accuracy /= len(test_data)

    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {val_loss:.4f}, Test Accuracy: {val_accuracy:.4f}")



model.eval()
test_true_labels = []
test_predicted_labels = []
test_loss = 0.0
test_correct = 0
test_total = 0
with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)

        test_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        test_correct += (predicted == labels).sum().item()
        test_total += labels.size(0)

        test_true_labels.extend(labels.cpu().numpy())
        test_predicted_labels.extend(predicted.cpu().numpy())



test_accuracy = test_correct / test_total
test_loss /= len(test_dataloader)

print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

test_kappa = cohen_kappa_score(test_true_labels, test_predicted_labels)
test_f1 = f1_score(test_true_labels, test_predicted_labels, average='weighted')
test_confusion_matrix = confusion_matrix(test_true_labels, test_predicted_labels)

print(f"Test Kappa: {test_kappa:.4f}")
print(f"Test F1 (weighted): {test_f1:.4f}")
print("Confusion Matrix:")
print(test_confusion_matrix)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(test_confusion_matrix, cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar()
plt.xticks(range(num_classes), range(num_classes))
plt.yticks(range(num_classes), range(num_classes))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


