import os
import json
import pandas as pd


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
#   BERT model
# ******************
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


df_BERT = pd.DataFrame(df['content'], columns=['content'])
df_BERT['bias'] = df['bias']



#%%
# Tokenization and Padding
def tokenize_text(text):
    return tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=256,  # Adjust according to your needs
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )


tokenized_texts = df['content'].apply(tokenize_text)

# Extract input_ids and attention_mask from the tokenized_texts DataFrame
input_ids = torch.cat(tokenized_texts.apply(lambda x: x['input_ids']).tolist(), dim=0)
attention_mask = torch.cat(tokenized_texts.apply(lambda x: x['attention_mask']).tolist(), dim=0)

# Convert labels to numerical representation (if not already done)
label_map = {'label1': 0, 'label2': 1, 'label3': 2}  # Define your label mapping
df['bias'] = df['bias'].map(label_map)

# Convert string labels to integer indices
label_encoder = LabelEncoder()
df['bias_encoded'] = label_encoder.fit_transform(df['bias'])

# Create DataLoader
dataset = TensorDataset(input_ids, attention_mask, torch.tensor(df['bias_encoded']))
train_dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Load model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)  # Assuming 3 classes

# Define optimizer and loss function
optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch_input_ids, batch_attention_mask, batch_labels in train_dataloader:
        optimizer.zero_grad()
        outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
        logits = outputs.logits

        # Ensure batch_labels is of type LongTensor
        batch_labels = batch_labels.long()

        loss = criterion(logits, batch_labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

# Evaluation
model.eval()
predictions = []
true_labels = []

with torch.no_grad():
    for batch_input_ids, batch_attention_mask, batch_labels in train_dataloader:
        outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
        predictions.extend(torch.argmax(outputs.logits, dim=1).tolist())
        true_labels.extend(batch_labels.tolist())

accuracy = accuracy_score(true_labels, predictions)
print(f"Accuracy: {accuracy}")


#################################################
#################################################
#################################################