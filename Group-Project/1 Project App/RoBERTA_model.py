from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_scheduler
from torch.utils.data import DataLoader
import torch
import os
import json
import pickle
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from datasets import Dataset


# Check for GPU availability
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")


#%%
# Read JSON files from folder
def read_json_folder(folder_path):
    if not folder_path or not os.path.exists(folder_path):
        print("Invalid folder path.")
        return None

    json_data_list = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    json_data = json.load(f)
                    json_data_list.append(json_data)
                except json.JSONDecodeError:
                    print(f"Error reading JSON from file: {file_path}")
                    continue

    if not json_data_list:
        print(f"No valid JSON files found in folder: {folder_path}")
        return None

    data_frame = pd.DataFrame(json_data_list)
    return data_frame

# Read JSON files and preprocess data
df = read_json_folder('Group Project/data/jsons')

# Drop unnecessary columns
df = df.drop(columns=['topic', 'source', 'url', 'title', 'date', 'authors',
                      'content_original', 'source_url', 'bias_text', 'ID'], axis=1)

num_epochs = 8

# Load data processing functions
X = df['content']
y = df['bias']

X = pd.DataFrame(X, columns=['content'])
y = pd.DataFrame(y, columns=['bias'])

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#%%
# Load tokenizer
checkpoint = "roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(checkpoint)

# Tokenize and encode data
def encode_data(data, tokenizer):
    return tokenizer(data['content'], padding='max_length', truncation=True, return_tensors='pt')

# Encode training and validation datasets
train_dataset = Dataset.from_pandas(X_train).map(lambda x: encode_data(x, tokenizer), batched=True)
val_dataset = Dataset.from_pandas(X_val).map(lambda x: encode_data(x, tokenizer), batched=True)

# Add labels column to the datasets
train_dataset = train_dataset.add_column('labels', y_train[y_train.columns[0]])
val_dataset = val_dataset.add_column('labels', y_val[y_val.columns[0]])

# Set format for training dataset
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Set format for validation dataset
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Create PyTorch dataloaders
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)
val_dataloader = DataLoader(val_dataset, batch_size=8)

# Model initialization
model = RobertaForSequenceClassification.from_pretrained(checkpoint, num_labels=3)  # Assuming 3 classes for classification

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Learning rate scheduler
num_training_steps = num_epochs * len(train_dataloader)
num_warmup_steps = 0.1 * num_training_steps  # You can adjust the percentage of warm-up steps as needed
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

# Device setup
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Training loop
progress_bar = tqdm(range(num_training_steps))
model.train()
for epoch in range(num_epochs):
    epoch_losses = []
    for batch in train_dataloader:
        batch_encoded = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
        labels = batch['labels'].to(device)

        outputs = model(**batch_encoded, labels=labels)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        epoch_losses.append(loss.item())

        progress_bar.update(1)

    # Calculate average loss for this epoch
    avg_train_loss = sum(epoch_losses) / len(epoch_losses)

    # Evaluation
    model.eval()
    val_losses = []
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_dataloader:
            batch_encoded = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)

            outputs = model(**batch_encoded, labels=labels)
            loss = outputs.loss
            val_losses.append(loss.item())

            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = sum(val_losses) / len(val_losses)
    val_acc = correct / total

    # Print epoch-wise training and validation loss, and validation accuracy
    print(f'Epoch: {epoch + 1:02} | Train Loss: {avg_train_loss:.3f} | Val Loss: {avg_val_loss:.3f} | Val Acc: {val_acc * 100:.2f}%')

    model.train()

#%%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Lists to store predictions and true labels
all_predictions = []
all_labels = []

# Evaluation loop
model.eval()
with torch.no_grad():
    for batch in val_dataloader:
        batch_encoded = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
        labels = batch['labels'].to(device)

        outputs = model(**batch_encoded, labels=labels)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate evaluation metrics
accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(all_labels, all_predictions, average='weighted')
recall = recall_score(all_labels, all_predictions, average='weighted')
f1 = f1_score(all_labels, all_predictions, average='weighted')
auc = roc_auc_score(all_labels, all_predictions, average='weighted', multi_class='ovr')
conf_matrix = confusion_matrix(all_labels, all_predictions)

# Print the evaluation metrics and confusion matrix
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
# print("AUC Score:", auc)
print("Confusion Matrix:")
print(conf_matrix)

#%%

import seaborn as sns
import matplotlib.pyplot as plt

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
# Save the plot in the working directory
plt.savefig('confusion_matrix.png')
plt.show()



#%%
# Specify the directory where you want to save the file
save_directory = "/home/ubuntu/hopgropter/Group Project/1 Project App"

# Ensure that the directory exists, create it if it doesn't
os.makedirs(save_directory, exist_ok=True)

# Save the final model after training completion
file_path = os.path.join(save_directory, "final_model_RoBERTA.pkl")
with open(file_path, 'wb') as file:
    pickle.dump(model, file)




#%%

# Save the final model after training completion
with open("final_model_RoBERTA.pkl", 'wb') as file:
    pickle.dump(model, file)

