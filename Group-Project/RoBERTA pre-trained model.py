from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import AdamW
from transformers import get_scheduler
import torch
import numpy as np

from tqdm.auto import tqdm
from torch.utils.data import TensorDataset, DataLoader
from datasets import load_metric
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import re
#*****
# Setting the device
#************

is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

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

# ******************
#    taking the necessary columns of the main dataset
# ******************

articles = df['content']
labels = df['bias']

# ******************
#   tokenizing with Roberta tokenizer
# ******************
# initialize the RoBERTA tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
# fixes the text sizes of different size
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
# Tokenize the articles
tokenized_articles = [tokenizer.encode(article, max_length=512, truncation=True) for article in articles]

# print the tokenized version of the first article
print("tokenized version of the first article:")
print(tokenized_articles[0])

#**********
#Model name
#********

model_name = "roberta-base"

#*********
# Splitting the data into train and validation sets
#*********
# split the sets
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(tokenized_articles, labels,
                                                                                    random_state=10, test_size=0.1)

# Convert data splits to PyTorch tensors
train_inputs_tensor = torch.tensor(train_inputs)
validation_inputs_tensor = torch.tensor(validation_inputs)
train_labels_tensor = torch.tensor(train_labels)
validation_labels_tensor = torch.tensor(validation_labels)

# Combine input data and labels into TensorDataset
train_dataset = TensorDataset(train_inputs_tensor, train_labels_tensor)
validation_dataset = TensorDataset(validation_inputs_tensor, validation_labels_tensor)

# Define batch size and number of epochs
#*******************
MAX_LEN = 512
batch_size = 100
num_epochs = 50

# Create DataLoaders for training and validation
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, collate_fn=data_collator)

for batch in train_dataloader:
    break
{k: v.shape for k, v in batch.items()}


model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=3)
outputs = model(**batch)
print(outputs.loss, outputs.logits.shape)
optimizer = AdamW(model.parameters(), lr=5e-5)

num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler("linear", optimizer=optimizer,
                              num_warmup_steps=0, num_training_steps=num_training_steps)

# to have a look the expected duration of learning, do print
print(num_training_steps)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)


progress_bar = tqdm(range(num_training_steps))

## pass the data, create a loss, then use the backward
model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

#************
# Evaluating the model performance metrics
#************
model.eval()  # Set the model to evaluation mode
# List to store predicted labels
prediction_labels = []
# List to store true labels
true_labels = []

# Iterate over batches in the validation dataloader
for batch in validation_dataloader:
    # Move batch to appropriate device (GPU if available, else CPU)
    batch = {k: v.to(device) for k, v in batch.items()}

    # Perform forward pass (prediction) without gradient calculation
    with torch.no_grad():
        outputs = model(**batch)

    # Extract predicted labels and true labels
    logits = outputs.logits
    predicted_labels = torch.argmax(logits, dim=-1).cpu().numpy()
    true_labels.extend(batch["labels"].cpu().numpy())
    prediction_labels.extend(predicted_labels)

# Convert lists to numpy arrays
true_labels = np.array(true_labels)
prediction_labels = np.array(prediction_labels)

# Generate classification report( this gives precision, recall, F1-score and support for each class)
report = classification_report(true_labels, prediction_labels, output_dict=True)
for class_name, metrics in classification_report.items():
    print(f"Class: {class_name}")
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall: {metrics['recall']}")
    print(f"F1-Score: {metrics['f1-score']}")
    print()

#print(report)