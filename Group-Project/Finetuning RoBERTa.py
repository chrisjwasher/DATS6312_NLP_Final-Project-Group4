from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import get_scheduler
import pandas as pd
import numpy as np
import torch
import os
import json
from tqdm import trange
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from datasets import load_metric
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
#    taking the necessary columns
# ******************

articles = df['content']
labels = df['bias']

# ******************
#   tokenizing with Roberta tokenizer
# ******************
# initialize the RoBERTA tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
# Tokenize the articles
tokenized_articles = [tokenizer.encode(article, max_length=512, truncation=True) for article in articles]

# print the tokenized version of the first article
print("tokenized version of the first article:")
print(tokenized_articles[0])

#*******************
MAX_LEN = 512
batch_size = 100
epochs = 50

#****************
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_articles]
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

#to seperate the real words from padded 0s, we need to create attention masks for padded input sequences
attention_masks = []
for seq in input_ids:
    seq_mask = [float(i > 0) for i in seq]
    attention_masks.append(seq_mask)

#*********
# Splitting the data into train and validation sets
#*********
# split the sets
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels,
                                                                                    random_state=10, test_size=0.1)
# split the attention masks with the same random state, but we get rid of the input IDs, they are splitted in the above line
train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids,
                                                       random_state=10, test_size=0.1)

train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)
train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)
train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)

#************
# to avoid overfitting, we reshuffle the training data, not to let it memorize the order and to find more generalizable patterns
# this is for learning optimization

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

#**********
# setting up model and  ADAM optimizer
#**********
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=3)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.1},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters,
                  lr=2e-5,
                  eps=1e-8
                  )
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=total_steps)

#************
#  Flattening the model to one layer
#calculating the percentage of predicted labels in relation to the original labels
# Calculating training loss
# Calculating validation accuracy
#************
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


train_loss_set = []

for _ in trange(epochs, desc="Epoch"):
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        optimizer.zero_grad()
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs['loss']
        train_loss_set.append(loss.item())
        loss.backward()
        optimizer.step()
        scheduler.step()
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1
    print("Train loss: {}".format(tr_loss / nb_tr_steps))

    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for batch in validation_dataloader:
        # move the batch of tensors to the appropriate device
        batch = tuple(t.to(device) for t in batch)
        #unpacking batch into input ids, masks and labels
        b_input_ids, b_input_mask, b_labels = batch
        # stop using the gradients
        with torch.no_grad():
            # passing the input IDs and masks to get the raw output for each class
            logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        # convert the logits into numpy array by detaching from cpus
        logits = logits['logits'].detach().cpu().numpy()
        #similarly, move the label ids to numpy array
        label_ids = b_labels.to('cpu').numpy()
        # use the flat accuracy function defined above to calculate the batch accuracy
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        # update the evaluation accuracy with the accuracy of the batch
        eval_accuracy += tmp_eval_accuracy
        # then increment the number of evaluation steps/batches
        nb_eval_steps += 1
    print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))

    #************
    # Printing a plot for training loss
    #************
    plt.figure(figsize=(15, 8))
    plt.title("Training loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.plot(train_loss_set)
    plt.show()

#***********
# Evaluation on validation data loader
model.eval()
predictions, true_labels = [], []

# Predict
for batch in validation_dataloader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch
    with torch.no_grad():
        logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

    logits = logits['logits'].detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    predictions.append(logits)
    true_labels.append(label_ids)


# Generate classification report( this gives precision, recall, F1-score and support for each class)
report = classification_report(true_labels, predictions,  output_dict=True)
for class_name, metrics in classification_report.items():
    print(f"Class: {class_name}")
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall: {metrics['recall']}")
    print(f"F1-Score: {metrics['f1-score']}")
    print()

#print(report)