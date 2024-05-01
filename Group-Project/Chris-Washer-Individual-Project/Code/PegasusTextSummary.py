import pandas as pd
import numpy as np
import torch
import os
import json

import torch
import torch.nn as nn

import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler,SequentialSampler
import torch.nn.functional as F

from transformers import PegasusTokenizer, PegasusForConditionalGeneration, DataCollatorForLanguageModeling, Trainer, TrainingArguments


#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
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


print("Reading data...")
df, json_data_list = read_json_folder('data/jsons')
df['full_content'] = df['title'] + ' ' + df['content']
df = df.drop(['topic', 'source', 'url', 'date', 'authors','title', 'content',
              'content_original', 'source_url', 'bias_text','ID'], axis=1)
df = df.loc[df['bias'] == 1]


class PegasusDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]).to(device) for key, val in self.encodings.items()}
        #item['decoder_input_ids'] = item['input_ids']
        return item
    def __len__(self):
        return len(self.encodings['input_ids'])


tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-large')
train_articles = df['full_content'].tolist()

train_encodings = tokenizer(train_articles, truncation=True, padding=True, max_length=512)
train_dataset = PegasusDataset(train_encodings)

model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-large').to(device)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    fp16=True,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

trainer.train()

output_dir = "./fine_tuned_model"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

input_text = "Mr. Obama should draw the circle of inclusion as large as possible \u2014 up to the eight million or so who might have qualified under an ambitious bipartisan bill that passed the Senate last year . But Mr. Obama , who wants to bolster his actions against legal attack , seems unlikely to include parents whose children lack citizenship or green cards . Tens of thousands of families will surely be disheartened by this exclusion and other politically motivated shortcomings \u2014 the plan is expected to bar recipients from health coverage under the Affordable Care Act , for example . Some immigrant advocacy groups have already denounced the plan as too cautious and too small .\nThe backlash on the right , too , is well underway , with Republican lawmakers condemning what they see as a tyrannical usurpation of congressional authority by \u201c Emperor \u201d Obama . They fail to mention , though , that new priorities will put the vast deportation machinery to better use against serious criminals , terrorists and security threats , which should be the goal of any sane law-enforcement regime . Nor did they ever complain when Mr. Obama aggressively used his executive authority to ramp up deportations to an unprecedented peak of 400,000 a year .\nIt has been the immigration system \u2019 s retreat from sanity , of course , that made Mr. Obama \u2019 s new plan necessary . Years were wasted , and countless families broken , while Mr. Obama clung to a futile strategy of luring Republicans toward a legislative deal . He has been his own worst enemy \u2014 over the years he stressed his executive impotence , telling advocates that he could not change the system on his own . This may have suited his legislative strategy , but it was not true .\nIt \u2019 s good that Mr. Obama has finally turned the page . He plans to lead a rally in Las Vegas on Friday at a high school where he outlined his immigration agenda in January 2013 . Legislative solutions are a dim hope for some future day when the Republican fever breaks . But until then , here we are .\nThis initiative can not be allowed to fail for lack of support from those who accept the need for progress on immigration , however incremental . Courageous immigrant advocates , led by day laborers , Dreamers and others , have pressed a reluctant president to acknowledge the urgency of their cause \u2014 and to do something about it . The only proper motion now is forward ."
inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=100, early_stopping=True)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
