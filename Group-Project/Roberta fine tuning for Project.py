from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
from datasets import load_metric

num_epochs = 3

## we need to read from our dataset, that's a task!!!!! for us
raw_datasets = load_dataset("glue", "mrpc")
# select only the content column for the summarization purposes
articles = raw_dataset["our_dataset_split_name"]["content"]

# Initialize the Pegasus model and tokenizer for summarization
model_name_summary = "google/pegasus-xsum"
tokenizer_summ = PegasusTokenizer.from_pretrained(model_name_summary)
model_summ = PegasusForConditionalGeneration.from_pretrained(model_name_summary)
# summarize each article and store them in a list, this is necessary for pegasus
summaries = []

# summarize the articles
for article in articles:
    inputs = tokenizer_summ(article, padding="longest", return_tensors='pt', truncation=True)
    summary_ids = model_summ.generate(inputs.input_ids, num_beams=4, min_length=30, max_length=100, length_penalty = 2.0, early_stopping=True)
    summary = tokenizer_summ.decode(summary_ids[0], skip_special_tokens=True)
    summaries.append(summary)

# Create a new dataframe with the original articles and their summaries
df = pd.DataFrame({"content": articles, "summary": summaries, })


# use RoBERTa as checkpoint - or model name for the classification task
model_name = "roberta-base"
# created tokenizer
tokenizer = RobertaTokenizer.from_pretrained(model_name)

def tokenize_function(article):
    return tokenizer(article["content"], truncation=True)

# map is a lamda function
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
# fixes the text sizes of different size
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

tokenized_datasets = tokenized_datasets.remove_columns(["topic", "source", "url", "date", "authors", "source_url", "ID", "bias_text"])
tokenized_datasets = tokenized_datasets.rename_column("bias", "labels")
tokenized_datasets.set_format("torch")

## we should split the train test set here
#tokenized_datasets["train"].column_names

# accepts the input, the IDs coming form the tokenizer
train_dataloader = DataLoader(tokenized_datasets["train"],
                              shuffle=True, batch_size=8, collate_fn=data_collator)
eval_dataloader = DataLoader(tokenized_datasets["validation"],
                             batch_size=8, collate_fn=data_collator)

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

metric = load_metric("glue", "mrpc")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    # temorary disabling gradient calculation to speed up
    with torch.no_grad():
        outputs = model(**batch)
# extracting the raw outputs from the model's outputs
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()

## we may need to look at the accuracy score, the ROC_AUC if imbalanced classes
## also I think a confusion matrix would be a nice way to see how many were classified as True Pos vs Neg etc