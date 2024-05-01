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



## we may need to look at the accuracy score, the ROC_AUC if imbalanced classes
## also I think a confusion matrix would be a nice way to see how many were classified as True Pos vs Neg etc