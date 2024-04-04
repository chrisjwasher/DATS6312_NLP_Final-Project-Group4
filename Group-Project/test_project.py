import time
import os
import csv
import numpy as np
import pandas as pd
import json


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score, f1_score


import nltk
from nltk.tokenize.casual import casual_tokenize
from nltk.stem import WordNetLemmatizer


import re


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
#   Preprocessing
# ******************

nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('english')

def preprocess(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r"[^\w\s]", ' ', text)
    # Remove stopwords
    word_tokens = casual_tokenize(text)
    filtered_text = [w for w in word_tokens if w not in stop_words]
    text = ' '.join(filtered_text)
    # Lemmatize text
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(w) for w in casual_tokenize(text)]
    return ' '.join(lemmatized)


df['content_clean'] = df['content'].apply(preprocess)
#%%

def split(df):
    X = df.drop(columns=['topic', 'source', 'url',
                         'title', 'date', 'authors', 'content_original',
                         'source_url', 'bias_text', 'ID', 'bias'])
    y = df['bias']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=False)
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = split(df)
#%%

def model_traning(X_train, y_train):
    vectorizer = TfidfVectorizer(tokenizer=casual_tokenize)
    X_train = vectorizer.fit_transform(X_train)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model, vectorizer


model, vectorizer = model_traning(X_train['content_clean'], y_train)
#%%

def model_testing(model, vectorizer, X_test):
    X_test = vectorizer.transform(X_test)
    predictions = model.predict(X_test)
    return predictions


predictions = model_testing(model, vectorizer, X_test['content_clean'])
#%%

def metrics(predictions, y_test):
    kappa = cohen_kappa_score(y_test, predictions)
    return kappa


kappa = metrics(predictions, y_test)

# Accuracy of the results
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# F1-score with 'weighted' averaging
f1_weighted = f1_score(y_test, predictions, average='weighted')
print("F1-score (weighted):", f1_weighted)

# F1-score with 'micro' averaging
f1_micro = f1_score(y_test, predictions, average='micro')
print("F1-score (micro):", f1_micro)

# F1-score with 'macro' averaging
f1_macro = f1_score(y_test, predictions, average='macro')
print("F1-score (macro):", f1_macro)
