import time
import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

import nltk
from nltk.tokenize.casual import casual_tokenize
from nltk.stem import WordNetLemmatizer

import re



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


df, json_data_list = read_json_folder('../data/jsons')

pd.options.display.max_columns = None
print(df.head())


## Basic EDA

df.describe(include='object')


plt.figure(figsize=(10, 6))
sns.countplot(x='bias_text', data=df, palette='Set2')
plt.title('Distribution of Bias Categories')
plt.xlabel('Bias Category')
plt.ylabel('Count')
plt.xticks(rotation=45)  # Rotates the labels on the x-axis to prevent overlap
plt.show()

bias_text_counts = df['bias_text'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(bias_text_counts, labels=bias_text_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('Pastel1'))
plt.title('Pie Chart of Bias Text Distribution')
plt.show()