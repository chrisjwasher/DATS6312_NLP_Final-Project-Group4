# ******************************************
#               Load Libraries
# ******************************************
import streamlit as st
import requests
import pickle
from datetime import datetime
from newspaper import Article
import validators
import torch
import os
from sklearn.metrics import accuracy_score, f1_score
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import RobertaTokenizer
import re


# ******************************************
#              Load Saved Models
# ******************************************
# Set Hugging Face token
os.environ["HUGGING_FACE_HUB_TOKEN"] = "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"

# Load the saved RoBERTa model
# with open("/home/ubuntu/hopgropter/Group Project/1 Project App/final_model_RoBERTA.pkl", 'rb') as file:
    # model_RoBERTA = pickle.load(file)


# ******************************************
#   Classical Models (Naive & Logistic)
# ******************************************
def calculate_metrics(predictions, y_test):
    accuracy = accuracy_score(y_test, predictions)
    f1_weighted = f1_score(y_test, predictions, average='weighted')
    f1_micro = f1_score(y_test, predictions, average='micro')
    f1_macro = f1_score(y_test, predictions, average='macro')
    return accuracy, f1_weighted, f1_micro, f1_macro


# ******************************************
#           Pegasus Summarization
# ******************************************
# Create a tokenizer and model
tokenizer_pegasus = PegasusTokenizer.from_pretrained('google/pegasus-multi_news')
model_pegasus = PegasusForConditionalGeneration.from_pretrained('google/pegasus-multi_news')

# Load the tokenizer and model
tokenizer_RoBERTA = RobertaTokenizer.from_pretrained("roberta-base")
# model_RoBERTA = RobertaForSequenceClassification.from_pretrained("final_model_RoBERTA.pkl")


# Function to generate summary using Pegasus
@st.cache_data
def summary_pegasus(content):
    inputs = tokenizer_pegasus(content, padding="longest", return_tensors='pt', truncation=True)
    summary_ids = model_pegasus.generate(inputs.input_ids, num_beams=4, min_length=150, max_length=266,
                                          length_penalty=2.0, early_stopping=True)

    summary = tokenizer_pegasus.decode(summary_ids[0], skip_special_tokens=True)
    return summary


# ******************************************
#              RoBERTA model
# ******************************************
# Define label mapping
label_map_RoBERTA = {'LABEL_0': "left", 'LABEL_1': "center", 'LABEL_2': "right"}


def predict(text):
    # Preprocess the input text
    preprocessed_text = preprocess(text)

    # Tokenize the input text
    inputs = tokenizer_RoBERTA(preprocessed_text, padding=True, truncation=True, return_tensors="pt")

    # Move the input tensors to the same device as the model
    inputs = {key: value.to(model_RoBERTA.device) for key, value in inputs.items()}

    # Forward pass through the model
    with torch.no_grad():
        outputs = model_RoBERTA(**inputs)

    # Get the predicted class (index with highest probability)
    predicted_class_idx = torch.argmax(outputs.logits)

    # Map the index to the actual label
    predicted_label = model_RoBERTA.config.id2label[predicted_class_idx.item()]

    # Get the corresponding label from the dictionary
    if predicted_label in label_map_RoBERTA:
        return label_map_RoBERTA[predicted_label]
    else:
        return "Unknown label"


# ******************************************
#    initialization stage for App function
# ******************************************
# Initialize session states
def initialize_session_state():
    if 'query_enter' not in st.session_state:
        st.session_state.query_enter = ""

    if 'show_full_content' not in st.session_state:
        st.session_state.show_full_content = False

    if "url_input" not in st.session_state:
        st.session_state.url_input = ""


# Formate date from %Y-%m-%dT%H:%M:%SZ to %Y-%m-%d
def formated_date(published_at):
    formatted_date = datetime.strptime(published_at, '%Y-%m-%dT%H:%M:%SZ').strftime('%Y-%m-%d')
    return formatted_date


# These results where found in files 'Classical Models.py' please refer to it
logistic_metrics = {
    "Accuracy": round(0.7097590201038477, 3),
    "F1-score (weighted)": round(0.7095347697062324, 3),
}

naive_metrics = {
    "Accuracy": round(0.5410730927972307, 3),
    "F1-score (weighted)": round(0.5040879849427269, 3),
}


def is_url(url):
    url_pattern = r'^https?://'
    return bool(re.match(url_pattern, url))


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ******************************************
#         Main Streamlit application
# ******************************************
def main():
    st.title("Our path")
    st.write("We used pre-labeled data to train RoBERTA transformer model from .... containing 37554 Articles from "
             "different News sources. Our research was in conducting a classification task  ")

    st.title("Classical Model Evaluation")
    col1, col2, col3 = st.columns([1, 0.1, 1])

    with col1:
        st.write("*************************")
        st.subheader("Naive Regression")
        for metric, value in naive_metrics.items():
            st.write(f"{metric}: {value}")
        st.write("*************************")
        st.image("confusion_matrix_naive.png")
        st.markdown(
            "<small style='color: #6e6e6e; font-size: 12px;'>**Note: 0 - Left; 1 - Center; 2 - Right.</small>",
            unsafe_allow_html=True)

    with col2:
        st.markdown('<div style="height: 90vh; border-left: 1px solid #ccc;"></div>', unsafe_allow_html=True)

    with col3:
        st.write("*************************")
        st.subheader("Logistic Regression")
        for metric, value in logistic_metrics.items():
            st.write(f"{metric}: {value}")
        st.write("*************************")
        st.image("confusion_matrix_logistic.png")

    # **************************************************************
    st.text(" ")
    st.text(" ")
    st.text(" ")
    st.text(" ")
    st.text(" ")
    st.text(" ")
    st.text(" ")
    st.text(" ")
    st.text(" ")
    st.text(" ")
    st.text(" ")
    st.text(" ")
    st.text(" ")
    st.text(" ")
    st.text(" ")
    st.text(" ")
    st.text(" ")
    st.text(" ")
    st.text(" ")
    st.text(" ")
    st.text(" ")
    # **************************************************************


    st.title("Lets explore it on new Unseen data")

    initialize_session_state()

    st.image('new banner.png')

    st.title("Search New's Articles and Get Informed")
    st.subheader("You will get most relevant and new articles")

    st.text(" ")
    st.text(" ")
    st.text(" ")
    # st.markdown("""<hr style="height:2px;border:none;color:#333;
    # background-color:#333;" /> """, unsafe_allow_html=True)

    # Get user query for news articles
    user_query = st.text_input("Search for news articles or paste the URL", st.session_state.query_enter)

    if user_query:
        if is_url(user_query):
            # If the user input is a valid URL, fetch and process the article
            article_full = Article(user_query)
            article_full.download()
            article_full.parse()
            MAX_PREVIEW_LENGTH = 400

            if len(article_full.text) > MAX_PREVIEW_LENGTH:
                truncated_text = article_full.text[:MAX_PREVIEW_LENGTH] + "..."
                with st.expander("Preview", expanded=True):
                    st.write(truncated_text)
                with st.expander("Full Content"):
                    st.write(article_full.text)
            else:
                st.write(f"**Full content:** {article_full.text}")
            # **************************************
            summary = summary_pegasus(article_full.text)
            st.write(f"**Summary:** {summary}")
            # Make prediction using the model
            prediction = predict(article_full.text)
            # Display the prediction to the user
            st.write("Predicted class (Roberta):", prediction)
            # **************************************

        else:
            # If the user input is not a valid URL, search for articles using the News API
            st.session_state.query = user_query

            # Construct the URL with query parameters including date filter
            url = ('https://newsapi.org/v2/everything?'
                   f'q={user_query}&'
                   'language=en&'
                   'sortBy=relevancy&'
                   'pageSize=5&'
                   f'apiKey=a91f440fdad74f36a8695761264b3e4c')

            # Make a GET request to the News API
            response = requests.get(url)

            # Check if the request was successful (status code 200)
            if response.status_code == 200:
                # Extract and display the titles of the articles
                data = response.json()
                # total_results = data['totalResults']  <--- Uncomment this
                articles = data['articles']
                if articles:
                    st.success("News articles fetched successfully!")
                    # st.write(f"Total results: {total_results}")   <--- To get all the possible results
                    st.write("Select an article to view more details:")
                    for index, article in enumerate(articles):
                        st.write(f"{index + 1}. {article['title']} - {formated_date(article['publishedAt'])}")
                        if st.button(f"View full content of article {index + 1}"):
                            st.write(f"**Source:** {article['source']['name']}")
                            st.write(f"**Url:** {article['url']}")
                            # **************************************
                            article_full = Article(article['url'])
                            article_full.download()
                            article_full.parse()
                            MAX_PREVIEW_LENGTH = 400

                            if len(article_full.text) > MAX_PREVIEW_LENGTH:
                                truncated_text = article_full.text[:MAX_PREVIEW_LENGTH] + "..."
                                with st.expander("Preview", expanded=True):
                                    st.write(truncated_text)
                                with st.expander("Full Content"):
                                    st.write(article_full.text)
                            else:
                                st.write(f"**Full content:** {article_full.text}")
                            # **************************************
                            summary = summary_pegasus(article_full.text)
                            st.write(f"**Summary:** {summary}")
                            # **************************************
                            # Make prediction using the model
                            prediction = predict(article_full.text)
                            # Display the prediction to the user
                            st.write("Predicted class (Roberta):", prediction)
                            # **************************************
                else:
                    st.warning("No articles found for the given query.")
            else:
                # Print an error message if the request failed
                st.error("Failed to fetch news articles. Please check your API key and try again.")


if __name__ == "__main__":
    main()
