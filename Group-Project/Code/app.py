# ******************************************
#               Load Libraries
# ******************************************
import streamlit as st
import streamlit.components.v1 as components
import requests
from datetime import datetime
from newspaper import Article
import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import RobertaForSequenceClassification, RobertaTokenizer

import re


# ******************************************
#              Load Saved Models
# ******************************************
# Specify the file path for the saved model
file_path = './model.pth'

# Load the model
model_RoBERTA = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=3)
model_RoBERTA.load_state_dict(torch.load(file_path))
model_RoBERTA.eval()

tokenizer_RoBERTA = RobertaTokenizer.from_pretrained('roberta-base')


# ******************************************
#           Pegasus Summarization
# ******************************************
# Create a tokenizer and model
tokenizer_pegasus = PegasusTokenizer.from_pretrained('google/pegasus-multi_news')
model_pegasus = PegasusForConditionalGeneration.from_pretrained('google/pegasus-multi_news')


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
label_map_RoBERTA = {0: "Left", 1: "Center", 2: "Right"}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the model to the selected device
model_RoBERTA.to(device)

def predict(text, threshold=0.5):
    # Tokenize the input text
    inputs = tokenizer_RoBERTA(text, padding=True, truncation=True, return_tensors="pt")

    # Move the input tensors to the same device as the model
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Assuming you've defined `device`

    # Forward pass through the model
    with torch.no_grad():
        outputs = model_RoBERTA(**inputs)

    # Get the predicted probabilities
    probabilities = torch.softmax(outputs.logits, dim=-1)

    # Get the index of the class with the highest probability
    predicted_class_idx = torch.argmax(probabilities)

    # Get the probability of the predicted class
    predicted_prob = probabilities[0][predicted_class_idx]

    # Check if the predicted probability is above the threshold
    if predicted_prob.item() >= threshold:
        # Map the index to the actual label
        predicted_label = predicted_class_idx.item()

        # Get the corresponding label from the dictionary
        if predicted_label in label_map_RoBERTA:
            return label_map_RoBERTA[predicted_label], predicted_prob.item()
        else:
            return "Unknown label", predicted_prob.item()
    else:
        return "Unknown label", predicted_prob.item()



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

MLP_metrics = {
    "Accuracy": '58.47%',
    "F1-score (weighted)": 0.57,
}

CNN_metrics = {
    "Accuracy": '73%',
    "F1-score (weighted)": 0.73,
}

LSTM_metrics = {
    "Accuracy": 0.73,
    "F1-score (weighted)": 0.73,
}

model_metrics = {
    "Accuracy": round(0.9022896698615549, 3),
    "F1-score (weighted)": round(0.9023137006164763, 3)
    # Precision: 0.9024003230218862
    # Recall: 0.9022896698615549
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
    st.title("NLP Project - Political Bias Detection ")
    components.iframe("https://docs.google.com/presentation/d/e/2PACX-1vRuDIm9jOMllon851G-aXAnmgSlBFtUXLoFJq8t4koSmlvdCNjbOWkn5jreUxmGWx9ZJJNenOOpKC1r/embed?start=false&loop=false&delayms=3000",
                      height=509, width=809)

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
        st.markdown('<div style="height: 70vh; border-left: 1px solid #ccc;"></div>', unsafe_allow_html=True)

    with col3:
        st.write("*************************")
        st.subheader("Logistic Regression")
        for metric, value in logistic_metrics.items():
            st.write(f"{metric}: {value}")
        st.write("*************************")
        st.image("confusion_matrix_logistic.png")

    st.text(" ")
    st.text(" ")
    st.text(" ")
    st.text(" ")

    # **************************************************************
    # **************************************************************
    st.title("Neural Network Model Evaluation")
    col1_n, col2_n, col3_n = st.columns([1, 0.1, 1])

    with col1_n:
        st.write("*************************")
        st.subheader("MLP")
        for metric, value in MLP_metrics.items():
            st.write(f"{metric}: {value}")
        st.write("*************************")
        st.image("MLP_confusion_matrix.png")
        st.markdown(
            "<small style='color: #6e6e6e; font-size: 12px;'>**Note: 0 - Left; 1 - Center; 2 - Right.</small>",
            unsafe_allow_html=True)

    with col2_n:
        st.markdown('<div style="height: 70vh; border-left: 1px solid #ccc;"></div>', unsafe_allow_html=True)

    with col3_n:
        st.write("*************************")
        st.subheader("CNN")
        for metric, value in CNN_metrics.items():
            st.write(f"{metric}: {value}")
        st.write("*************************")
        st.image("CNN_confusion_matrix.png")

    st.text(" ")
    st.text(" ")
    st.text(" ")
    st.text(" ")

    # **************************************************************
    # **************************************************************

    st.title("LSTM & RoBERTa Model Evaluation")
    col1_m, col2_m, col3_m = st.columns([1, 0.1, 1])

    with col1_m:
        st.write("*************************")
        st.subheader("RoBERTa Model")
        for metric, value in model_metrics.items():
            st.write(f"{metric}: {value}")
        st.write("*************************")
        st.image("confusion_matrix_roberta.png")
        st.markdown(
            "<small style='color: #6e6e6e; font-size: 12px;'>**Note: 0 - Left; 1 - Center; 2 - Right.</small>",
            unsafe_allow_html=True)

    with col2_m:
        st.markdown('<div style="height: 70vh; border-left: 1px solid #ccc;"></div>', unsafe_allow_html=True)

    with col3_m:
        st.write("*************************")
        st.subheader("LSTM")
        for metric, value in LSTM_metrics.items():
            st.write(f"{metric}: {value}")
        st.write("*************************")
        st.image("CNN_confusion_matrix.png")
    # **************************************************************
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


    st.title("Lets test it out in For Real")

    initialize_session_state()

    st.image('banner 19.23.09.png')

    st.title("Search New's Articles Or Simple past url from any news source")
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
                   'pageSize=10&'
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
