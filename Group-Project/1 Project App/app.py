import streamlit as st
import requests
import pickle
from datetime import datetime
import matplotlib.pyplot as plt


from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score

from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import RobertaForSequenceClassification, RobertaTokenizer

import nltk
from nltk.tokenize.casual import casual_tokenize
from nltk.stem import WordNetLemmatizer

import re

# ******************************************
#
# ******************************************
def calculate_metrics(predictions, y_test):
    kappa = cohen_kappa_score(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)
    f1_weighted = f1_score(y_test, predictions, average='weighted')
    f1_micro = f1_score(y_test, predictions, average='micro')
    f1_macro = f1_score(y_test, predictions, average='macro')
    return kappa, accuracy, f1_weighted, f1_micro, f1_macro




# ******************************************
#   Classical Models (Naive & Logistic)
# ******************************************
# Load logistic regression model and vectorizer
with open('logistic_regression_model.pkl', 'rb') as file:
    loaded_model_lr = pickle.load(file)

with open('tfidf_vectorizer.pkl', 'rb') as file:
    loaded_vectorizer_lr = pickle.load(file)

# Load naive Bayes model and vectorizer
with open('naive_bayes_model.pkl', 'rb') as file:
    loaded_model_naive = pickle.load(file)

with open('tfidf_vectorizer_naive.pkl', 'rb') as file:
    loaded_vectorizer_naive = pickle.load(file)


# Map numerical labels to categories
label_map = {0: "left", 1: "center", 2: "right"}

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


def predict_lr(text):
    preprocessed_text = preprocess(text)
    vectorized_text = loaded_vectorizer_lr.transform([preprocessed_text])
    prediction = loaded_model_lr.predict(vectorized_text)
    return label_map[prediction[0]]


def predict_naive(text):
    preprocessed_text = preprocess(text)
    vectorized_text = loaded_vectorizer_naive.transform([preprocessed_text])
    prediction = loaded_model_naive.predict(vectorized_text)
    return label_map[prediction[0]]


# ******************************************
#              Summarization
# ******************************************
# Create a tokenizer and model
tokenizer_pegasus = PegasusTokenizer.from_pretrained('google/pegasus-xsum')
model_pegasus = PegasusForConditionalGeneration.from_pretrained('google/pegasus-xsum')


# Function to generate summary using Pegasus
@st.cache_data
def summary_pegasus_Ani_version(content):
    inputs = tokenizer_pegasus(content, padding="longest", return_tensors='pt', truncation=True)
    summary_ids = model_pegasus.generate(inputs.input_ids, num_beams=4, min_length=30, max_length=100,
                                          length_penalty=2.0, early_stopping=True)

    summary = tokenizer_pegasus.decode(summary_ids[0], skip_special_tokens=True)
    return summary


# ******************************************
#              Main App function
# ******************************************
# Initialize session state
def initialize_session_state():
    if 'query_enter' not in st.session_state:
        st.session_state.query_enter = ""
    # if 'publication_date' not in st.session_state:
        # st.session_state.publication_date = datetime.now().strftime('%Y-%m')


# Formate date from %Y-%m-%dT%H:%M:%SZ to %Y-%m-%d
def formated_date(published_at):
    formatted_date = datetime.strptime(published_at, '%Y-%m-%dT%H:%M:%SZ').strftime('%Y-%m-%d')
    return formatted_date


logistic_metrics = {
    "Kappa score": round(0.5610043206032206, 3),
    "Accuracy": round(0.7097590201038477, 3),
    "F1-score (weighted)": round(0.7095347697062324, 3),
    "F1-score (micro)": round(0.7097590201038477, 3),
    "F1-score (macro)": round(0.708539233891807, 3)
}

naive_metrics = {
    "Kappa score": round(0.28807474561318536, 3),
    "Accuracy": round(0.5410730927972307, 3),
    "F1-score (weighted)": round(0.5040879849427269, 3),
    "F1-score (micro)": round(0.5410730927972307, 3),
    "F1-score (macro)": round(0.48572983916007545, 3)
}


# Main Streamlit application
def main():
    st.title("Our path")



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
        st.markdown('<div style="height: 95vh; border-left: 1px solid #ccc;"></div>', unsafe_allow_html=True)

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
    # st.markdown("""<hr style="height:2px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

    # Get user query for news articles
    st.session_state.query_enter = st.text_input("Search relevant articles based on key words", st.session_state.query_enter)

    # Get publication date filter from user (Year and Month only)
    # st.session_state.publication_date = st.text_input("Enter publication date (YYYY-MM-DD)",
                                                      # st.session_state.publication_date)

    # Trigger "Get News" action when Enter is pressed or button is clicked
    if st.session_state.query_enter or st.button("Get Article"):
        st.session_state.query = st.session_state.query_enter

        # Construct the URL with query parameters including date filter
        url = ('https://newsapi.org/v2/everything?'
               f'q={st.session_state.query_enter}&'
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
            total_results = data['totalResults']
            articles = data['articles']
            if articles:
                st.success("News articles fetched successfully!")
                # st.write(f"Total results: {total_results}")
                st.write("Select an article to view more details:")
                for index, article in enumerate(articles):
                    st.write(f"{index+1}. {article['title']} - {formated_date(article['publishedAt'])}")
                    if st.button(f"View full content of article {index + 1}"):
                        st.write(f"**Source:** {article['source']['name']}")
                        prediction_lr = predict_lr(article['content'])
                        st.write("Predicted class (Logistic Regression):", prediction_lr)
                        prediction_naive = predict_naive(article['content'])
                        st.write("Predicted class (Naive Bayes):", prediction_naive)
                        st.write(f"**Content:** {article['content']}")
                        summary_ani = summary_pegasus_Ani_version(article['content'])
                        st.write(f"**Summary Ani:** {summary_ani}")
                        st.write(f"**Url:** {article['url']}")
            else:
                st.warning("No articles found for the given query.")
        else:
            # Print an error message if the request failed
            st.error("Failed to fetch news articles. Please check your API key and try again.")


if __name__ == "__main__":
    main()
