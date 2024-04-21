import streamlit as st
import requests
from datetime import datetime
from transformers import PegasusForConditionalGeneration, PegasusTokenizer


# Create a tokenizer and model
tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-xsum')
model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-xsum')


# Initialize session state
def initialize_session_state():
    if 'query_enter' not in st.session_state:
        st.session_state.query_enter = ""
    if 'publication_date' not in st.session_state:
        st.session_state.publication_date = datetime.now().strftime('%Y-%m')


# Function to generate summary using Pegasus
@st.cache_data
def summary_pegasus(content):
    tokens = tokenizer(content, return_tensors='pt', padding='longest', truncation=True)
    summary = model.generate(**tokens)
    output_string = tokenizer.decode(summary[0])
    # Remove <pad> token
    output_string = output_string.replace("<pad>", "")
    # Remove </s> token
    output_string = output_string.replace("</s>", "")
    return output_string


def formated_date(published_at):
    formatted_date = datetime.strptime(published_at, '%Y-%m-%dT%H:%M:%SZ').strftime('%Y-%m-%d')
    return formatted_date


# Main Streamlit application
def main():
    initialize_session_state()

    st.title("News Chatbot")

    # Get user query for news articles
    st.session_state.query_enter = st.text_input("Search relevant articles", st.session_state.query_enter)

    # Get publication date filter from user (Year and Month only)
    st.session_state.publication_date = st.text_input("Enter publication date (YYYY-MM-DD)",
                                                      st.session_state.publication_date)

    # Trigger "Get News" action when Enter is pressed or button is clicked
    if st.session_state.query_enter or st.button("Get News"):
        st.session_state.query = st.session_state.query_enter

        # Construct the URL with query parameters including date filter
        url = ('https://newsapi.org/v2/everything?'
               f'q={st.session_state.query_enter}&'
               f'from={st.session_state.publication_date}&'
               f'to={st.session_state.publication_date}&'
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
                st.write(f"Total results: {total_results}")
                st.write("Select an article to view more details:")
                for index, article in enumerate(articles):
                    st.write(f"{index+1}. {article['title']} - {formated_date(article['publishedAt'])}")
                    if st.button(f"View full content of article {index + 1}"):
                        st.write(f"**Title:** {article['title']}")
                        st.write(f"**Source:** {article['source']['name']}")
                        st.write(f"**Published At:** {article['publishedAt']}")
                        st.write(f"**Content:** {article['content']}")
                        summary = summary_pegasus(article['content'])
                        st.write(f"**Summary:** {summary}")
            else:
                st.warning("No articles found for the given query.")
        else:
            # Print an error message if the request failed
            st.error("Failed to fetch news articles. Please check your API key and try again.")


if __name__ == "__main__":
    main()
