import streamlit as st
import time
from newspaper import Article

def extract_article_text(url, max_retries=3, delay=1):
    retries = 0
    while retries < max_retries:
        try:
            article = Article(url)
            article.download()
            article.parse()
            return article.text
        except Exception as e:
            retries += 1
            if retries == max_retries:
                return f"Error occurred while extracting the article text: {str(e)}"
            time.sleep(delay)


def main():
    st.title("Article Text Extractor")

    # Input URL
    url = st.text_input("Enter the URL of the article:")

    if url:
        # Extract the article text
        article_text = extract_article_text(url)

        # Display the article text
        st.write("Article Text:")
        st.write(article_text)


if __name__ == '__main__':
    main()