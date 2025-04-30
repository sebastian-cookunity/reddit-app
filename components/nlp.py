import streamlit as st
import nltk
from nltk.corpus import stopwords
import re
from .constants import NEGATIVE_WORDS


# Initialize and cache NLTK resources - do this once at startup
@st.cache_resource
def initialize_nltk_resources():
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)

    try:
        nltk.data.find("sentiment/vader_lexicon")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)

    return {"stopwords": stopwords.words("english")}


def analyze_sentiment_words(df):
    """Analyze the most common words in negative sentiment content"""
    # Initialize NLTK resources
    nltk_resources = initialize_nltk_resources()

    # Filter for negative sentiment (below 3.0)
    negative_df = df[df["sentiment_score"] < 3.0].copy()

    if negative_df.empty:
        return {"common_negative_words": [], "negative_content_sample": []}

    # Get all content from negative posts/comments
    all_negative_content = negative_df["content"].tolist()

    # Combine all content into one string and convert to lowercase
    combined_content = " ".join(
        [str(content) for content in all_negative_content]
    ).lower()

    # Get stop words from cached resource
    stop_words = set(nltk_resources["stopwords"])

    # Find instances of negative words
    negative_word_counts = {}
    for word in NEGATIVE_WORDS:
        # Use regex to find whole word matches
        count = len(re.findall(r"\b" + re.escape(word) + r"\b", combined_content))
        if count > 0:
            negative_word_counts[word] = count

    # Sort by frequency
    sorted_negative_words = sorted(
        negative_word_counts.items(), key=lambda x: x[1], reverse=True
    )

    # Get samples of negative content (limit to 5)
    negative_samples = []
    for _, row in negative_df.head(5).iterrows():
        negative_samples.append(
            {
                "type": row["type"],
                "subreddit": row["subreddit"],
                "date": row["date"]
                if isinstance(row["date"], str)
                else row["date"].strftime("%Y-%m-%d"),
                "sentiment_score": row["sentiment_score"],
                "content": row["content"][:150] + "..."
                if len(str(row["content"])) > 150
                else row["content"],
            }
        )

    return {
        "common_negative_words": sorted_negative_words[:20],  # Top 20 negative words
        "negative_content_sample": negative_samples,
    }
