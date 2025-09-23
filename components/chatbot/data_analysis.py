import streamlit as st
import pandas as pd
import re
from datetime import datetime
from ..nlp import analyze_sentiment_words


def init_df(data, negative_words, nltk_resources):
    """Initialize and analyze the DataFrame for the chatbot"""
    st.session_state.chatbot_df = pd.DataFrame(data)

    # Dynamically determine the schema - no hardcoding!
    df = st.session_state.chatbot_df

    # Create schema information dynamically based on the actual data
    schema_info, sample_counts = create_schema_info(df)

    # Store the schema in session state
    st.session_state.dataset_schema = {
        "columns": df.columns.tolist(),
        "descriptions": schema_info,
        "sample_counts": sample_counts,
    }

    # Pre-compute sentiment analysis for chat context
    if "sentiment_analysis" not in st.session_state:
        st.session_state.sentiment_analysis = analyze_sentiment_words(
            st.session_state.chatbot_df
        )

    # Pre-compute statistics by week for WoW analysis
    if "weekly_data" not in st.session_state:
        compute_weekly_data(df, negative_words)


def create_schema_info(df):
    """Create schema information dynamically based on the data"""
    schema_info = {}
    sample_counts = {}

    # For each column, analyze its content and infer its purpose
    for col in df.columns:
        col_lower = col.lower()

        # Count unique values (up to 50 for efficiency)
        unique_vals = df[col].head(50).unique()
        sample_counts[col] = len(unique_vals)

        # Basic descriptions based on column name and content
        if col_lower == "type":
            schema_info[
                col
            ] = f"The type of Reddit content: {', '.join(df[col].unique().tolist())}"
        elif col_lower == "subreddit":
            schema_info[
                col
            ] = f"The subreddit where the content was posted: {', '.join(df[col].unique().tolist())}"
        elif col_lower == "thread_name":
            schema_info[
                col
            ] = "The title of the original post that the content belongs to"
        elif col_lower == "date":
            min_date = (
                pd.to_datetime(df[col]).min()
                if not pd.api.types.is_datetime64_any_dtype(df[col])
                else df[col].min()
            )
            max_date = (
                pd.to_datetime(df[col]).max()
                if not pd.api.types.is_datetime64_any_dtype(df[col])
                else df[col].max()
            )
            schema_info[
                col
            ] = f"The date when the content was posted (range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')})"
        elif "content" in col_lower:
            schema_info[col] = "The actual text content of the post or comment"
        elif "link" in col_lower or "url" in col_lower:
            schema_info[col] = "The direct URL to the post or comment on Reddit"
        elif "sentiment" in col_lower:
            min_val = df[col].min()
            max_val = df[col].max()
            schema_info[
                col
            ] = f"A sentiment score from {min_val} to {max_val} (lower = negative, higher = positive)"
        elif "matching" in col_lower:
            schema_info[
                col
            ] = "The specific CookUnity-related term that matched in the content"
        elif "page" in col_lower:
            schema_info[col] = "Internal pagination marker used during data collection"
        else:
            # Generic fallback for unknown columns
            schema_info[
                col
            ] = f"Column containing {col.replace('_', ' ').lower()} information"

    return schema_info, sample_counts


def compute_weekly_data(df, negative_words):
    """Compute weekly statistics for analysis"""
    df_copy = df.copy()

    # Convert date strings to datetime objects
    df_copy["date"] = pd.to_datetime(df_copy["date"])

    # Filter out future dates
    today = datetime.now().date()
    df_copy = df_copy[df_copy["date"].dt.date <= today]

    # Add week number for WoW analysis
    df_copy["week"] = df_copy["date"].dt.isocalendar().week
    df_copy["year"] = df_copy["date"].dt.isocalendar().year

    # Create proper date for week start
    df_copy["week_start_date"] = df_copy["date"].dt.to_period("W").dt.start_time

    # Create year-week column
    df_copy["year_week"] = df_copy["week_start_date"].dt.strftime("%Y-%m-%d")

    # Store in session state
    st.session_state.weekly_data = {
        "df": df_copy,
        # Count by week and type
        "counts_by_week": df_copy.groupby(["year_week", "type"])
        .size()
        .reset_index(name="count"),
        # Average sentiment by week
        "sentiment_by_week": df_copy.groupby("year_week")
        .agg({"sentiment_score": "mean"})
        .reset_index(),
    }

    # Calculate word frequencies by week
    word_counts_by_week = {}

    for week in df_copy["year_week"].unique():
        week_df = df_copy[df_copy["year_week"] == week]
        week_content = " ".join(
            [str(content) for content in week_df["content"]]
        ).lower()

        week_word_counts = {}
        for word in negative_words:
            count = len(re.findall(r"\b" + re.escape(word) + r"\b", week_content))
            if count > 0:
                week_word_counts[word] = count

        word_counts_by_week[week] = week_word_counts

    st.session_state.word_counts_by_week = word_counts_by_week
