import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords


def render_analytics(data):
    """Render the analytics tab with visualizations of Reddit data"""
    st.subheader("Reddit Data Analytics")

    # Create DataFrame from data
    df = pd.DataFrame(data)

    # Convert date strings to datetime objects
    df["date"] = pd.to_datetime(df["date"])

    # Filter out any future dates (dates greater than today)
    today = datetime.now().date()
    df = df[df["date"].dt.date <= today]

    # Add week number for WoW analysis
    df["week"] = df["date"].dt.isocalendar().week
    df["year"] = df["date"].dt.isocalendar().year

    # Create proper date for week start (first day of each week)
    df["week_start_date"] = df["date"].dt.to_period("W").dt.start_time

    # Create year-week column for sorting that shows week start date
    df["year_week"] = df["week_start_date"].dt.strftime("%Y-%m-%d")

    # Add filter options
    st.write("Filter Data:")
    col1, col2, col3 = st.columns(3)

    with col1:
        # Filter by subreddit
        subreddit_options = ["All"] + sorted(df["subreddit"].unique().tolist())
        selected_subreddit = st.selectbox(
            "Subreddit", subreddit_options, key="analytics_subreddit"
        )

    with col2:
        # Filter by type (Post or Comment)
        type_options = ["All"] + sorted(df["type"].unique().tolist())
        selected_type = st.selectbox("Content Type", type_options, key="analytics_type")

    with col3:
        # Filter by sentiment
        sentiment_options = ["All"] + sorted(df["sentiment_score"].unique().tolist())
        selected_sentiment = st.selectbox(
            "Sentiment Score", sentiment_options, key="analytics_sentiment"
        )

    # Add content search filter
    content_search = st.text_input(
        "Search in content (case-insensitive):", "", key="analytics_content_search"
    )

    # Apply filters
    filtered_df = df.copy()

    if selected_subreddit != "All":
        filtered_df = filtered_df[filtered_df["subreddit"] == selected_subreddit]

    if selected_type != "All":
        filtered_df = filtered_df[filtered_df["type"] == selected_type]

    if selected_sentiment != "All":
        filtered_df = filtered_df[filtered_df["sentiment_score"] == selected_sentiment]

    # Apply content search filter (case-insensitive)
    if content_search:
        filtered_df = filtered_df[
            filtered_df["content"].str.contains(content_search, case=False, na=False)
        ]

    # Filter selector for metrics
    metrics = st.selectbox(
        "Select Metric to Analyze", ["Post/Comment Count", "Average Sentiment"]
    )

    # Group by week for analysis
    if metrics == "Post/Comment Count":
        _render_post_comment_count_analytics(filtered_df)
    else:  # Average Sentiment
        _render_sentiment_analytics(filtered_df)

    # Add word frequency analysis and wordcloud
    _render_word_analysis(filtered_df)


def _render_post_comment_count_analytics(filtered_df):
    """Render post/comment count analytics"""
    # Count by week
    weekly_counts = (
        filtered_df.groupby(["year_week", "type"]).size().reset_index(name="count")
    )

    # Create week-over-week changes for tooltips
    pivot_counts = (
        weekly_counts.pivot(index="year_week", columns="type", values="count")
        .fillna(0)
        .reset_index()
    )

    # Calculate total counts
    pivot_counts["Total"] = pivot_counts.sum(axis=1, numeric_only=True)

    # Calculate WoW changes
    for col in pivot_counts.columns[1:]:  # Skip the year_week column
        pivot_counts[f"{col}_prev"] = pivot_counts[col].shift(1)
        pivot_counts[f"{col}_wow"] = pivot_counts[col] - pivot_counts[f"{col}_prev"]
        pivot_counts[f"{col}_wow_pct"] = (
            (pivot_counts[col] / pivot_counts[f"{col}_prev"]) - 1
        ) * 100

    # Remove the first row which will have NaN for WoW calculations
    pivot_counts_no_first = pivot_counts.dropna(
        subset=[
            f"{col}_prev"
            for col in pivot_counts.columns[1:]
            if not col.endswith("_prev")
            and not col.endswith("_wow")
            and not col.endswith("_wow_pct")
        ]
    )

    # Create week-over-week comparison
    st.subheader("Weekly Post/Comment Count")

    # Create line chart with Plotly
    fig = go.Figure()

    # Add traces for each content type
    for content_type in filtered_df["type"].unique():
        type_data = weekly_counts[weekly_counts["type"] == content_type]

        # Create hover text with WoW information
        hover_text = []
        for _, row in type_data.iterrows():
            yr_wk = row["year_week"]
            count = row["count"]

            # Find corresponding row in pivot_counts for WoW data
            if yr_wk in pivot_counts["year_week"].values:
                pivot_row = pivot_counts[pivot_counts["year_week"] == yr_wk].iloc[0]

                # Check if WoW data exists
                if f"{content_type}_wow" in pivot_row and not pd.isna(
                    pivot_row[f"{content_type}_wow"]
                ):
                    wow_change = pivot_row[f"{content_type}_wow"]
                    wow_pct = pivot_row[f"{content_type}_wow_pct"]
                    hover_text.append(
                        f"Date: {yr_wk}<br>"
                        f"Count: {count}<br>"
                        f"WoW Change: {wow_change:.0f} ({wow_pct:.1f}%)"
                    )
                else:
                    hover_text.append(
                        f"Date: {yr_wk}<br>" f"Count: {count}<br>" f"WoW Change: N/A"
                    )
            else:
                hover_text.append(
                    f"Date: {yr_wk}<br>" f"Count: {count}<br>" f"WoW Change: N/A"
                )

        fig.add_trace(
            go.Scatter(
                x=type_data["year_week"],
                y=type_data["count"],
                mode="lines+markers",
                name=content_type,
                text=hover_text,
                hoverinfo="text",
            )
        )

    # Update layout
    fig.update_layout(
        title="Weekly Post/Comment Count Over Time",
        xaxis_title="Week Start Date",
        yaxis_title="Count",
        hovermode="closest",
    )

    st.plotly_chart(fig, use_container_width=True)

    # WoW Analysis - show as a table instead
    if len(weekly_counts["year_week"].unique()) > 1:
        st.subheader("Week-over-Week Analysis")

        # Create a clean table for WoW analysis
        wow_table = pivot_counts_no_first.copy()
        display_cols = ["year_week"]

        for col in ["Post", "Comment", "Total"]:
            if col in wow_table.columns:
                display_cols.extend([col, f"{col}_wow", f"{col}_wow_pct"])

        # Select only the columns we want to display
        display_table = wow_table[display_cols].copy()

        # Rename columns for better display
        column_renames = {}
        for col in display_cols:
            if col == "year_week":
                column_renames[col] = "Week Start Date"
            elif col.endswith("_wow"):
                base_col = col.replace("_wow", "")
                column_renames[col] = f"{base_col} Change"
            elif col.endswith("_wow_pct"):
                base_col = col.replace("_wow_pct", "")
                column_renames[col] = f"{base_col} Change %"
            else:
                column_renames[col] = col

        display_table = display_table.rename(columns=column_renames)

        # Format percentage columns
        for col in display_table.columns:
            if "Change %" in col:
                display_table[col] = display_table[col].round(1).astype(str) + "%"

        st.dataframe(display_table, use_container_width=True)
    else:
        st.info("Not enough weeks of data for week-over-week analysis")


def _render_sentiment_analytics(filtered_df):
    """Render sentiment analytics"""
    # Average sentiment by week
    weekly_sentiment = (
        filtered_df.groupby(["year_week"])
        .agg({"sentiment_score": "mean"})
        .reset_index()
    )

    # Calculate WoW changes for tooltips
    weekly_sentiment["prev_score"] = weekly_sentiment["sentiment_score"].shift(1)
    weekly_sentiment["wow_change"] = (
        weekly_sentiment["sentiment_score"] - weekly_sentiment["prev_score"]
    )
    weekly_sentiment["wow_pct"] = (
        (weekly_sentiment["sentiment_score"] / weekly_sentiment["prev_score"]) - 1
    ) * 100

    # Create sentiment time series
    st.subheader("Weekly Average Sentiment")

    # Create hover text with WoW information
    hover_text = []
    for _, row in weekly_sentiment.iterrows():
        yr_wk = row["year_week"]
        score = row["sentiment_score"]

        # Check if WoW data exists
        if not pd.isna(row["wow_change"]):
            wow_change = row["wow_change"]
            wow_pct = row["wow_pct"]
            hover_text.append(
                f"Date: {yr_wk}<br>"
                f"Sentiment: {score:.2f}/5<br>"
                f"WoW Change: {wow_change:.2f} ({wow_pct:.1f}%)"
            )
        else:
            hover_text.append(
                f"Date: {yr_wk}<br>" f"Sentiment: {score:.2f}/5<br>" f"WoW Change: N/A"
            )

    # Create line chart with Plotly
    fig_sent = go.Figure()

    # Add sentiment trace
    fig_sent.add_trace(
        go.Scatter(
            x=weekly_sentiment["year_week"],
            y=weekly_sentiment["sentiment_score"],
            mode="lines+markers",
            name="Average Sentiment",
            text=hover_text,
            hoverinfo="text",
            line=dict(color="blue", width=2),
        )
    )

    # Add reference line for neutral sentiment (3.0)
    fig_sent.add_hline(
        y=3.0, line_dash="dash", line_color="gray", annotation_text="Neutral"
    )

    # Update layout
    fig_sent.update_layout(
        title="Weekly Average Sentiment Score",
        xaxis_title="Week Start Date",
        yaxis_title="Average Sentiment",
        hovermode="closest",
    )

    st.plotly_chart(fig_sent, use_container_width=True)

    # WoW Sentiment Analysis - show as a table instead
    if len(weekly_sentiment) > 1:
        st.subheader("Week-over-Week Sentiment Analysis")

        # Remove first row which has NaN for previous values
        wow_sentiment = weekly_sentiment.dropna().copy()

        if not wow_sentiment.empty:
            # Format the columns for display
            display_sentiment = wow_sentiment[
                ["year_week", "sentiment_score", "wow_change", "wow_pct"]
            ].copy()
            display_sentiment = display_sentiment.rename(
                columns={
                    "year_week": "Week Start Date",
                    "sentiment_score": "Sentiment Score",
                    "wow_change": "Change",
                    "wow_pct": "Change %",
                }
            )

            # Format values
            display_sentiment["Sentiment Score"] = display_sentiment[
                "Sentiment Score"
            ].round(2)
            display_sentiment["Change"] = display_sentiment["Change"].round(2)
            display_sentiment["Change %"] = (
                display_sentiment["Change %"].round(1).astype(str) + "%"
            )

            st.dataframe(display_sentiment, use_container_width=True)
        else:
            st.info("Not enough data for week-over-week analysis")
    else:
        st.info("Not enough weeks of data for week-over-week analysis")


def _render_word_analysis(filtered_df):
    """Render word frequency and word cloud visualizations"""
    st.subheader("Content Text Analysis")

    # Ensure stopwords are available
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")

    # Get stopwords
    stop_words = set(stopwords.words("english"))

    # Add common words that may not add value to the analysis
    additional_stopwords = {
        "cookunity",
        "cook",
        "unity",
        "food",
        "meal",
        "meals",
        "just",
        "like",
        "get",
        "got",
        "would",
        "im",
        "ive",
        "one",
        "also",
        "really",
        "much",
        "even",
        "though",
        "still",
        "back",
    }
    stop_words.update(additional_stopwords)

    # Combine all content for processing
    all_content = " ".join([str(content) for content in filtered_df["content"]]).lower()

    # Tokenize the text (split into words)
    words = re.findall(r"\b\w+\b", all_content)

    # Filter out stopwords and short words
    filtered_words = [
        word for word in words if word not in stop_words and len(word) > 2
    ]

    # Count word frequencies
    word_counts = Counter(filtered_words)

    # Display word frequency and wordcloud in side-by-side columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 10 Most Frequent Words")

        # Get top 10 words
        top_words = word_counts.most_common(10)

        if not top_words:
            st.info("No significant words found in the filtered content.")
        else:
            # Create a bar chart using Plotly
            words, counts = zip(*top_words)
            fig = px.bar(
                x=words,
                y=counts,
                labels={"x": "Words", "y": "Frequency"},
                title="Word Frequency Analysis",
            )

            # Customize the layout
            fig.update_layout(
                xaxis_title="Word",
                yaxis_title="Frequency",
                xaxis={"categoryorder": "total descending"},
            )

            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Word Cloud")

        if not word_counts:
            st.info("No significant words found in the filtered content.")
        else:
            # Generate the wordcloud
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color="white",
                max_words=100,
                contour_width=1,
                contour_color="steelblue",
            ).generate_from_frequencies(word_counts)

            # Display the wordcloud using matplotlib
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
