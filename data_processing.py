import streamlit as st
import pandas as pd
import base64


def filter_and_display_data(data):
    """
    Filter and display Reddit data
    """
    st.subheader("Results")

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Convert date strings to datetime objects
    df["date"] = pd.to_datetime(df["date"])

    # Add filter options
    st.write("Filter Data:")
    col1, col2, col3 = st.columns(3)

    with col1:
        # Filter by subreddit
        subreddit_options = ["All"] + sorted(df["subreddit"].unique().tolist())
        selected_subreddit = st.selectbox("Subreddit", subreddit_options)

    with col2:
        # Filter by type (Post or Comment)
        type_options = ["All"] + sorted(df["type"].unique().tolist())
        selected_type = st.selectbox("Content Type", type_options)

    with col3:
        # Filter by sentiment
        sentiment_options = ["All"] + sorted(df["sentiment_score"].unique().tolist())
        selected_sentiment = st.selectbox("Sentiment Score", sentiment_options)

    # Add content search filter
    content_search = st.text_input("Search in content (case-insensitive):", "")

    # Apply filters
    filtered_df = apply_filters(
        df, selected_subreddit, selected_type, selected_sentiment, content_search
    )

    # Display filtered data with interactive sorting enabled
    display_data_table(filtered_df)

    # Provide download option
    provide_download_option(filtered_df)

    # Show statistics
    display_statistics(filtered_df)


def apply_filters(
    df, selected_subreddit, selected_type, selected_sentiment, content_search
):
    """
    Apply filters to the DataFrame
    """
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

    return filtered_df


def display_data_table(filtered_df):
    """
    Display the filtered data in an interactive table
    """
    st.write(
        "You can sort the data by clicking on column headers. Hold Shift to sort by multiple columns."
    )
    st.data_editor(
        filtered_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "link": st.column_config.LinkColumn("Link"),
            "sentiment_score": st.column_config.NumberColumn(
                "Sentiment",
                help="Sentiment score from 1-5 (higher is more positive)",
                format="%.1f",
            ),
            "date": st.column_config.DateColumn("Date"),
            "content": st.column_config.TextColumn("Content", width="large"),
        },
    )


def provide_download_option(filtered_df):
    """
    Provide a download option for the filtered data
    """
    csv = filtered_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    download_link = f'<a href="data:file/csv;base64,{b64}" download="reddit_data.csv">Download CSV file</a>'
    st.markdown(download_link, unsafe_allow_html=True)


def display_statistics(filtered_df):
    """
    Display statistics about the filtered data
    """
    st.subheader("Statistics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Items", len(filtered_df))
    col2.metric("Posts", len(filtered_df[filtered_df["type"] == "Post"]))
    col3.metric("Comments", len(filtered_df[filtered_df["type"] == "Comment"]))

    # Calculate average sentiment only if there's data
    avg_sentiment = (
        f"{filtered_df['sentiment_score'].mean():.2f}/5"
        if not filtered_df.empty
        else "N/A"
    )
    col4.metric("Average Sentiment", avg_sentiment)
