import streamlit as st
from reddit_api import process_reddit_data


def process_analysis():
    """Process the main application flow when Run Analysis is clicked"""
    # Parse subreddits from input text
    subreddits_list = [
        s.strip() for s in st.session_state.subreddit_input.split("\n") if s.strip()
    ]
    start_date = st.session_state.start_date
    end_date = st.session_state.end_date

    if not subreddits_list:
        st.error("Please enter at least one subreddit to monitor.")
    elif start_date > end_date:
        st.error("Start date cannot be after end date.")
    else:
        # Display status
        st.subheader("Analyzing Reddit Data")
        st.info(f"Monitoring subreddits: {', '.join(subreddits_list)}")
        st.info(f"Date range: {start_date} to {end_date}")

        # Run the analysis
        st.session_state.data = process_reddit_data(
            st.session_state.reddit, subreddits_list, start_date, end_date
        )
