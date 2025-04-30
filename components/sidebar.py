import streamlit as st
from datetime import datetime, timedelta


def render_sidebar():
    """Render the sidebar with authentication and inputs, return run_button state"""
    with st.sidebar:
        st.header("Settings")

        # Authentication section
        from auth import setup_authentication

        setup_authentication()

        # Initialize run_button to False (default)
        run_button = False

        # OpenAI API Key for chatbot
        if st.session_state.authenticated:
            st.subheader("OpenAI API Authentication")
            openai_key = st.text_input(
                "OpenAI API Key",
                value=st.session_state.openai_api_key,
                type="password",
                key="openai_key_input",
            )
            if st.button("Save API Key"):
                st.session_state.openai_api_key = openai_key
                st.success("OpenAI API Key saved!")

        # Subreddit selection and other options only shown when authenticated
        if st.session_state.authenticated:
            # Subreddit selection
            st.subheader("Subreddits to Monitor")
            default_subreddits = ["cookunityfans", "cookunity", "ReadyMeals"]
            subreddit_input = st.text_area(
                "Enter subreddits (one per line)", value="\n".join(default_subreddits)
            )

            # Store subreddit input in session state
            st.session_state.subreddit_input = subreddit_input

            # Date range selection
            st.subheader("Date Range")
            today = datetime.now().date()
            start_date = st.date_input("Start Date", today - timedelta(days=7))
            end_date = st.date_input("End Date", today)

            # Store dates in session state
            st.session_state.start_date = start_date
            st.session_state.end_date = end_date

            # Run button with #FFBB1C color and black text
            run_button = st.button(
                "Run Analysis",
                type="primary",
                use_container_width=True,
                key="run_analysis_button",
                help="Run the Reddit analysis with the selected settings",
                on_click=None,
            )

            # Apply custom CSS to change button color to #FFBB1C with black text
            st.markdown(
                """
            <style>
            div[data-testid="stButton"] button[kind="primary"] {
                background-color: #FFBB1C;
                color: black;
            }
            </style>
            """,
                unsafe_allow_html=True,
            )

    return run_button
