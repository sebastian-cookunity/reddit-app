import streamlit as st

# Set page configuration - must be the first Streamlit command
st.set_page_config(
    page_title="Reddit CookUnity Monitor",
    page_icon="https://i.imgur.com/tnwmOZN.png",
    layout="wide",
)

import pandas as pd
from datetime import datetime, timedelta

# Import all modules
from components.constants import cu_image_url, reddit_image_url, NEGATIVE_WORDS
from components.nlp import initialize_nltk_resources
from components.sidebar import render_sidebar
from components.main_flow import process_analysis
from components.analytics import render_analytics
from components.chatbot import render_chatbot
from auth import setup_authentication
from reddit_api import restore_reddit_instance
from utils import setup_logging
from ui_components import display_faq, display_openai_faq
from data_processing import filter_and_display_data

# Display header with images
col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1])
with col3:
    st.image(cu_image_url, width=150)
with col4:
    st.image(reddit_image_url, width=150)

# Initialize NLTK resources
nltk_resources = initialize_nltk_resources()

# Setup logging
setup_logging()

# Initialize session state
if "data" not in st.session_state:
    st.session_state.data = None
if "reddit" not in st.session_state:
    st.session_state.reddit = None
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "credentials" not in st.session_state:
    st.session_state.credentials = {
        "client_id": "",
        "client_secret": "",
        "user_agent": "",
    }
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Try to restore Reddit instance on app load
restore_reddit_instance()

# App title and description
st.title("Reddit CookUnity Monitor")
st.markdown("Monitor subreddits for posts and comments related to CookUnity")

# Create sidebar for inputs
run_button = render_sidebar()

# Display FAQ
display_faq()
display_openai_faq()

# Main app flow
if run_button:
    process_analysis()

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["Reddit Data", "Analytics", "Chatbot"])

# Tab 1: Reddit Data - Display results
with tab1:
    if st.session_state.data:
        filter_and_display_data(st.session_state.data)
    elif run_button and st.session_state.authenticated:
        st.info("No matching posts or comments found in the selected date range.")

# Tab 2: Analytics - Data visualization
with tab2:
    if st.session_state.data:
        render_analytics(st.session_state.data)
    else:
        st.info("Run the analysis to see analytics visualizations")

# Tab 3: Chatbot - Chat with the data
with tab3:
    if st.session_state.data and st.session_state.openai_api_key:
        render_chatbot(
            st.session_state.data,
            st.session_state.openai_api_key,
            NEGATIVE_WORDS,
            nltk_resources,
        )
    elif not st.session_state.data:
        st.info("Run the analysis to generate data for the chatbot")
    elif not st.session_state.openai_api_key:
        st.warning("Please enter your OpenAI API key in the sidebar to use the chatbot")

# Footer
st.markdown("---")
st.markdown("Reddit CookUnity Monitor - Powered by Streamlit")
