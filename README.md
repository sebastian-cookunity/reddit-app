# Reddit CookUnity Monitor

A Streamlit app that monitors Reddit subreddits for posts and comments related to CookUnity.

## Project Structure

The application is organized into multiple modules:

- `app.py` - Main application file
- `auth.py` - Reddit API authentication
- `reddit_api.py` - Reddit data fetching and processing
- `data_processing.py` - Data filtering and display
- `ui_components.py` - UI-related components
- `utils.py` - Utility functions

## Features

- Monitor multiple subreddits for CookUnity-related content
- Specify date ranges for data collection
- Sentiment analysis of posts and comments
- Filter and export results
- Interactive data visualization
- Custom Reddit API credential input

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:

```bash
streamlit run app.py
```

2. Enter your Reddit API credentials (see FAQ section in the app for how to obtain these)
3. Select the subreddits you want to monitor
4. Choose a date range
5. Click "Run Analysis"
6. View and filter the results directly in the app

## Requirements

- Python 3.7+
- Reddit API credentials (see FAQ section in the app)
- Internet connection

## Privacy & Terms of Use

- Ensure you comply with Reddit's API Terms of Service when using this app
- This tool is intended for legitimate research purposes only 