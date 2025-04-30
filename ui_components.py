import streamlit as st


def display_faq():
    """
    Display the FAQ section with information about Reddit API
    """
    with st.expander("FAQ - Reddit API Information"):
        st.markdown(
            """
        ## Steps to Obtain Reddit API Credentials
        
        1. **Log in to Reddit:**
           - Go to https://www.reddit.com and sign in with your Reddit account.
        
        2. **Access the Developer Apps Page:**
           - Navigate to https://www.reddit.com/prefs/apps or:
           - Click your profile icon (top-right) > User Settings > Privacy & Security tab.
           - Scroll down to the Developer Apps section and click Apps.
        
        3. **Create an App:**
           - At the bottom of the apps page, click Create App or Create Another App.
           - Fill out the form:
             - **Name:** Choose a descriptive name (e.g., CookUnityMonitor).
             - **App Type:** Select Script (for personal use scripts like yours).
             - **Description:** Optional, add a brief note (e.g., Monitor subreddit comments).
             - **About URL:** Optional, can leave blank or add a project link.
             - **Redirect URI:** Set to http://localhost:8080 (default for script apps).
             - **Permissions:** Leave as default.
           - Click Create App.
        
        4. **Retrieve Credentials:**
           - After creating the app, you'll see it listed under Developed Applications.
           - **Client ID:** This is the string just below the app name (a short alphanumeric code).
           - **Client Secret:** Labeled as secret, a longer alphanumeric string.
           - **User Agent:** name in the app
        """
        )


def display_openai_faq():
    """
    Display the FAQ section with information about OpenAI API
    """
    with st.expander("FAQ - OpenAI API Information"):
        st.markdown(
            """
        ## Steps to Obtain OpenAI API Key
        
        1. **Create an OpenAI Account:**
           - Go to https://platform.openai.com/signup to create an account if you don't have one.
           - Log in if you already have an account.
        
        2. **Access API Keys:**
           - Navigate to https://platform.openai.com/api-keys 
           - Or go to the API section from your dashboard and select "API Keys"
        
        3. **Create a New API Key:**
           - Click "Create new secret key"
           - Give your key a name that helps you remember what it's for (e.g., "Reddit CookUnity Monitor")
           - Click "Create secret key"
        
        4. **Copy Your API Key:**
           - Copy the key immediately and paste it into the "OpenAI API Key" field in this app.
           - **Important:** You will only see this key once. If you don't save it now, you'll need to create a new one.
        
        5. **Usage Considerations:**
           - OpenAI API is a paid service that charges based on usage.
           - The API uses a token-based pricing model.
           - For this app, usage should be minimal, but be aware that costs can accrue if you make many requests.
           - You can set usage limits in your OpenAI account settings.
           
        **Note:** Your OpenAI API key is stored only in your browser's session and is not saved on any server.
        """
        )
