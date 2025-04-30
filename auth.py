import streamlit as st
import praw


def setup_authentication():
    """
    Handle the Reddit API authentication process
    """
    # Authentication section
    st.subheader("Reddit API Authentication")

    # If already authenticated, show status and logout option
    if st.session_state.authenticated:
        st.success("✅ Authenticated and ready to use")

        # Show saved credentials (partially hidden)
        st.info(f"Using credentials for: {st.session_state.credentials['user_agent']}")

        # Logout button
        if st.button("Logout / Change Credentials"):
            st.session_state.authenticated = False
            st.session_state.reddit = None
            # We don't clear credentials to allow easy re-login
    else:
        # Reddit API credentials
        st.subheader("Reddit API Credentials")
        client_id = st.text_input(
            "Client ID",
            value=st.session_state.credentials["client_id"],
            key="client_id_input",
        )
        client_secret = st.text_input(
            "Client Secret",
            value=st.session_state.credentials["client_secret"],
            type="password",
            key="client_secret_input",
        )
        user_agent = st.text_input(
            "User Agent",
            value=st.session_state.credentials["user_agent"],
            placeholder="app_name/version by username",
            key="user_agent_input",
        )

        # Authentication button
        auth_button = st.button("Authenticate")

        if auth_button:
            authenticate_with_reddit(client_id, client_secret, user_agent)


def authenticate_with_reddit(client_id, client_secret, user_agent):
    """
    Authenticate with the Reddit API using the provided credentials
    """
    try:
        # Create Reddit instance
        reddit_instance = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
            check_for_async=False,
        )

        # Check if instance is working by getting Reddit info
        reddit_name = reddit_instance.config.reddit_url

        # Test if we can access Reddit (even if read-only)
        subreddit = reddit_instance.subreddit("askreddit")
        subreddit_name = (
            subreddit.display_name
        )  # This will fail if connection isn't working

        # Save to session state (even if read-only)
        st.session_state.reddit = reddit_instance
        st.session_state.authenticated = True

        # Store credentials in our session state dictionary
        st.session_state.credentials = {
            "client_id": client_id,
            "client_secret": client_secret,
            "user_agent": user_agent,
        }

        # Try to get username, but it's okay if it's None (read-only mode)
        try:
            username = reddit_instance.user.me()
            if username:
                auth_message = f"Authenticated successfully as: {username}"
            else:
                auth_message = "Authenticated successfully (read-only mode)"
        except:
            auth_message = "Authenticated successfully (read-only mode)"

        st.success(auth_message)
    except Exception as e:
        st.error(f"Authentication failed: {e}")
        st.session_state.authenticated = False
