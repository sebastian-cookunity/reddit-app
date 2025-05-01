import streamlit as st
import praw
from datetime import datetime, timezone
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# Initialize VADER sentiment analyzer - cached using streamlit
@st.cache_resource
def initialize_vader():
    # Vader initialization should be done once
    try:
        nltk.data.find("sentiment/vader_lexicon")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)

    return SentimentIntensityAnalyzer()


# Get VADER sentiment analyzer instance
sia = initialize_vader()

# Search terms (case-insensitive) for CookUnity variations
search_terms = [
    r"cookunity",
    r"cook-unity",
    r"cook unity",
    r"cook\s*unity",
    r"CookUnity",
    r"Cook-Unity",
    r"Cook Unity",
    r"Cook\s*Unity",
    r"\bCU\b",
    r"\bcu\b",  # Added CU and cu as variations
]
pattern = re.compile("|".join(search_terms), re.IGNORECASE)


def restore_reddit_instance():
    """
    Restore the Reddit instance if authenticated but the instance is None
    This can happen after page reload
    """
    if st.session_state.authenticated and st.session_state.reddit is None:
        try:
            # Recreate the Reddit instance from stored credentials
            creds = st.session_state.credentials
            st.session_state.reddit = praw.Reddit(
                client_id=creds["client_id"],
                client_secret=creds["client_secret"],
                user_agent=creds["user_agent"],
                check_for_async=False,
            )
            # Verify it works
            subreddit = st.session_state.reddit.subreddit("askreddit")
            subreddit_name = subreddit.display_name
        except Exception as e:
            # If recreation fails, reset authentication
            st.error(f"Error restoring session: {e}")
            st.session_state.authenticated = False


def is_within_date_range(timestamp, start_date, end_date):
    """Check if item is within the specified date range"""
    # Get creation time (UTC)
    item_time = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    item_date = item_time.date()

    # Convert start_date and end_date to datetime.date if they're not already
    if isinstance(start_date, datetime):
        start_date = start_date.date()
    if isinstance(end_date, datetime):
        end_date = end_date.date()

    # Check if item is within the specified date range (inclusive)
    return start_date <= item_date <= end_date


def subreddit_contains_cookunity(subreddit_name):
    """Check if the subreddit name contains a variation of CookUnity."""
    return bool(pattern.search(subreddit_name))


def get_sentiment(text):
    """Calculate sentiment score from text using NLTK VADER

    VADER is specifically designed for social media text and handles things like:
    - Capitalization and punctuation emphasis
    - Emoticons and emojis
    - Negations
    - Common slang and abbreviations
    """
    # Get sentiment scores
    sentiment_scores = sia.polarity_scores(text)

    # The compound score is a normalized score between -1 (most negative) and 1 (most positive)
    compound_score = sentiment_scores["compound"]

    # Convert compound score (-1 to 1) to 1-5 scale
    # -1 → 1, -0.5 → 2, 0 → 3, 0.5 → 4, 1 → 5
    score = round((compound_score + 1) * 2) + 1

    # Ensure score is within 1-5 range
    return max(1, min(5, score))


def process_comment(
    comment,
    submission,
    start_date,
    end_date,
    data,
    depth=1,
    scan_all=False,
    page_number=1,
):
    """Recursively process a comment and its replies."""
    comment_time = datetime.fromtimestamp(comment.created_utc, tz=timezone.utc)

    # Skip Community Highlights or pinned/mod comments
    if comment.distinguished or comment.stickied:
        return

    # Validate comment is within the specified date range
    if not is_within_date_range(comment.created_utc, start_date, end_date):
        return

    # If scan_all is True (subreddit contains CookUnity), include all comments
    # Otherwise, only include comments with CookUnity matches
    matches = pattern.findall(comment.body) if not scan_all else ["null"]
    if scan_all or matches:
        comment_date = comment_time.strftime("%Y-%m-%d")
        if not any(
            d["link"] == f"https://reddit.com{comment.permalink}" for d in data
        ):  # Avoid duplicates
            data.append(
                {
                    "type": "Comment",
                    "subreddit": submission.subreddit.display_name,
                    "thread_name": submission.title,
                    "date": comment_date,
                    "original_matching_word": (
                        "null" if scan_all else matches[0]
                    ),  # Use first match if filtering
                    "content": comment.body,
                    "link": f"https://reddit.com{comment.permalink}",
                    "sentiment_score": get_sentiment(comment.body),
                    "page_number": page_number,
                }
            )

    # Process replies recursively
    try:
        for reply in comment.replies:
            process_comment(
                reply,
                submission,
                start_date,
                end_date,
                data,
                depth + 1,
                scan_all,
                page_number,
            )
    except Exception as e:
        st.warning(f"Error processing replies: {e}")


def process_reddit_data(reddit_client, subreddits_list, start_date, end_date):
    """
    Process Reddit data from specified subreddits within date range
    """
    data = []

    # Create progress bar
    progress_text = "Analyzing Reddit posts..."
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Check specific post (1k8x48d) - note: this post is in r/cookunity
        status_text.text("Checking specific post (ID: 1k8x48d)...")
        try:
            specific_post = reddit_client.submission(id="1k8x48d")
            post_time = datetime.fromtimestamp(
                specific_post.created_utc, tz=timezone.utc
            )
            post_content = (
                f"{specific_post.title} {specific_post.selftext or ''}".strip()
            )

            # Determine if we scan all (r/cookunity contains CookUnity)
            scan_all = subreddit_contains_cookunity(
                specific_post.subreddit.display_name
            )
            if is_within_date_range(specific_post.created_utc, start_date, end_date):
                post_date = post_time.strftime("%Y-%m-%d")
                # If scan_all is True, include post regardless; otherwise, check for matches
                post_matches = (
                    pattern.findall(post_content) if not scan_all else ["null"]
                )
                if scan_all or post_matches:
                    data.append(
                        {
                            "type": "Post",
                            "subreddit": specific_post.subreddit.display_name,
                            "thread_name": specific_post.title,
                            "date": post_date,
                            "original_matching_word": (
                                "null" if scan_all else post_matches[0]
                            ),
                            "content": post_content,
                            "link": f"https://reddit.com{specific_post.permalink}",
                            "sentiment_score": get_sentiment(post_content),
                            "page_number": 0,  # Specific post check, not part of pagination
                        }
                    )

            # Check comments and replies
            specific_post.comments.replace_more(limit=0)
            for comment in specific_post.comments:
                process_comment(
                    comment,
                    specific_post,
                    start_date,
                    end_date,
                    data,
                    scan_all=scan_all,
                    page_number=0,
                )
        except Exception as e:
            st.warning(f"Error checking specific post: {e}")

        progress_bar.progress(10)

        # Scan recent posts in the specified subreddits
        status_text.text(f"Scanning posts in {', '.join(subreddits_list)}...")

        # Create a combined subreddit object
        subreddit = reddit_client.subreddit("+".join(subreddits_list))

        # Pagination setup
        page_size = 100  # Number of posts per page
        page_number = 1
        after = None  # For Reddit API pagination
        keep_fetching = True
        total_posts_processed = 0

        while keep_fetching:
            # Fetch posts for the current page
            posts = subreddit.new(limit=page_size, params={"after": after})
            posts_fetched = 0
            all_posts_too_old = True

            for submission in posts:
                posts_fetched += 1
                total_posts_processed += 1
                submission_time = datetime.fromtimestamp(
                    submission.created_utc, tz=timezone.utc
                )
                post_content = f"{submission.title} {submission.selftext or ''}".strip()

                # Update progress periodically
                if total_posts_processed % 10 == 0:
                    progress_value = min(10 + (total_posts_processed / 200) * 90, 100)
                    progress_bar.progress(int(progress_value))
                    status_text.text(f"Analyzed {total_posts_processed} posts...")

                # Check if within date range
                if is_within_date_range(submission.created_utc, start_date, end_date):
                    all_posts_too_old = False

                    # Determine if we scan all content for this subreddit
                    scan_all = subreddit_contains_cookunity(
                        submission.subreddit.display_name
                    )

                    # Include post if within date range and matches criteria
                    post_matches = (
                        pattern.findall(post_content) if not scan_all else ["null"]
                    )
                    if scan_all or post_matches:
                        post_date = submission_time.strftime("%Y-%m-%d")
                        if not any(
                            d["link"] == f"https://reddit.com{submission.permalink}"
                            for d in data
                        ):  # Avoid duplicates
                            data.append(
                                {
                                    "type": "Post",
                                    "subreddit": submission.subreddit.display_name,
                                    "thread_name": submission.title,
                                    "date": post_date,
                                    "original_matching_word": (
                                        "null" if scan_all else post_matches[0]
                                    ),
                                    "content": post_content,
                                    "link": f"https://reddit.com{submission.permalink}",
                                    "sentiment_score": get_sentiment(post_content),
                                    "page_number": page_number,
                                }
                            )

                    # Load and check comments and replies
                    submission.comments.replace_more(limit=0)
                    for comment in submission.comments:
                        process_comment(
                            comment,
                            submission,
                            start_date,
                            end_date,
                            data,
                            scan_all=scan_all,
                            page_number=page_number,
                        )
                elif submission_time.date() < start_date:
                    # If this post is older than our start date, we've gone too far back
                    keep_fetching = False
                    break

            # Update pagination
            if posts_fetched < page_size or not keep_fetching or all_posts_too_old:
                break  # No more posts to fetch or we hit the date limit

            # Update the 'after' parameter for the next page
            after = submission.name if posts_fetched > 0 else None
            page_number += 1

        # Complete the progress bar
        progress_bar.progress(100)
        status_text.text(
            f"Analysis complete! Found {len(data)} matching posts/comments."
        )

        return data

    except Exception as e:
        progress_bar.progress(100)
        status_text.text(f"Error: {e}")
        st.error(f"Error during Reddit analysis: {e}")
        return []
