import streamlit as st
import re
import pandas as pd


def prepare_base_context(df):
    """
    Prepare base data context with general information about the dataset
    """
    data_context = ""
    if not df.empty and "dataset_schema" in st.session_state:
        schema = st.session_state.dataset_schema

        # Add comprehensive statistics about the data
        data_context += f"## Dataset Statistics:\n"
        data_context += f"- Total Items: {len(df)} items\n"

        if "type" in df.columns:
            type_counts = df["type"].value_counts().to_dict()
            for type_name, count in type_counts.items():
                data_context += f"- {type_name}s: {count} items\n"

        if "sentiment_score" in df.columns:
            data_context += (
                f"- Average Sentiment Score: {df['sentiment_score'].mean():.2f}\n"
            )

        # Make sure dates are properly handled if date column exists
        if "date" in df.columns:
            try:
                # Check if date column is already datetime
                if pd.api.types.is_datetime64_any_dtype(df["date"]):
                    date_min = df["date"].min().strftime("%Y-%m-%d")
                    date_max = df["date"].max().strftime("%Y-%m-%d")
                else:
                    # Convert to datetime first if it's a string
                    date_series = pd.to_datetime(df["date"])
                    date_min = date_series.min().strftime("%Y-%m-%d")
                    date_max = date_series.max().strftime("%Y-%m-%d")

                data_context += f"- Date Range: {date_min} to {date_max}\n"
            except Exception as e:
                # Fallback if date conversion fails
                data_context += "- Date Range: Not available\n"

        # Add subreddit statistics if subreddit column exists
        if "subreddit" in df.columns:
            data_context += f"- Subreddits: {', '.join(df['subreddit'].unique())}\n\n"

        # Add column information directly from schema
        data_context += "## Available Columns in Dataset:\n"
        for col in schema["columns"]:
            data_context += f"- **{col}**: {schema['descriptions'][col]}\n"

        data_context += "\n"

        # Add sentiment analysis results
        if "sentiment_analysis" in st.session_state:
            sentiment_analysis = st.session_state.sentiment_analysis

            data_context += "## Most Common Negative Words:\n"
            if sentiment_analysis["common_negative_words"]:
                for word, count in sentiment_analysis["common_negative_words"][
                    :10
                ]:  # Top 10 for brevity
                    data_context += f"- {word}: {count} occurrences\n"
            else:
                data_context += "No significant negative words found.\n"

        # Add WoW data for chat context if available
        if (
            "weekly_data" in st.session_state
            and "word_counts_by_week" in st.session_state
        ):
            data_context += "\n## Week-over-Week Analysis Available:\n"
            data_context += "- Post/Comment counts by week\n"
            data_context += "- Sentiment scores by week\n"
            data_context += "- Word frequency by week\n"

        # Add sample data for context - dynamically using actual column names
        data_context += "\n## Sample Data (3 rows):\n"
        sample_df = df.head(3)
        for i, row in sample_df.iterrows():
            data_context += f"\n### Row {i+1}:\n"
            for col in df.columns:
                # Format the value nicely, limiting content length
                value = row[col]
                if col == "content" and isinstance(value, str) and len(value) > 100:
                    value = value[:100] + "..."
                data_context += f"- **{col}**: {value}\n"

    return data_context


def prepare_rag_context(df, query, negative_words, nltk_resources):
    """
    Prepare RAG context based on the user's query
    """
    import streamlit as st

    # Analyze the user query to determine what context to retrieve
    query_lower = query.lower()
    rag_context = ""

    # Check if query is about links, examples, or specific content
    content_query_terms = [
        "link",
        "url",
        "example",
        "show me",
        "list",
        "content",
        "where",
        "appear",
    ]
    if any(term in query_lower for term in content_query_terms):
        rag_context += prepare_content_context(
            df, query_lower, negative_words, nltk_resources
        )

    # Check if query is about columns, schema or structure
    schema_related_terms = [
        "column",
        "schema",
        "field",
        "attribute",
        "structure",
        "data type",
        "format",
    ]
    if any(term in query_lower for term in schema_related_terms):
        rag_context += prepare_schema_context(df)

    # Check for mentions of specific words or terms
    if any(word in query_lower for word in negative_words):
        rag_context += prepare_word_frequency_context(query_lower, negative_words)

    # Check for WoW (Week-over-Week) mentions
    if (
        "wow" in query_lower
        or "week-over-week" in query_lower
        or "week over week" in query_lower
    ):
        rag_context += prepare_wow_context()

    # Prepare dynamic query-specific data context
    query_data_context = prepare_query_data_context(query_lower)

    # Add the query-specific context to the RAG context
    return query_data_context + rag_context


def prepare_content_context(df, query_lower, negative_words, nltk_resources):
    """Prepare context for content-related queries"""
    import streamlit as st
    import re

    context = ""
    # Check if query mentions specific words or topics to find
    search_terms = []

    # Check for negative words mentions
    for word in negative_words:
        if word in query_lower:
            search_terms.append(word)

    # If no specific negative words found, check for other potential search terms in query
    if not search_terms:
        # Extract potential search terms (nouns/key terms) from the query
        # Simple approach: words of 4+ chars that aren't stop words
        stop_words = set(nltk_resources["stopwords"])
        query_words = query_lower.split()
        potential_terms = [
            word for word in query_words if len(word) >= 4 and word not in stop_words
        ]
        search_terms.extend(potential_terms)

    # If we have terms to search for, find matching content
    if search_terms:
        context += f"\n### Actual Content Examples with '{', '.join(search_terms)}':\n"

        # First check if content column exists
        if "content" not in df.columns:
            context += "Cannot search content - the dataset does not have a 'content' column.\n"
        else:
            # Find matching content - use exact search to ensure we find actual matches
            matches = []
            match_count = 0

            # Count total matches
            for _, row in df.iterrows():
                if not isinstance(row["content"], str):
                    continue

                content = row["content"].lower()
                for term in search_terms:
                    if re.search(r"\b" + re.escape(term.lower()) + r"\b", content):
                        match_count += 1
                        break

            if match_count == 0:
                context += f"No content found matching these search terms: {', '.join(search_terms)}\n"
            else:
                # Add matches with detailed information
                context += f"Found {match_count} items containing your search terms. Here are some examples:\n\n"

                limit = min(10, match_count)  # Set a reasonable limit
                added = 0

                for i, row in df.iterrows():
                    if added >= limit:
                        break  # Exit once we've added enough examples

                    if not isinstance(row["content"], str):
                        continue

                    content = row["content"].lower()

                    # Check if any search term appears
                    for term in search_terms:
                        if re.search(r"\b" + re.escape(term.lower()) + r"\b", content):
                            # Create a well-formatted match info
                            match_info = f"- **{row['type']}** in r/{row['subreddit']}"

                            # Add date if available
                            if "date" in row and row["date"] is not None:
                                date_val = row["date"]
                                if isinstance(date_val, pd.Timestamp):
                                    date_str = date_val.strftime("%Y-%m-%d")
                                else:
                                    date_str = str(date_val)
                                match_info += f" ({date_str})"

                            # Add sentiment score
                            if "sentiment_score" in row:
                                match_info += f", sentiment: {row['sentiment_score']}"

                            # Add link - this is crucial for the question
                            if "link" in row and row["link"]:
                                match_info += f"\n  **URL**: {row['link']}"

                            # Format the content to highlight the match
                            content_preview = row["content"]
                            if len(content_preview) > 200:
                                # Find the position of the term to center the preview
                                pos = content.find(term.lower())
                                if pos >= 0:
                                    start = max(0, pos - 75)
                                    end = min(len(content), pos + 75)

                                    # Try to find word boundaries
                                    while start > 0 and content[start] != " ":
                                        start -= 1
                                    while end < len(content) and content[end] != " ":
                                        end += 1

                                    # Create snippet with ellipses if needed
                                    snippet = content_preview[start:end]
                                    if start > 0:
                                        snippet = f"...{snippet}"
                                    if end < len(content):
                                        snippet = f"{snippet}..."

                                    content_preview = snippet
                                else:
                                    # If term not found in this pass, just truncate
                                    content_preview = f"{content_preview[:197]}..."

                            match_info += f'\n  **Content**: "{content_preview}"'

                            matches.append(match_info)
                            added += 1
                            break  # Only add once per row

                # Add the matches to the context
                for i, match in enumerate(matches):
                    context += f"{i+1}. {match}\n\n"

    return context


def prepare_schema_context(df):
    """Prepare context for schema-related queries"""
    import streamlit as st

    context = ""
    # Access the dynamically generated schema
    if "dataset_schema" in st.session_state:
        schema = st.session_state.dataset_schema

        context += "\n### Dataset Schema Details:\n"
        context += "The Reddit data contains the following columns:\n"

        # List all columns with their dynamically generated descriptions
        for col in schema["columns"]:
            # Include sample values for context where useful
            if schema["sample_counts"][col] < 10 and col != "content" and col != "link":
                sample_values = df[col].dropna().unique()[:5]
                samples = ", ".join([f"'{str(val)}'" for val in sample_values])
                context += f"- **{col}**: {schema['descriptions'][col]}\n  Example values: {samples}\n"
            else:
                context += f"- **{col}**: {schema['descriptions'][col]}\n"

    return context


def prepare_word_frequency_context(query_lower, negative_words):
    """Prepare context for word frequency queries"""
    import streamlit as st

    context = ""
    # If asking about specific negative words
    mentioned_words = [word for word in negative_words if word in query_lower]

    for word in mentioned_words:
        context += f"\n### Frequency of '{word}':\n"

        # Total occurrences
        if "sentiment_analysis" in st.session_state:
            word_counts = dict(
                st.session_state.sentiment_analysis["common_negative_words"]
            )
            if word in word_counts:
                context += f"- Total occurrences: {word_counts[word]}\n"

        # Weekly occurrences if available
        if "word_counts_by_week" in st.session_state:
            word_counts_by_week = st.session_state.word_counts_by_week
            context += f"- Weekly occurrences of '{word}':\n"

            weeks_sorted = sorted(word_counts_by_week.keys())
            for week in weeks_sorted:
                counts = word_counts_by_week[week]
                count = counts.get(word, 0)
                if count > 0:
                    context += f"  - Week {week}: {count} occurrences\n"

            # Add WoW changes
            if len(weeks_sorted) > 1:
                context += f"- Week-over-Week changes for '{word}':\n"
                for i in range(1, len(weeks_sorted)):
                    current_week = weeks_sorted[i]
                    prev_week = weeks_sorted[i - 1]

                    current_count = word_counts_by_week[current_week].get(word, 0)
                    prev_count = word_counts_by_week[prev_week].get(word, 0)

                    if prev_count > 0:
                        wow_change = current_count - prev_count
                        wow_pct = (
                            ((current_count / prev_count) - 1) * 100
                            if prev_count > 0
                            else 0
                        )

                        context += f"  - {prev_week} → {current_week}: {wow_change} ({wow_pct:.1f}%)\n"

    return context


def prepare_wow_context():
    """Prepare context for week-over-week analysis"""
    import streamlit as st

    context = ""
    if "weekly_data" in st.session_state:
        weekly_data = st.session_state.weekly_data

        context += "\n### Week-over-Week Analysis:\n"

        # Post/Comment counts by week
        counts_by_week = weekly_data["counts_by_week"]
        weeks = sorted(counts_by_week["year_week"].unique())

        if len(weeks) > 1:
            context += "- Weekly post/comment counts:\n"
            for week in weeks:
                week_counts = counts_by_week[counts_by_week["year_week"] == week]
                post_count = (
                    week_counts[week_counts["type"] == "Post"]["count"].sum()
                    if "Post" in week_counts["type"].values
                    else 0
                )
                comment_count = (
                    week_counts[week_counts["type"] == "Comment"]["count"].sum()
                    if "Comment" in week_counts["type"].values
                    else 0
                )

                context += (
                    f"  - Week {week}: {post_count} posts, {comment_count} comments\n"
                )

            # WoW changes
            context += "- Week-over-Week changes:\n"
            for i in range(1, len(weeks)):
                current_week = weeks[i]
                prev_week = weeks[i - 1]

                current_counts = counts_by_week[
                    counts_by_week["year_week"] == current_week
                ]
                prev_counts = counts_by_week[counts_by_week["year_week"] == prev_week]

                cur_post_count = (
                    current_counts[current_counts["type"] == "Post"]["count"].sum()
                    if "Post" in current_counts["type"].values
                    else 0
                )
                prev_post_count = (
                    prev_counts[prev_counts["type"] == "Post"]["count"].sum()
                    if "Post" in prev_counts["type"].values
                    else 0
                )

                cur_comment_count = (
                    current_counts[current_counts["type"] == "Comment"]["count"].sum()
                    if "Comment" in current_counts["type"].values
                    else 0
                )
                prev_comment_count = (
                    prev_counts[prev_counts["type"] == "Comment"]["count"].sum()
                    if "Comment" in prev_counts["type"].values
                    else 0
                )

                post_wow = cur_post_count - prev_post_count
                post_wow_pct = (
                    ((cur_post_count / prev_post_count) - 1) * 100
                    if prev_post_count > 0
                    else 0
                )

                comment_wow = cur_comment_count - prev_comment_count
                comment_wow_pct = (
                    ((cur_comment_count / prev_comment_count) - 1) * 100
                    if prev_comment_count > 0
                    else 0
                )

                context += f"  - {prev_week} → {current_week}:\n"
                context += f"    - Posts: {post_wow} ({post_wow_pct:.1f}%)\n"
                context += f"    - Comments: {comment_wow} ({comment_wow_pct:.1f}%)\n"

    return context


def prepare_query_data_context(query_lower):
    """Prepare context for specific data queries"""
    import streamlit as st
    import re
    from ..data_handling import find_data_by_attribute

    context = ""
    # Check for specific data queries (e.g., "show me posts with sentiment score < 2")
    # Add regex patterns to identify different types of data queries
    data_query_patterns = [
        (
            r"sentiment\s+(?:score|rating)\s*(?:<|less than|below)\s*(\d+)",
            "sentiment_score",
            "<",
        ),
        (
            r"sentiment\s+(?:score|rating)\s*(?:>|greater than|above)\s*(\d+)",
            "sentiment_score",
            ">",
        ),
        (
            r"sentiment\s+(?:score|rating)\s*(?:=|equal to|is)\s*(\d+)",
            "sentiment_score",
            "=",
        ),
        (r'(?:in|from|on)\s+subreddit\s+["\']?(\w+)["\']?', "subreddit", "="),
        (
            r'(?:post|comment)s?\s+(?:with|containing)\s+["\']([^"\']+)["\']',
            "content",
            "contains",
        ),
        (
            r'(?:post|comment)s?\s+(?:from|on|dated)\s+(?:date\s+)?["\']?(\d{4}-\d{2}-\d{2})["\']?',
            "date",
            "=",
        ),
    ]

    for pattern, attribute, operator in data_query_patterns:
        matches = re.search(pattern, query_lower)
        if matches:
            value = matches.group(1)

            # Convert value to appropriate type
            if attribute == "sentiment_score":
                value = float(value)

            # Find matching data
            if operator == "=":
                matching_rows = find_data_by_attribute(attribute, value, True)
            elif operator == "contains":
                matching_rows = find_data_by_attribute("content", value, False)
            elif operator == "<":
                # For less than comparisons
                matching_rows = [
                    row
                    for idx, row in st.session_state.dataframe_index["data"].items()
                    if attribute in row and row[attribute] < float(value)
                ]
            elif operator == ">":
                # For greater than comparisons
                matching_rows = [
                    row
                    for idx, row in st.session_state.dataframe_index["data"].items()
                    if attribute in row and row[attribute] > float(value)
                ]

            # Format results for RAG context
            if matching_rows:
                context += (
                    f"\n### Data matching your query ({len(matching_rows)} results):\n"
                )
                # Include up to 5 examples
                for i, row in enumerate(matching_rows[:5]):
                    context += f"\n#### Result {i+1}:\n"
                    for col in st.session_state.dataframe_index["metadata"]["columns"]:
                        # Format content to be shorter
                        if (
                            col == "content"
                            and isinstance(row[col], str)
                            and len(row[col]) > 100
                        ):
                            context += f'- **{col}**: "{row[col][:100]}..."\n'
                        else:
                            context += f"- **{col}**: {row[col]}\n"

                # Add summary stats if many results
                if len(matching_rows) > 5:
                    context += f"\n(Showing 5 of {len(matching_rows)} matches. There are {len(matching_rows)-5} more results.)\n"
            else:
                context += f"\nNo data found matching your query for {attribute} {operator} {value}.\n"

    return context
