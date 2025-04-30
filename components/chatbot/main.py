import streamlit as st
import pandas as pd
from datetime import datetime
import openai

from .context import prepare_base_context, prepare_rag_context
from .data_analysis import init_df, create_schema_info
from ..data_handling import initialize_dataframe_index


def render_chatbot(data, openai_api_key, negative_words, nltk_resources):
    """
    Render the chatbot UI and handle interactions
    """
    st.subheader("Chat with Your Reddit Data")

    # Add a button to clear chat history
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []

        # Also clear any cached analysis to ensure fresh data
        if "chatbot_df" in st.session_state:
            del st.session_state.chatbot_df
        if "sentiment_analysis" in st.session_state:
            del st.session_state.sentiment_analysis
        if "weekly_data" in st.session_state:
            del st.session_state.weekly_data
        if "word_counts_by_week" in st.session_state:
            del st.session_state.word_counts_by_week
        if "dataset_schema" in st.session_state:
            del st.session_state.dataset_schema

        st.rerun()

    # Initialize and analyze DataFrame
    if "chatbot_df" not in st.session_state and data:
        init_df(data, negative_words, nltk_resources)

    # Initialize the dataframe index if needed
    if "dataframe_index" not in st.session_state and "chatbot_df" in st.session_state:
        df = st.session_state.chatbot_df
        st.session_state.dataframe_index = initialize_dataframe_index(df)

    # Initialize chat container
    chat_container = st.container()

    # Display chat history
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat input
    user_input = st.chat_input("Ask a question about the Reddit data...")

    if user_input:
        # Ensure we have the DataFrame loaded
        if "chatbot_df" not in st.session_state and data:
            init_df(data, negative_words, nltk_resources)

        df = st.session_state.chatbot_df

        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)

        # Prepare context for the model
        data_context = prepare_base_context(df)
        rag_context = prepare_rag_context(
            df, user_input, negative_words, nltk_resources
        )

        # Combine base context with RAG context
        full_context = data_context + rag_context

        # Create the system message with enhanced context about the data
        system_content = f"""You are a data analyst assistant specialized in Reddit CookUnity data analysis.
The dataset contains {len(df) if not df.empty else 'unknown'} records from Reddit with the following information:

{full_context}

Important guidelines:
1. You have COMPLETE access to all data in the DataFrame.
2. You can count, filter, and aggregate data to answer questions precisely.
3. All your answers MUST be based on actual data, not assumptions.
4. Use specific examples from the data when available.
5. For word frequency analysis, use exact counts from the data.
6. For any date-based question, check the actual date range in the data.
7. If links or examples are requested, provide the ACTUAL links and content from the dataset.
8. NEVER make up information that isn't in the data.
9. If you can't find information in the data, clearly state that limitation.

The data has already been filtered to contain relevant CookUnity content from Reddit."""

        # Update the messages array with the new system content
        messages = [{"role": "system", "content": system_content}]

        # Check if the user is requesting a graph
        from ..visualization import detect_graph_request, generate_graph

        graph_request = detect_graph_request(user_input)

        # If a graph is requested, generate it
        if graph_request:
            process_graph_request(graph_request, df, rag_context)
        else:
            process_text_request(messages, openai_api_key)


def process_graph_request(graph_request, df, rag_context):
    """Process a graph request from the user"""
    from ..visualization import generate_graph

    # Generate the graph
    fig, error = generate_graph(
        graph_type=graph_request["type"],
        df=df,
        x=graph_request["x"],
        y=graph_request["y"],
        title=graph_request["title"],
    )

    # If the graph was generated successfully, display it
    if fig:
        # Create a unique key for this graph (based on timestamp)
        graph_key = f"graph_{datetime.now().timestamp()}"

        # Display the graph
        with st.chat_message("assistant"):
            st.write(
                f"Here's a {graph_request['type']} chart of {graph_request['y'] or 'counts'} by {graph_request['x']}:"
            )
            st.plotly_chart(fig, use_container_width=True, key=graph_key)

        # Store the response to add to chat history
        graph_response = f"I've created a {graph_request['type']} chart showing {graph_request['y'] or 'counts'} by {graph_request['x']}."

        # Add to chat history
        st.session_state.chat_history.append(
            {"role": "assistant", "content": graph_response}
        )
    elif error:
        # If there was an error, display it
        with st.chat_message("assistant"):
            st.error(f"I couldn't create the requested graph: {error}")

        # Store the error message to add to chat history
        graph_response = f"I couldn't create the requested graph: {error}"
        st.session_state.chat_history.append(
            {"role": "assistant", "content": graph_response}
        )


def process_text_request(messages, openai_api_key):
    """Process a text request using the OpenAI API"""
    try:
        # Set up OpenAI API with the updated client
        client = openai.OpenAI(api_key=openai_api_key)

        # Create a response using OpenAI with better chain-of-thought prompting
        with st.spinner("Thinking..."):
            # Get full conversation history for context
            messages.extend(
                [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in st.session_state.chat_history[-10:]
                ]
            )

            response = client.chat.completions.create(
                # use a more advanced model
                model="gpt-4o-mini-2024-07-18",
                messages=messages,
                temperature=0.2,  # Lower temperature for more factual responses
            )

        # Extract response content
        assistant_response = response.choices[0].message.content

        # Add assistant response to chat history
        st.session_state.chat_history.append(
            {"role": "assistant", "content": assistant_response}
        )

        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(assistant_response)

    except Exception as e:
        with st.chat_message("assistant"):
            st.error(f"Error: {str(e)}")
            st.session_state.chat_history.append(
                {"role": "assistant", "content": f"Error: {str(e)}"}
            )
