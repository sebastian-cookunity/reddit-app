import streamlit as st
import pandas as pd
import plotly.express as px
import re


def generate_graph(
    graph_type, df, x=None, y=None, title=None, color=None, filters=None
):
    """Generate a Plotly graph based on the specified parameters"""
    # Apply filters if provided
    if filters:
        filtered_df = df.copy()
        for column, value in filters.items():
            if column in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[column] == value]
    else:
        filtered_df = df

    if filtered_df.empty:
        return None, "No data available after applying filters"

    try:
        if graph_type == "bar":
            # Create a bar chart
            if not x or not y:
                return None, "Bar chart requires x and y parameters"

            # Check if we need to aggregate the data
            if y == "count":
                # Count occurrences of x values
                counts = filtered_df[x].value_counts().reset_index()
                counts.columns = [x, "count"]
                fig = px.bar(
                    counts, x=x, y="count", title=title or f"Count of {x}", color=color
                )
            else:
                # Use the specified y column
                if y not in filtered_df.columns:
                    return None, f"Column '{y}' not found in the data"

                # Group by x and calculate the mean of y
                grouped = filtered_df.groupby(x)[y].mean().reset_index()
                fig = px.bar(
                    grouped, x=x, y=y, title=title or f"{y} by {x}", color=color
                )

            # Customize the layout
            fig.update_layout(xaxis_title=x, yaxis_title=y if y != "count" else "Count")

            return fig, None

        elif graph_type == "line":
            # Create a line chart
            if not x or not y:
                return None, "Line chart requires x and y parameters"

            # Make sure x is in the dataframe
            if x not in filtered_df.columns:
                return None, f"Column '{x}' not found in the data"

            # For time series, make sure x is sorted
            if (
                pd.api.types.is_datetime64_any_dtype(filtered_df[x])
                or x == "date"
                or x == "year_week"
            ):
                # Convert to datetime if it's a string date
                if x == "date" and not pd.api.types.is_datetime64_any_dtype(
                    filtered_df[x]
                ):
                    filtered_df[x] = pd.to_datetime(filtered_df[x])

                # Sort by date
                filtered_df = filtered_df.sort_values(by=x)

            # Check if we need to aggregate the data
            if y == "count":
                # Group by x and count
                counts = filtered_df.groupby(x).size().reset_index(name="count")
                fig = px.line(
                    counts,
                    x=x,
                    y="count",
                    title=title or f"Count over {x}",
                    markers=True,
                )
                y_label = "Count"
            else:
                # Use the specified y column
                if y not in filtered_df.columns:
                    return None, f"Column '{y}' not found in the data"

                # Group by x and calculate the mean of y
                grouped = filtered_df.groupby(x)[y].mean().reset_index()
                fig = px.line(
                    grouped, x=x, y=y, title=title or f"{y} over {x}", markers=True
                )
                y_label = y

            # Customize the layout
            fig.update_layout(xaxis_title=x, yaxis_title=y_label)

            return fig, None

        elif graph_type == "pie":
            # Create a pie chart
            if not x:
                return None, "Pie chart requires at least x parameter"

            # Check if x is in the dataframe
            if x not in filtered_df.columns:
                return None, f"Column '{x}' not found in the data"

            # Count occurrences of x values
            counts = filtered_df[x].value_counts().reset_index()
            counts.columns = [x, "count"]

            fig = px.pie(
                counts, names=x, values="count", title=title or f"Distribution of {x}"
            )

            return fig, None

        elif graph_type == "scatter":
            # Create a scatter plot
            if not x or not y:
                return None, "Scatter plot requires x and y parameters"

            # Check if x and y are in the dataframe
            if x not in filtered_df.columns:
                return None, f"Column '{x}' not found in the data"
            if y not in filtered_df.columns:
                return None, f"Column '{y}' not found in the data"

            fig = px.scatter(
                filtered_df, x=x, y=y, color=color, title=title or f"{y} vs {x}"
            )

            # Customize the layout
            fig.update_layout(xaxis_title=x, yaxis_title=y)

            return fig, None

        elif graph_type == "histogram":
            # Create a histogram
            if not x:
                return None, "Histogram requires x parameter"

            # Check if x is in the dataframe
            if x not in filtered_df.columns:
                return None, f"Column '{x}' not found in the data"

            fig = px.histogram(
                filtered_df, x=x, color=color, title=title or f"Distribution of {x}"
            )

            # Customize the layout
            fig.update_layout(xaxis_title=x, yaxis_title="Count")

            return fig, None

        else:
            return None, f"Unsupported graph type: {graph_type}"

    except Exception as e:
        return None, f"Error generating graph: {str(e)}"


def detect_graph_request(query):
    """Detect if the user is requesting a graph and extract parameters"""
    query_lower = query.lower()

    # Check for graph-related keywords
    graph_keywords = [
        "graph",
        "plot",
        "chart",
        "visualize",
        "visualization",
        "display",
        "show",
    ]

    # Check for specific chart types
    chart_types = {
        "bar": ["bar chart", "bar graph", "bar plot", "column chart"],
        "line": ["line chart", "line graph", "line plot", "trend"],
        "pie": ["pie chart", "pie graph", "pie plot", "distribution"],
        "scatter": ["scatter plot", "scatter chart", "scatter graph", "correlation"],
        "histogram": ["histogram", "distribution"],
    }

    # If no graph keywords are present, return None
    if not any(keyword in query_lower for keyword in graph_keywords):
        return None

    # Determine the chart type
    detected_type = None
    for chart_type, type_keywords in chart_types.items():
        if any(keyword in query_lower for keyword in type_keywords):
            detected_type = chart_type
            break

    # Default to bar chart if no specific type detected
    if not detected_type:
        detected_type = "bar"

    # Extract potential x and y parameters
    x_param = None
    y_param = None
    title = None

    # Look for common column names in the query
    if "dataframe_index" in st.session_state:
        columns = st.session_state.dataframe_index["metadata"]["columns"]

        # Check for pattern: "by [column]" to determine x-axis
        by_match = re.search(r"by\s+(\w+)", query_lower)
        if by_match:
            by_column = by_match.group(1)
            # Match to closest column name
            for col in columns:
                if by_column in col.lower():
                    x_param = col
                    break

        # Check for pattern: "[column] over time" to determine y-axis
        over_match = re.search(r"(\w+)\s+over\s+time", query_lower)
        if over_match:
            over_column = over_match.group(1)
            # Match to closest column name
            for col in columns:
                if over_column in col.lower():
                    y_param = col
                    x_param = "date"  # Default x to date for time series
                    break

        # If x is still not determined, look for mentions of columns
        if not x_param:
            for col in columns:
                if col.lower() in query_lower:
                    if not x_param:
                        x_param = col
                    elif not y_param and col != x_param:
                        y_param = col
                        break

        # Check for specific words representing count/frequency
        count_keywords = ["count", "frequency", "how many", "number of"]
        if any(keyword in query_lower for keyword in count_keywords):
            y_param = "count"

        # If line chart and no x specified, default to date
        if detected_type == "line" and not x_param:
            if "date" in columns:
                x_param = "date"
            elif "year_week" in columns:
                x_param = "year_week"

        # Default to type for x and sentiment_score for y if still not determined
        if not x_param and "type" in columns:
            x_param = "type"
        if not y_param and "sentiment_score" in columns and detected_type != "pie":
            y_param = "sentiment_score"

        # Try to extract a title from the query
        title_match = re.search(
            r"(?:titled|named|called)\s+['\"]([^'\"]+)['\"]", query_lower
        )
        if title_match:
            title = title_match.group(1)

    # Return the detected parameters
    return {"type": detected_type, "x": x_param, "y": y_param, "title": title}
