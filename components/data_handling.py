import streamlit as st


def initialize_dataframe_index(df):
    """Create an indexed version of the dataframe for faster retrieval"""
    # Convert the DataFrame to a format that can be efficiently queried
    indexed_data = {}
    for idx, row in df.iterrows():
        indexed_data[idx] = {col: row[col] for col in df.columns}

    # Add metadata for faster attribute-based queries
    metadata = {
        "total_rows": len(df),
        "columns": df.columns.tolist(),
        "unique_values": {
            col: df[col].dropna().unique().tolist()
            for col in df.columns
            if df[col].nunique() < 100 and col not in ["content", "link"]
        },
    }

    return {"data": indexed_data, "metadata": metadata}


def find_data_by_attribute(attribute, value, exact_match=False):
    """Search the indexed dataframe for rows matching attribute/value"""
    if "dataframe_index" not in st.session_state:
        return []

    index = st.session_state.dataframe_index
    results = []

    # Check if attribute exists
    if attribute not in index["metadata"]["columns"]:
        return []

    # Search through the indexed data
    for idx, row in index["data"].items():
        if attribute not in row:
            continue

        row_value = row[attribute]

        # Handle different matching logic
        if exact_match:
            if str(row_value).lower() == str(value).lower():
                results.append(row)
        else:
            # For text fields, do contains matching
            if isinstance(row_value, str) and isinstance(value, str):
                if value.lower() in row_value.lower():
                    results.append(row)
            # For numeric comparisons
            elif isinstance(row_value, (int, float)) and isinstance(
                value, (int, float)
            ):
                if row_value == value:
                    results.append(row)

    return results
