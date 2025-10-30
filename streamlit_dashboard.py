import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Set Streamlit page configuration
st.set_page_config(
    page_title="Music Track Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. Load Data Function ---
@st.cache_data
def load_data(file_path):
    """Loads the CSV data and performs initial cleanup."""
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"Error: The data file '{file_path}' was not found in the repository. Please ensure it is uploaded.")
        return pd.DataFrame()

# --- 2. Data Cleaning and Preparation ---
def prepare_data(df):
    """Cleans up and prepares data for visualization."""
    if df.empty:
        return df

    # Replace empty strings/whitespace with NaN for accurate non-empty counting
    df = df.replace(r'^\s*$', np.nan, regex=True)
    
    # Standardize 'combined_key'
    if 'combined_key' in df.columns:
        df['combined_key'] = df['combined_key'].replace(r'^\s*$', np.nan, regex=True)

    # Convert counts to integer
    for col in ['viewCount', 'likeCount', 'commentCount', 'popularity']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].astype('Int64', errors='ignore')
            
    return df

# --- 3. Visualization Functions ---

def generate_pie_chart(data, column_name):
    """Generates a Plotly Pie Chart for a given column."""
    data_filtered = data.dropna(subset=[column_name])
    
    if data_filtered.empty:
        st.warning(f"No non-empty data available for **{column_name}** to generate a pie chart.")
        return

    value_counts = data_filtered[column_name].value_counts().reset_index()
    value_counts.columns = [column_name, 'Count']
    total = value_counts['Count'].sum()

    fig = px.pie(
        value_counts, 
        values='Count', 
        names=column_name, 
        title=f'Distribution of **{column_name}** (N={total})',
        hole=0.3, 
        color_discrete_sequence=px.colors.qualitative.D3
    )
    
    fig.update_traces(
        textinfo='percent+label',
        hovertemplate='%{label}: %{value} entries (<extra>%{percent}</extra>)'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def generate_top_tracks_table(data, sort_column):
    """Generates a table of the top 50 tracks based on a count column."""
    data_filtered = data.dropna(subset=[sort_column])
    
    if data_filtered.empty:
        st.warning(f"No non-empty data available for **{sort_column}** to generate the table.")
        return

    display_cols = ['track_name', 'artist_credit_name', 'album_title', sort_column]
    final_cols = [col for col in display_cols if col in data_filtered.columns]
    
    top_tracks = data_filtered[final_cols] \
        .sort_values(by=sort_column, ascending=False) \
        .head(50) \
        .reset_index(drop=True)

    st.subheader(f"🏆 Top 50 Tracks by **{sort_column}**")
    
    st.dataframe(
        top_tracks.style.format({
            sort_column: "{:,.0f}" 
        }), 
        use_container_width=True, 
        height=750
    )

# --- 5. Song Details Page Function ---
def show_song_details(df):
    """Allows selection of a song and displays its full details."""
    st.title("🔍 Individual Song Details")
    
    # Create a unique identifier for each track for the selection box
    # Using track_name and artist_credit_name for better disambiguation
    df['display_name'] = df['track_name'] + ' - ' + df['artist_credit_name'].fillna('Unknown Artist')
    
    # Allow selection
    selected_track_name = st.selectbox(
        "Select a Song to View Full Details:", 
        options=sorted(df['display_name'].unique())
    )

    if selected_track_name:
        # Filter the DataFrame to the selected song
        selected_row = df[df['display_name'] == selected_track_name].iloc[0]
        
        st.markdown("---")
        
        # Display the main track information
        st.header(selected_row['track_name'])
        st.subheader(f"Artist: {selected_row['artist_credit_name']}")
        
        st.markdown("### All Feature Details")
        
        # Convert the Series (row) into a DataFrame suitable for displaying
        details_df = selected_row.drop('display_name').reset_index()
        details_df.columns = ['Feature', 'Value']
        
        # Filter to only show non-null values for a clean view
        details_df = details_df.dropna(subset=['Value'])
        
        # Display the details table
        st.dataframe(
            details_df, 
            use_container_width=True, 
            hide_index=True,
            height=600 # Set a fixed height
        )

# --- 6. Dashboard Page Function ---
def show_dashboard(df_filtered):
    """Displays the main visualization dashboard."""
    
    st.header(f"General Dashboard (Analyzing {len(df_filtered)} rows)")
    st.markdown("---")
    
    
    # --- PIE CHARTS SECTION ---
    st.header("Pie Chart Analysis: Categorical Features")
    
    pie_chart_cols = [
        'super_theme', 
        'genre_ros', 
        'timbre', 
        'danceability',
        'combined_key'
    ]
    
    all_cols = df_filtered.columns.tolist()
    mood_cols = [col for col in all_cols if col.startswith('mood_')]
    ai_cols = [col for col in all_cols if col.startswith('ai_')]
    
    pie_chart_cols.extend(mood_cols)
    pie_chart_cols.extend(ai_cols)
    
    pie_chart_cols = sorted(list(set(pie_chart_cols)))
    if 'ai_notes' in pie_chart_cols: pie_chart_cols.remove('ai_notes')
    if 'lyrics_text' in pie_chart_cols: pie_chart_cols.remove('lyrics_text')

    num_cols = 3
    cols = st.columns(num_cols)
    for i, col_name in enumerate(pie_chart_cols):
        try:
            with cols[i % num_cols]:
                generate_pie_chart(df_filtered, col_name)
        except KeyError:
            st.warning(f"Column '{col_name}' missing from the dataset.")

    st.markdown("---")

    
    # --- TOP TRACKS TABLES SECTION ---
    st.header("Top 50 Tracks: Quantitative Measures")
    
    top_track_cols = ['popularity', 'viewCount', 'likeCount', 'commentCount']
    
    num_cols_tables = 2
    table_cols = st.columns(num_cols_tables)
    
    for i, col_name in enumerate(top_track_cols):
        try:
            with table_cols[i % num_cols_tables]:
                generate_top_tracks_table(df_filtered, col_name)
        except KeyError:
             st.warning(f"Column '{col_name}' missing from the dataset.")


# --- 7. Main App Logic ---

def main():
    st.title("🎶 Music Track Feature Analysis")

    # Load and Prepare Data
    FILE_NAME = 'final_modified_tracks.csv'
    df = load_data(FILE_NAME)
    df = prepare_data(df)

    if df.empty:
        return

    # --- Sidebar Filters ---
    st.sidebar.header("Data Filters (Applies to Dashboard)")
    
    # Filter for Combined Key
    available_keys = df['combined_key'].dropna().unique()
    selected_keys = st.sidebar.multiselect(
        "Filter by Combined Key", 
        options=available_keys, 
        default=[]
    )

    df_filtered = df
    if selected_keys:
        df_filtered = df[df['combined_key'].isin(selected_keys)]
        st.sidebar.info(f"Filtered to **{len(df_filtered)}** rows based on Combined Key selection.")

    # --- Tabbed Interface ---
    tab_dashboard, tab_details = st.tabs(["📊 Main Dashboard", "🎵 Song Details"])

    with tab_dashboard:
        show_dashboard(df_filtered)

    with tab_details:
        # The Song Details tab uses the full, unfiltered dataframe for selection
        show_song_details(df)


if __name__ == "__main__":
    main()
