import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from collections import Counter

# Set Streamlit page configuration
st.set_page_config(
    page_title="Music Track Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. Load Data Function ---
@st.cache_data
def load_data(file_path):
    """Loads the CSV data."""
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"Error: The data file '{file_path}' was not found in the repository. Please ensure it is uploaded.")
        return pd.DataFrame()

# --- 2. Data Cleaning and Preparation (Includes Album Consolidation) ---
@st.cache_data
def prepare_data(df):
    """Cleans up and prepares data for visualization, including album consolidation."""
    if df.empty:
        return df

    # Replace empty strings/whitespace with NaN for accurate non-empty counting
    df = df.replace(r'^\s*$', np.nan, regex=True)
    
    # Standardize 'combined_key'
    for col in ['combined_key', 'album_title']:
        if col in df.columns:
            df[col] = df[col].replace(r'^\s*$', np.nan, regex=True)

    # Convert counts to integer
    for col in ['viewCount', 'likeCount', 'commentCount', 'popularity']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].astype('Int64', errors='ignore')
            
    # --- ALBUM CONSOLIDATION LOGIC ---
    if 'album_title' in df.columns:
        non_null_albums = df['album_title'].dropna().astype(str)
        
        # 1. Global Sub-content Frequency Count
        all_sub_contents = []
        for title in non_null_albums.unique():
            parts = [part.strip() for part in title.split('|')]
            valid_parts = [part for part in parts if part]
            all_sub_contents.extend(valid_parts)
        sub_content_counts = Counter(all_sub_contents)

        # 2. Create Mapping
        title_mapping = {}
        for original_title in non_null_albums.unique():
            sub_contents = [part.strip() for part in original_title.split('|')]
            valid_sub_contents = [part for part in sub_contents if part]
            
            if valid_sub_contents:
                # Find the sub-content with the highest global frequency count
                consolidated_title = max(valid_sub_contents, key=lambda x: sub_content_counts[x])
            else:
                consolidated_title = original_title 
            title_mapping[original_title] = consolidated_title

        # 3. Apply Mapping
        df['consolidated_album_title'] = df['album_title'].map(title_mapping)
    # --- END ALBUM CONSOLIDATION ---

    return df

# --- 3. Visualization Helper Functions ---

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

# --- 4. New Album Dashboard Function ---
def show_new_album_dashboard(df):
    """Displays a detailed dashboard for the '屬於' album."""
    ALBUM_NAME = "屬於"
    
    if 'consolidated_album_title' not in df.columns:
        st.error("Error: Album consolidation failed. Cannot filter by consolidated title.")
        return

    df_album = df[df['consolidated_album_title'] == ALBUM_NAME].copy()

    if df_album.empty:
        st.warning(f"Album '{ALBUM_NAME}' not found in the dataset after consolidation.")
        return

    st.title(f"🎵 New Album Analysis: **{ALBUM_NAME}**")
    st.subheader(f"Analyzing {len(df_album)} tracks from this album.")
    
    # 1. Album Track Listing
    st.markdown("### Album Track Listing & Features")
    track_list_cols = ['track_name', 'artist_credit_name', 'popularity', 'viewCount', 'ai_sentiment', 'combined_key']
    
    st.dataframe(
        df_album[track_list_cols].sort_values(by='popularity', ascending=False).reset_index(drop=True),
        use_container_width=True
    )
    
    st.markdown("---")
    
    # 2. Detailed Distribution Charts
    st.markdown("### Detailed Feature Distribution")
    
    detail_cols = ['normalized_key', 'mood_sad', 'ai_theme', 'genre_ros']
    
    cols = st.columns(2)
    for i, col_name in enumerate(detail_cols):
        try:
            with cols[i % 2]:
                generate_pie_chart(df_album, col_name)
        except KeyError:
             st.warning(f"Column '{col_name}' missing from the dataset.")

# --- 5. Song Details Page Function (RESTORED/REFINED) ---
def show_song_details(df):
    """Allows selection of a song and displays its full details."""
    st.title("🔍 Individual Song Details")
    
    # Create a unique identifier for each track for the selection box
    df['display_name'] = df['track_name'] + ' - ' + df['artist_credit_name'].fillna('Unknown Artist')
    
    # Allow selection
    selected_track_name = st.selectbox(
        "Select a Song to View Full Details:", 
        options=sorted(df['display_name'].unique())
    )

    if selected_track_name:
        # Filter the DataFrame to the selected song
        # Use .copy() to avoid SettingWithCopyWarning if further changes were needed
        selected_row = df[df['display_name'] == selected_track_name].iloc[0]
        
        st.markdown("---")
        
        # Display the main track information
        st.header(selected_row['track_name'])
        st.subheader(f"Artist: {selected_row['artist_credit_name']}")
        
        st.markdown("### All Feature Details")
        
        # Convert the Series (row) into a DataFrame suitable for displaying
        details_df = selected_row.drop('display_name', errors='ignore').reset_index()
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

    # Load and Prepare Data (Consolidation happens here)
    FILE_NAME = 'final_modified_tracks.csv'
    df = load_data(FILE_NAME)
    df = prepare_data(df)

    if df.empty:
        return

    # --- Sidebar Filters ---
    st.sidebar.header("Data Filters (Applies to Dashboard)")
    
    # Filter for Combined Key
    available_keys = df['combined_key'].dropna().unique() if 'combined_key' in df.columns else []
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
    tab_dashboard, tab_album, tab_details = st.tabs(["📊 Main Dashboard", "💿 New Album: 屬於", "🎵 Song Details"])

    with tab_dashboard:
        show_dashboard(df_filtered)

    with tab_album:
        show_new_album_dashboard(df) # Use the full DF for album specific analysis

    with tab_details:
        show_song_details(df) # <-- THIS FUNCTION IS NOW DEFINED ABOVE


if __name__ == "__main__":
    main()
