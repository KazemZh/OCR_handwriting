import streamlit as st
import pandas as pd
import numpy as np
import os
import io

# Plotting and Visualization
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import itertools

from PIL import Image

from nltk.corpus import stopwords
import nltk

# Streamlit title
st.title("Handwritten Recognition using Deep Learning and OCR Approach")

# Define a function to load data and cache it
@st.cache_data
def load_data():
    # Define the dataset path
    dataset_path = '../words'  # Update this path based on your specific directory structure

    # Initialize data list and load transcription file
    data = []  # List to store dataset information
    words = open("../ascii/words.txt", "r").readlines()  # Read the words.txt file which contains metadata for each image
    inexistent_or_corrupted = 0  # Counter for inexistent or corrupted image files

    # Iterate over each line of the metadata file
    for line in words:
        if line.startswith("#"):  # Skip header lines that start with '#'
            continue

        # Split the line into different components
        parts = line.strip().split()
        fixed_part = parts[:8]  # Extract fixed part of the metadata (e.g., ID, coordinates, etc.)
        transcription_part = ' '.join(parts[8:])  # Extract the transcription part (actual word)

        # Construct folder paths and image file names
        folder_parts = parts[0].split('-')
        folder1 = folder_parts[0]
        folder2 = folder_parts[0] + '-' + folder_parts[1]
        file_name = parts[0] + ".png"

        # Construct the relative path for the image
        rel_path = os.path.join(dataset_path, folder1, folder2, file_name)

        # Check if the image exists and is not corrupted (file size > 0)
        if os.path.exists(rel_path) and os.path.getsize(rel_path) > 0:
            # Append metadata and image path to data list
            data.append(fixed_part + [transcription_part, rel_path])
        else:
            # Increment counter if image is missing or corrupted
            inexistent_or_corrupted += 1

    # Create a DataFrame from the collected data
    df = pd.DataFrame(data, columns=['line_id', 'result', 'graylevel', 'x', 'y', 'w', 'h', 'annotation', 'transcription', 'image_path'])

    return df

# Load the data
df = load_data()

# Streamlit sidebar navigation
st.sidebar.title("Table of contents")
pages = ["Exploration", "Data Visualization", "Modeling"]
page = st.sidebar.radio("Go to", pages)

# Display content based on the selected page
if page == pages[0]:
    st.write("### Presentation of Data")
    st.dataframe(df.head(10))
    st.write(f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
    # Show number of missing values per column
    st.write("### Missing Values")
    missing_values = df.isnull().sum()
    st.write(missing_values)
    st.write("### Data Summary Statistics")
    st.write(df.describe())
    # Show data types of each column
    st.write("### Data Types")
    st.write(df.dtypes)
    st.write("### Random Data Sample")
    st.dataframe(df.sample(10))

# Page 1: Data Visualization
if page == pages[1]:
    st.title("Data Visualization")


    # --- Plot 1: Frequency of Repetitions of Transcriptions ---
    st.write("### Frequency of Repetitions of Transcriptions")

    # Prepare the data for Plotly
    transcription_counts = df['transcription'].value_counts()
    frequency_counts = transcription_counts.value_counts().sort_index()

    # Create a figure using Plotly
    fig1 = go.Figure()

    # Add scatter trace for the data points
    fig1.add_trace(go.Scatter(
        x=frequency_counts.index, 
        y=frequency_counts.values, 
        mode='markers+lines',
        marker=dict(size=6, color='red'),
        name='Frequency of Transcriptions',
        hoverinfo='x+y'
    ))

    # Update axis labels and title
    fig1.update_layout(
        #title='Frequency of Repetitions of Transcriptions',
        xaxis=dict(
            title='Number of Occurrences of Transcription',
            type='log',  # Initially set to log scale
        ),
        yaxis=dict(
            title='Frequency (Count of Transcriptions)',
            type='log',  # Initially set to log scale
        ),
        hovermode='closest'
    )

    # Add buttons to switch between log and linear scales
    fig1.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=list([
                    dict(
                        args=[{"xaxis.type": "log", "yaxis.type": "log"}],
                        label="Log-Log Scale",
                        method="relayout"
                    ),
                    dict(
                        args=[{"xaxis.type": "linear", "yaxis.type": "linear"}],
                        label="Linear Scale",
                        method="relayout"
                    ),
                    dict(
                        args=[{"xaxis.type": "log", "yaxis.type": "linear"}],
                        label="Log X - Linear Y",
                        method="relayout"
                    ),
                    dict(
                        args=[{"xaxis.type": "linear", "yaxis.type": "log"}],
                        label="Linear X - Log Y",
                        method="relayout"
                    )
                ]),
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.11,
                xanchor="left",
                y=1.15,
                yanchor="top"
            ),
        ]
    )

    # Display the plot using Streamlit
    st.plotly_chart(fig1, use_container_width=True)

    # --- Filtering Transcriptions ---
    # Set filtering parameters
    min_samples = 100
    max_samples = 200

    # Filter transcriptions based on count thresholds
    class_counts = df['transcription'].value_counts()
    classes_to_keep = class_counts[(class_counts >= min_samples) & (class_counts <= max_samples)].index
    df_filtered = df[df['transcription'].isin(classes_to_keep)].copy()
    df_filtered.reset_index(drop=True, inplace=True)

    # Remove stopwords
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    stop_words.update([')', ':', '...', "'s"])
    df_filtered = df_filtered[~df_filtered['transcription'].isin(stop_words)]

    # Display the number of unique transcriptions remaining
    st.write(f"Number of unique transcriptions with samples between 100 and 200: {df_filtered['transcription'].nunique()}")
    # --- Plot 2 and Unique Transcriptions Side by Side ---
    st.write("### Counts of Each Transcription of Unique Transcriptions Remaining after filtering")

    # Plot 2: Counts of Each Transcription in the first column
    # Create a DataFrame for plotting purposes
    transcription_counts_df = df_filtered['transcription'].value_counts().reset_index()
    transcription_counts_df.columns = ['Transcription', 'Count']

    # Create an interactive bar plot using Plotly
    fig2 = px.bar(
        transcription_counts_df, 
        x='Transcription', 
        y='Count', 
        labels={'Transcription': 'Transcription', 'Count': 'Count'},
        hover_data=['Count'],  # Show the count on hover
        template='plotly_white'  # Use a clean style for better visuals
    )

    # Customize the layout for better visibility
    fig2.update_layout(
        xaxis_tickangle=-45,  # Rotate x-axis labels for better readability
        #title_x=0.5,  # Center the title
        #title_font_size=16,  # Increase the title font size
        xaxis_title_font_size=14,  # Increase x-axis label font size
        yaxis_title_font_size=14  # Increase y-axis label font size
    )

    # Display the interactive bar plot using Streamlit
    st.plotly_chart(fig2, use_container_width=True)

# --- Display 6 Random Images from Dataset ---
    st.write("### Sample Images from the Filtered Dataset")

    # Select 6 random images from the filtered dataset
    if len(df_filtered) > 6:
        sample_images = df_filtered.sample(6)
    else:
        sample_images = df_filtered

    # Display images in 3 columns
    cols = st.columns(3)  # Create 3 columns

    for index, row in sample_images.iterrows():
        image_path = row['image_path']
        image = Image.open(image_path)

        # Assign each image to a column in a round-robin fashion
        col_index = index % 3
        with cols[col_index]:
            st.image(image, caption=f"Transcription: {row['transcription']}", use_container_width=True)