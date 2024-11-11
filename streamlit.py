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

    # Print the count of inexistent or corrupted images to Streamlit
    st.write('Number of inexistent or corrupted files:', inexistent_or_corrupted)

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