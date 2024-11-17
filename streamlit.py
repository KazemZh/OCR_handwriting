import streamlit as st
from streamlit_drawable_canvas import st_canvas

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
import seaborn as sns

from PIL import Image
import cv2

from nltk.corpus import stopwords
import nltk

import tensorflow as tf
from sklearn.metrics import confusion_matrix

#path to checkpoints and several data required in the script
path_to_checkpoints = "../models_check_points/" # adjust this path according to your location of models_check_points directroy

# Set the page configuration
st.set_page_config(
    page_title="Handwritten Text Recognition Project",
    page_icon="📝",
    layout="wide"  # This sets the page layout to wide mode
)

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
    df = pd.DataFrame(data, columns=['word_id', 'result', 'graylevel', 'x', 'y', 'w', 'h', 'annotation', 'transcription', 'image_path'])

    return df

# Load the data
df = load_data()

# Streamlit sidebar navigation
st.sidebar.title("Table of contents")
pages = ["Introduction", "Exploration", "Data Visualization", "Modeling", "Testing", "Key Results and Findings"]
page = st.sidebar.radio("Go to", pages)

#Page0: Introduction
if page==pages[0]:
    st.title("Handwritten Text Recognition Using Deep Learning")
    
    st.markdown("""
    ## Welcome to Our Handwritten Text Recognition Project!
    Handwritten text recognition is a critical challenge in the field of **Optical Character Recognition (OCR)**, with widespread applications across industries like **healthcare, insurance, and administration**. In these sectors, vast amounts of paperwork are still handwritten, making it imperative to develop automated systems for digitizing documents efficiently and accurately.
    
    This project aims to develop a deep learning-based OCR system capable of accurately recognizing and extracting text from handwritten documents. Our focus was to leverage the latest **deep learning** technologies, including custom **Convolutional Neural Networks (CNNs)** and **transfer learning** with **VGG16**, to overcome the challenges of recognizing diverse handwriting styles.
    """)

    st.markdown("### Project Motivation")
    st.write("""
    The primary motivation for this project is to contribute towards **automating document digitization** in industries where handwritten content is still prevalent. Converting handwritten forms into digital text can **save time**, **reduce manual errors**, and **cut costs** by eliminating the need for manual data entry.
    """)

    st.markdown("### Project Workflow and Approach")
    st.markdown("""
    1. **Exploration of Available OCR Tools**: We started by assessing existing OCR tools like **PyTesseract**, **docTR**, **EasyOCR**, and **Apache Tika**. While these tools are great for typed text, they showed significant limitations in recognizing handwritten texts.
    2. **Custom Deep Learning Models**: Given the limitations of existing OCR tools, we developed our own custom models. We began with a simple **Convolutional Neural Network (CNN)** and **LeNet**, before moving to more advanced techniques like **transfer learning** using **VGG16**.
    3. **Dataset Balancing**: The dataset was balanced to improve model performance.
    """)

    st.markdown("### Project Team")
    st.write("""
    - **Claudia Wisniewski**
    - **Kazem Zhour**
    """)
    st.markdown("**Mentor**: Yaniv")

    st.markdown("### Get Involved")
    st.write("""
    The code, models, and resources used in this project are available for collaboration on [GitHub](https://github.com/KazemZh/OCR_handwriting). We welcome contributions and feedback.
    """)

# Page1: Display content based on the selected page
if page == pages[1]:
    st.markdown("### Dataset Information")
    st.write("""
    The dataset used for this project is the **IAM Handwriting Database**.
    
    - **Dataset Link**: [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)
    - **Google Drive**: A copy of the dataset can also be found on Google Drive for easy access: [Google Drive Dataset Link](https://drive.google.com/drive/folders/188WHgj4Z4PE7l-N41XmUKNXnfux8TYbd?usp=drive_link)
    
    The dataset consists of **1,539 pages of scanned text** gathered from handwritten forms filled out by **657 writers**. These texts are then segmented into **115,000 individual labeled words**, providing a rich variety of handwriting styles for training and evaluation.
    """)
    # List of image paths to be displayed
    st.write("""Sample handwritten forms and some corresponding segmented words from the IAM Handwriting Database""")
    images_list = ['a01-000u.png', "a01-000u-00-00.png", "a01-000u-00-01.png", "a01-000u-00-02.png", "a01-000u-00-03.png", "a01-000u-00-04.png"]
    image_path = []
    for i in images_list:
        image_path.append(path_to_checkpoints+i)
    # Display all images
    # Creating columns to place the images next to each other
    col0, col1, col2, col3, col4, col5 = st.columns([10,1,4,2,4,3])
    # Displaying each image in its respective column
    with col0:
        st.image(image_path[0], width=600)
    with col1:
        st.image(image_path[1], width=50)
    with col2:
        st.image(image_path[2], width=400)
    with col3:
        st.image(image_path[3], width=100)
    with col4:
        st.image(image_path[4], width=400)
    with col5:
        st.image(image_path[5], width=300)
    
    st.write("### Construction of DataFrame")
    st.write("The DataFrame is constructed using the metadata provided by the IAM Handwriting Database and is linked to the paths of the corresponding images in the database.")

# Creating four columns for different buttons to display data information
col1, col2, col3, col4 = st.columns(4)
with col1:
    head_data = st.button("View DataFrame", key="df_head")
with col2:
    random_data = st.button("View Random Rows", key="df_rand")
with col3:
    missing_values = st.button("Check Missing Values", key="df_na")
with col4:
    data_stat = st.button("View Summary Statistics", key="df_stat")

# Handling button clicks to display relevant data
if head_data:
    col1, col2 = st.columns([2, 3])
    with col1:
        st.write("#### Metadata Overview")
        st.image(path_to_checkpoints + 'words_metadata.png', caption="Metadata Overview of the IAM Handwriting Database")
    with col2:
        st.write("#### DataFrame Head")   
        st.dataframe(df.head(10))
        st.write(f"The dataset contains **{df.shape[0]} rows** and **{df.shape[1]} columns**.")
        st.markdown("""
            This DataFrame is constructed using the metadata provided by the IAM Handwriting Database, 
            linking each word’s metadata to the path of the corresponding image in the dataset.
        """)

if random_data:
    st.write("### Random Rows from DataFrame")
    st.dataframe(df.sample(10))
    st.markdown("""
        The randomly sampled rows provide a quick overview of the diversity of handwritten text samples. 
        These random rows help in understanding the distribution of the data.
    """)

if missing_values:
    st.write("### Missing Values in DataFrame")
    missing_values_summary = df.isnull().sum()
    st.write(missing_values_summary)
    if missing_values_summary.sum() == 0:
        st.write("The DataFrame has **no missing values**, indicating data completeness.")
    else:
        st.write("There are missing values in the dataset that need to be addressed.")

if data_stat:
    st.write("### Summary Statistics of the DataFrame")
    st.write(df.describe())
    st.markdown("""
        These statistics provide key insights into the distribution of numerical columns such as the image dimensions,
        grayscale values, and bounding box properties. It helps in understanding the range, average, and standard 
        deviation of various features.
    """)


# Page 2: Data Visualization
if page == pages[2]:
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

# Page 3: Modeling
if page == pages[3]:
    st.header("Pre-Trained Engines")
    st.markdown("PyTesseract, Doctr, EasyOCR, and Apache Tika were tested on both page-level and segmented word-level handwritten text. Although these models are well-known for detecting printed text, they performed poorly with handwritten text, likely because they were not trained on datasets containing handwritten samples.")
    st.markdown("Therefore, we decided to customize our own model and train it on handwritten data from IAM Handwriting Database.")
    st.header("Customized Models on Full Data")
    tab1, tab2 = st.tabs(["CNN Model", "LeNet Model"])
    # Define a helper function to display content based on button clicks
    def display_model_info(model_name, model_path, history_file, evaluation_metrics):
        st.subheader(f"{model_name}")

        # Adding buttons side by side using columns
        col1, col2, col3 = st.columns(3)
        with col1:
            show_summary = st.button("Show Model Architecture and Summary", key=f"{model_name}_summary")
        with col2:
            show_accuracy = st.button("Show Accuracy Over Epochs", key=f"{model_name}_accuracy")
        with col3:
            show_evaluation_metrics = st.button("Show Evaluation Metrics", key=f"{model_name}_metrics")

        # Display sections based on button clicks
        if show_summary:
            st.markdown("#### Model Architecture")
            st.image(model_path[0], width=600)
            st.markdown("#### Model Summary")
            st.image(model_path[1], width=600)

        if show_accuracy:
            st.markdown("#### Accuracy over Epochs")
            # Load the history data
            df_history = pd.read_csv(history_file)
            # Create an accuracy plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_history['Epoch'], y=df_history['Accuracy'], mode='lines', name='Training Accuracy'))
            fig.add_trace(go.Scatter(x=df_history['Epoch'], y=df_history['Val_Accuracy'], mode='lines', name='Validation Accuracy'))
            fig.update_layout(
                xaxis_title='Epoch',
                yaxis_title='Accuracy',
                legend_title='Legend',
                width=700,
                yaxis=dict(range=[0, 1])  # Set y-axis limit from 0 to 1
            )
            st.plotly_chart(fig)

        if show_evaluation_metrics:
            st.markdown("#### Evaluation Metrics")
            st.write('Mean Precision =', evaluation_metrics[0])
            st.write('Mean Recall =', evaluation_metrics[1])
            st.write('Mean F1-Score =', evaluation_metrics[2])

    # Simple CNN Model Tab
    with tab1:
        display_model_info(
            model_name="CNN Model",
            model_path=[path_to_checkpoints + 'CNN.png', path_to_checkpoints + 'naive_cnn_model_summary.png'],
            history_file=path_to_checkpoints + 'Naive_CNN_training_history.csv',
            evaluation_metrics=[0.37, 0.45, 0.36]
        )

    # VGG16 Frozen Model Tab
    with tab2:
        display_model_info(
            model_name="LeNet Model",
            model_path=[path_to_checkpoints + 'LeNet.png', path_to_checkpoints + 'lenet_model_summary.png'],
            history_file=path_to_checkpoints + 'LeNet_training_history.csv',
            evaluation_metrics=[0.30, 0.34, 0.27]
        )

    st.header("Customized Models on Filtered Data")
    tab1, tab2, tab3 = st.tabs(["Simple CNN Model", "VGG16 Frozen Model", "VGG16 Unfrozen Model"])

    # Define a helper function to display content based on button clicks
    def display_model_info(model_name, model_path, history_file, conf_matrix_img):
        st.subheader(f"{model_name}")

        # Adding buttons side by side using columns
        col1, col2, col3 = st.columns(3)
        with col1:
            show_summary = st.button("Show Model Architecture and Summary", key=f"{model_name}_summary")
        with col2:
            show_accuracy = st.button("Show Accuracy Over Epochs", key=f"{model_name}_accuracy")
        with col3:
            show_confusion_matrix = st.button("Show Confusion Matrix", key=f"{model_name}_conf_matrix")

        # Display sections based on button clicks
        if show_summary:
            st.markdown("#### Model Architecture")
            st.image(model_path[0], width=800)
            st.markdown("#### Model Summary")
            model = tf.keras.models.load_model(model_path[1])
            # Capture model summary in a string buffer
            summary_str = io.StringIO()
            model.summary(print_fn=lambda x: summary_str.write(x + "\n"))
            st.text(summary_str.getvalue())

        if show_accuracy:
            st.markdown("#### Accuracy Over Epochs")
            # Load the history data
            df_history = pd.read_csv(history_file)
            # Create an accuracy plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_history['Epoch'], y=df_history['Accuracy'], mode='lines', name='Training Accuracy'))
            fig.add_trace(go.Scatter(x=df_history['Epoch'], y=df_history['Val_Accuracy'], mode='lines', name='Validation Accuracy'))
            fig.update_layout(
                xaxis_title='Epoch',
                yaxis_title='Accuracy',
                legend_title='Legend',
                width=700,
                yaxis=dict(range=[0, 1])  # Set y-axis limit from 0 to 1
            )
            st.plotly_chart(fig)

        if show_confusion_matrix:
            st.markdown("#### Confusion Matrix")
            st.image(conf_matrix_img, width=700)

    # Simple CNN Model Tab
    with tab1:
        display_model_info(
            model_name="Simple CNN Model",
            model_path=[path_to_checkpoints + 'CNN.png', path_to_checkpoints + 'CNN.h5'],
            history_file=path_to_checkpoints + 'CNN_training_history.csv',
            conf_matrix_img=path_to_checkpoints + 'cnn_model_conf_matrix.png'
        )

    # VGG16 Frozen Model Tab
    with tab2:
        display_model_info(
            model_name="VGG16 Frozen Model",
            model_path=[path_to_checkpoints + 'vgg16-freez.png', path_to_checkpoints + 'vgg16_224.h5'],
            history_file=path_to_checkpoints + 'vgg16_training_history.csv',
            conf_matrix_img=path_to_checkpoints + 'vgg16_model_conf_matrix.png'
        )

    # VGG16 Unfrozen Model Tab
    with tab3:
        display_model_info(
            model_name="VGG16 Unfrozen Model",
            model_path=[path_to_checkpoints + 'vgg16-unfreez.png', path_to_checkpoints + 'vgg16_224_unfreez_last_4.h5'],
            history_file=path_to_checkpoints + 'vgg16_unfreez_training_history.csv',
            conf_matrix_img=path_to_checkpoints + 'vgg16_unfreez_model_conf_matrix.png'
        )
    

@st.cache_resource
def load_models():
    # Load all three models: Simple CNN, VGG16 frozen, and VGG16 unfrozen
    simple_cnn = tf.keras.models.load_model(path_to_checkpoints+'CNN.h5')
    vgg16_frozen = tf.keras.models.load_model(path_to_checkpoints+'vgg16_224.h5')
    vgg16_unfrozen = tf.keras.models.load_model(path_to_checkpoints+'vgg16_224_unfreez_last_4.h5')
    return simple_cnn, vgg16_frozen, vgg16_unfrozen

# Function to preprocess the uploaded image
def process_image(image, model_type):
    # Read the image from the uploaded file
    image = Image.open(image)

    if model_type == "Simple CNN Model (Grayscale)":
        # Convert to grayscale and resize for Simple CNN (28x28)
        image = image.convert('L')
        image = image.resize((28, 28))
        image_array = np.array(image) / 255.0
        # Reshape to match the input format expected by the model
        return image_array.reshape(1, 28, 28, 1)

    else:
        # Assuming the model expects RGB images resized to 224x224 (as in VGG16)
        image = image.convert('RGB')
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        # Reshape to match the input format expected by the model
        return image_array.reshape(1, 224, 224, 3)

# Page4: Testing
from streamlit_drawable_canvas import st_canvas
import cv2

# Page4: Testing
# Page4: Testing
if page == pages[4]:
    st.title("Testing the models")
    st.write("Upload an image of a handwritten word or draw directly on the canvas below and let a model you choose predict the word.")
    st.write("Available words for prediction: ['A', 'And', 'But', 'In', 'This', 'We', 'You', 'could', 'first', 'like', 'made', 'man', 'may', 'much', 'new', 'people', 'time', 'told', 'two', 'well']")

    # Load models
    simple_cnn, vgg16_frozen, vgg16_unfrozen = load_models()

    # File upload widget
    uploaded_file = st.file_uploader("Upload an image (JPG or PNG)", type=["jpg", "jpeg", "png"])

    # Create a canvas component for drawing
    st.write("Or draw your word below:")
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",  # Fixed fill color with opacity
        stroke_width=10,
        stroke_color="black",
        background_color="white",
        width=224,
        height=224,
        drawing_mode="freedraw",
        key="canvas"
    )

    # Process uploaded image or canvas drawing
    image_to_predict = None

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", width=300)
        image_to_predict = uploaded_file
    elif canvas_result.image_data is not None:
        # If the canvas has a drawing, process it
        st.image(canvas_result.image_data, caption="Drawn Image", width=300)
        # Convert canvas image to uint8 and remove alpha channel
        image_to_predict = cv2.cvtColor(canvas_result.image_data.astype(np.uint8), cv2.COLOR_RGBA2RGB)

    if image_to_predict is not None:
        # Select which model to use for prediction
        model_choice = st.selectbox("Select the model to use for prediction:", 
                                    ["Simple CNN Model", "VGG16 Frozen Model", "VGG16 Unfrozen Model"])

        # Predict button
        if st.button("Predict"):
            # Process the uploaded image or drawn canvas based on the selected model
            if isinstance(image_to_predict, np.ndarray):
                # If it's from the canvas
                if model_choice in ["VGG16 Frozen Model", "VGG16 Unfrozen Model"]:
                    # Resize and normalize for VGG16 models
                    processed_image = cv2.resize(image_to_predict, (224, 224)) / 255.0
                elif model_choice == "Simple CNN Model":
                    # Convert to grayscale for Simple CNN
                    processed_image = cv2.cvtColor(image_to_predict, cv2.COLOR_RGB2GRAY)
                    processed_image = cv2.resize(processed_image, (28, 28)) / 255.0
                    processed_image = np.expand_dims(processed_image, axis=-1)  # Add channel dimension
            else:
                # If it's an uploaded file, process it accordingly
                processed_image = process_image(image_to_predict, model_choice)

            processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension

            # Select the correct model based on user choice
            if model_choice == "Simple CNN Model":
                model = simple_cnn
            elif model_choice == "VGG16 Frozen Model":
                model = vgg16_frozen
            else:
                model = vgg16_unfrozen

            # Make the prediction
            predictions = model.predict(processed_image)
            predicted_class_index = np.argmax(predictions)

            # List of labels (words)
            labels = ['A', 'And', 'But', 'In', 'This', 'We', 'You', 'could', 'first', 'like', 'made', 'man', 'may', 'much', 'new', 'people', 'time', 'told', 'two', 'well']
            
            # Display the predicted word
            st.write(f"Predicted word: **{labels[predicted_class_index]}**")

# Page4: Key Results and Findings
if page == pages[5]:
    st.write("### Key Results and Findings")
    st.markdown("""
    - **Initial Attempts with Pre-trained Models**: Existing OCR tools were ineffective for handwritten text, with a **word error rate (WER)** of up to **50%**.
    - **Custom CNN and LeNet Models**: Developed custom CNN and LeNet models to improve accuracy but still faced issues with class imbalance.
    - **Balanced Dataset and Transfer Learning**: By limiting the dataset to words with between **100-200 samples**, accuracy improved significantly to over **80%**. Using **transfer learning with VGG16** (unfreezing the last 4 layers) pushed the final accuracy to **90%**, which was the best-performing model.
    """)