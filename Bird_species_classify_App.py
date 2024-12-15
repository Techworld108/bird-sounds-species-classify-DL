import os
import tempfile
import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
# from tensorflow.image import resize

# Load the model once to save time
@st.cache_resource()
def load_model():
    model = tf.keras.models.load_model("Trained_model.keras")
    return model

# Load and preprocess audio data
def load_and_preprocess_data(file_path, target_shape=(150, 150)):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr=None)

    # Define the duration of each chunk and overlap
    chunk_duration = 4  # seconds
    overlap_duration = 2  # seconds

    # Convert durations to samples
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate

    # Calculate the number of chunks
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1

    # Process each chunk
    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        chunk = audio_data[start:end]

        # Calculate Mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
        
        # Convert to TensorFlow tensor
        mel_spectrogram_tensor = tf.convert_to_tensor(mel_spectrogram, dtype=tf.float32)
        
        # Add a channel dimension (to match expected input shape for resize)
        mel_spectrogram_tensor = tf.expand_dims(mel_spectrogram_tensor, axis=-1)
        
        # Resize the spectrogram
        resized_mel_spectrogram = tf.image.resize(mel_spectrogram_tensor, target_shape)
        
        # Append the resized mel spectrogram to the data list
        data.append(resized_mel_spectrogram.numpy())  # Convert back to numpy array if needed

    return np.array(data)

# Predict using the TensorFlow model
def model_prediction(X_test):
    model = load_model()
    y_pred = model.predict(X_test)
    predicted_categories = np.argmax(y_pred, axis=1)
    unique_elements, counts = np.unique(predicted_categories, return_counts=True)
    max_elements = unique_elements[counts == np.max(counts)]
    return max_elements[0]

# Sidebar Navigation
st.sidebar.markdown(
    """
    <style>
    /* Specific title styling for "🎵 Bird Sounds Classification" */
    .sidebar-title {
        color: red !important; /* Title in red */
        font-size: 1.5em;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Add a red-colored title in the sidebar
st.sidebar.markdown('<div class="sidebar-title">Dashboard</div>', unsafe_allow_html=True)

# Sidebar Navigation Options
app_mode = st.sidebar.radio("", ["Home", "About Project", "Prediction"], index=0)


# Home Page
if app_mode == "Home":
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #181646;  /* Blue background */
            color: #ffffff;
        }
        .header {
            color: #ffcc00;
            font-weight: bold;
        }
        .subheader {
            color: #cccccc;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(''' ## Welcome to the,\n
    ## 🎶 Bird Sounds Species Classification Model''')
    st.image("bird-species.webp", caption="Identify Bird Sounds with AI", use_container_width=True)

    st.markdown(
        """
        ### **🎯 Key Features**
        - **Accuracy:** Identify bird species from sounds with state-of-the-art deep learning.
        - **User-Friendly Interface:** Seamless and intuitive navigation.
        - **Quick Results:** Get predictions instantly.
        
        ### **📖 How It Works**
        1. Upload an audio file of bird sounds.
        2. The system processes the audio using advanced spectrogram analysis.
        3. View the predicted bird species.

        Ready to explore? Go to the **Prediction** page and upload your file!
        """,
        unsafe_allow_html=True,
    )

# About Project Page
elif app_mode == "About Project":
    st.title("📚 About the Bird Sounds Project")
    st.markdown(
        """
        This project leverages deep learning to classify bird species based on their unique audio characteristics.

        ### **Dataset Highlights**
        - **Species original :**  A collection of 8 Species with 100 audio files each, all having a length of 30 seconds (the famous Kaggle dataset, the BirdCLEF 2024 of sounds)
        - **List of Species :**  asbfly, ashdro1, ashpri1, ashwoo2, asikoe2, asiope1, aspfly1, aspswi1.
        - **images original :**  A visual representation for each audio file. One way to classify data is through neural networks. 
                                Because NNs (like CNN, what we will be using today) usually take in some sort of image representation, 
                                the audio files were converted to Mel Spectrograms to make this possible.
        - **2 CSV files :**  Containing features of the audio files. One file has for each song (30 seconds long) a mean and variance 
                            computed over multiple features that can be extracted from an audio file. The other file has the same structure, 
                            but the songs were split before into 3 seconds audio files (this way increasing 10 times the amount of data 
                            we fuel into our classification models). With data, more is always better.
        - **Utility :** Suitable for researchers, bird enthusiasts, and eco-conservation efforts.

        ### **Technology Stack**
        - **TensorFlow/Keras:** For building and training the model.
        - **Librosa:** For audio processing and feature extraction.
        - **Streamlit:** For creating this interactive web app.
        """
    )

# Prediction Page
elif app_mode == "Prediction":
    st.title("🎤 Predict Bird Species from Audio")

    test_ogg = st.file_uploader("📤 Upload an audio file", type=["ogg"])
    filepath = None

    if test_ogg is not None:
        # Use a temporary file to save the uploaded file
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            tmpfile.write(test_ogg.getbuffer())
            filepath = tmpfile.name

        st.success("Audio file uploaded successfully!")

    # Play audio button
    if test_ogg and st.button("▶️ Play Audio"):
        st.audio(test_ogg)

    # Predict button
    if test_ogg and st.button("🔍 Predict Bird Species"):
        with st.spinner("Analyzing the audio..."):
            X_test = load_and_preprocess_data(filepath)
            result_index = model_prediction(X_test)
            label = ['asbfly', 'ashdro1','ashpri1','ashwoo2','asikoe2','asiope1','aspfly1','aspswi1']
            st.balloons()
            st.markdown(f"### **🎉 The bird sound belongs to the `{label[result_index]}` Species**")
