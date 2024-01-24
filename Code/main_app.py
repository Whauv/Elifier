import streamlit as st
from PIL import Image
import time
import pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Load CSV Files
offensive_words = pd.read_csv('offensive_words.csv')['word'].tolist()

def classify_input(input_text, offensive_words):
    input_text = input_text.lower()
    words = word_tokenize(input_text)
    words = [word for word in words if word not in stopwords.words('english') and word.isalnum()]

    offensive_count = sum(1 for word in words if word in offensive_words)

    if offensive_count > 0:
        return "Offensive"
    else:
        return "Neutral"

# Define custom CSS for styling including the background image
custom_css = """
<style>
body {
    background-image: url('your_background_image.jpg');  /* Replace with your background image URL or path */
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    background-position: center center;
    background-color: #f5f5f5; /* Fallback color in case the image is not available */
    font-family: Arial, sans-serif;
    text-align: center;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100vh;
    margin: 0;
}

.title {
    font-size: 36px;
    color: #0071e3;
    margin-bottom: 20px;
}

.header {
    font-size: 24px;
    color: #333;
    margin-top: 20px;
}

.text-area {
    width: 80%;
    padding: 10px;
    border: 2px solid #0071e3;
    border-radius: 5px;
    transition: border-color 0.3s ease-in-out;
}

.text-area:focus {
    border-color: #ff6f61;
    outline: none;
}

.image-container {
    max-width: 80%;
    margin-top: 20px;
    border-radius: 5px;
    box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.2);
    transition: transform 0.3s ease-in-out;
}

.image-container:hover {
    transform: scale(1.05);
}
</style>
"""

# Display custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

# Load your trained ML model using pickle
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)  # Replace 'model.pkl' with the actual path to your trained model

# Define a dictionary to map numerical labels to class names
label_map = {0: 'Abusive', 1: 'Neutral', 2: 'Hate'}

# Title for the app
st.title("ELIFIER")
    
# Create a form to enter text
with st.form("text_form"):
    st.markdown("<div class='title'>Text Input</div>", unsafe_allow_html=True)
    user_input = st.text_area("Enter some text here:", key="text")
    submit_text = st.form_submit_button("Submit Text")

# Create a form to upload an image
with st.form("image_form"):
    st.markdown("<div class='title'>Image Upload</div>", unsafe_allow_html=True)
    image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"], key="image")
    submit_image = st.form_submit_button("Submit Image")

# Display a loading spinner while processing
if (submit_text or submit_image) and (image is None or user_input is None):
    with st.spinner("Processing..."):
        time.sleep(2)  # Simulate some processing time

# Display the uploaded image (if any)
if image is not None and submit_image:
    st.markdown("<div class='header'>Uploaded Image:</div>", unsafe_allow_html=True)
    img = Image.open(image)
    st.image(img, caption="Uploaded Image", use_column_width=True)

# Display the uploaded text (if any)
if user_input and submit_text:
    st.markdown("<div class='header'>Uploaded Text:</div>", unsafe_allow_html=True)
    st.markdown(user_input)

    # Perform Classification
    classification = classify_input(user_input, offensive_words)
    st.markdown(f"<div class='header'>Classification Result: {classification}</div>", unsafe_allow_html=True)
