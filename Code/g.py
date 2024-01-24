import streamlit as st
from PIL import Image
import pickle

st.title("ELIFIER")
st.title("Text and Image Input App")
st.write("This app allows you to input text and upload images.")

text_input = st.text_area("Enter Text Here", "")

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

label_map = {0: 'Abusive', 1: 'Neutral', 2: 'Hate', 3:'Normal\n', 
             4:'Abusive\n', 5:'Hate\n', 6:'abusive', 
             7:'hate',8:'normal'}

if st.button("Submit"):
    if text_input:
        st.write("Text Input:")
        st.write(text_input)
        numerical_prediction = model.predict([text_input])[0]

        if numerical_prediction in label_map:
            prediction = label_map[numerical_prediction]
            st.write("Text Classification Result:")
            st.write(prediction)
        else:
            st.write("Error: Unknown prediction label")

