from flask import Flask, request, render_template
import tensorflow
import numpy as np
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding
from transformers import TFAutoModelForSeq2SeqLM, PegasusTokenizer
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.layers import Bidirectional,GRU,concatenate,SpatialDropout1D
from keras.layers import GlobalMaxPooling1D,GlobalAveragePooling1D,Conv1D
from keras.models import Model
from keras.layers import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.layers import Concatenate
import matplotlib.pyplot as plt
from keras import layers
from keras.optimizers import Adam,SGD,RMSprop


app = Flask(__name__)
max_len = 100
max_features = 10000
embed_size = 300

# Function to perform text paraphrasing using PEGASUS
def paraphrase_text(text):
# Load pre-trained model and tokenizer
    #model_path= 'tf_model.h5'
    model_name= 'tuner007/pegasus_paraphrase'
    #model = tensorflow.keras.models.load_model(model_path)
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name, from_pt=True)
    input_ids = tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True)
    paraphrased_output = model.generate(input_ids, max_length=1024, num_beams=5, early_stopping=True)
    paraphrased_text = tokenizer.decode(paraphrased_output[0], skip_special_tokens=True)
    return paraphrased_text

# Function to preprocess input text
def preprocess_text(text):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])
    sequence = tokenizer.texts_to_sequences([text])
    sequence = pad_sequences(sequence, maxlen=max_len)
    #return sequence
    return np.squeeze(sequence, axis=0)

# Function to predict the category
def predict_category(text):
    # Load the pre-trained model
    model = tensorflow.keras.models.load_model('model_jinum.h5')

    # Preprocess the input text
    processed_text = preprocess_text(text)

    # Use the model to predict the category
    prediction = model.predict(np.array([processed_text]))

    # Get the indices of the top 2 highest probabilities
    top_2_indices = prediction[0].argsort()[-2:][::-1]

    # Get the corresponding categories
    categories = ['hate', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    top_2_categories = [categories[idx] for idx in top_2_indices]

    return top_2_categories


@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         text = request.form['text']
#         predicted_category = predict_category(text)
#         return render_template('index.html', text=text, predicted_category="hello this is a test hadhashdoahsdohao")
#     except Exception as e:
#         return f"An error occurred: {str(e)}"
    
@app.route('/predict', methods=['GET', 'POST'])
def paraphrase():
    try:
        if request.method == 'POST':
            text = request.form['text']
            paraphrased_text = paraphrase_text(text)
            predicted_category = predict_category(text)
            return render_template('index.html', text=text, paraphrased_text=paraphrased_text, predicted_category=predicted_category)
    except Exception as e:
        return f"An error occurred: {str(e)}"
    
if __name__ == '__main__':
    app.run(debug=True)
