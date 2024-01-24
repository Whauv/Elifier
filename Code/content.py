import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

# Load CSV Files
# neutral_words = pd.read_csv('neutral_words.csv')['word'].tolist()
# abusive_words = pd.read_csv('abusive_words.csv')['word'].tolist()
offensive_words = pd.read_csv('offensive_words.csv')['word'].tolist()

def classify_input(input_text, offensive_words):
    input_text = input_text.lower()
    words = word_tokenize(input_text)
    words = [word for word in words if word not in stopwords.words('english') and word.isalnum()]

    # neutral_count = sum(1 for word in words if word in neutral_words)
    # abusive_count = sum(1 for word in words if word in abusive_words)
    offensive_count = sum(1 for word in words if word in offensive_words)

    if offensive_count > 0:
        return "Offensive"
    # elif abusive_count > neutral_count and abusive_count > offensive_count:
    #     return "Abusive"
    else:
        return "Neutral"

# User Input (You can change this input for testing)
user_input = ""

# Perform Classification
classification = classify_input(user_input, offensive_words)

# Display Classification Result
print(f"Classification Result: {classification}")