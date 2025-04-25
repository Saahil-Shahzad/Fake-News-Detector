# importing necessary libraries
import streamlit as st
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os

# downloading required nltk resources
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

# loading the trained logistic regression model
model_path = os.path.join(os.path.dirname(__file__), "logreg_model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

# loading the tf-idf vectorizer used during model training
vectorizer_path = os.path.join(os.path.dirname(__file__), "tfidf_vectorizer.pkl")
with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)

# defining a function to preprocess the input text
def preprocess_text(text):
    # converting text to lowercase
    text = text.lower()
    # removing digits from the text
    text = re.sub(r'\d+', '', text)
    # removing punctuation from the text
    text = text.translate(str.maketrans('', '', string.punctuation))
    # tokenizing the text into words
    tokens = word_tokenize(text)
    # getting the list of stopwords
    stop_words = set(stopwords.words('english'))
    # filtering out stopwords from the token list
    filtered = [w for w in tokens if w not in stop_words]
    # joining the filtered words back into a string
    return ' '.join(filtered)

# setting the title of the streamlit app
st.title("Fake News Detector")

# creating a text area for user input
user_input = st.text_area("Enter News Article Text:")

# when the 'Classify' button is clicked
if st.button("Classify"):
    # checking if the input text is empty
    if not user_input.strip():
        st.warning("Please enter some text to classify.")
    else:
        
        try:
            # preprocessing the input text
            cleaned_input = preprocess_text(user_input)
        except LookupError:    
            nltk.download('punkt_tab')

        # transforming the preprocessed text using the loaded vectorizer
        vectorized_input = vectorizer.transform([cleaned_input])
        # predicting the class (fake or real) using the loaded model
        prediction = model.predict(vectorized_input)[0]
        # displaying the prediction result
        label = "REAL" if prediction == 1 else "FAKE"
        st.write(f"### This news article is **{label}**.")
