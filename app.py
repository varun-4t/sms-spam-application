# Import necessary libraries
import nltk
nltk.download('punkt')  # Tokenization support
nltk.download('stopwords')  # Stopwords for text processing

from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize Porter Stemmer
ps = PorterStemmer()

# Function to preprocess and transform the input text
def transform_text(text):
    # Convert to lowercase
    text = text.lower()
    # Tokenize into words
    text = nltk.word_tokenize(text)
    
    # Remove non-alphanumeric tokens
    text = [i for i in text if i.isalnum()]
    
    # Remove stopwords and punctuation
    text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]
    
    # Apply stemming
    text = [ps.stem(i) for i in text]
    
    return " ".join(text)

# Load the vectorizer and trained model
try:
    tk = pickle.load(open("vectorizer.pkl", 'rb'))
    model = pickle.load(open("model.pkl", 'rb'))
except FileNotFoundError:
    st.error("Required files (vectorizer.pkl and model.pkl) not found. Please ensure they are in the same directory as this script.")

# Streamlit application title and description
st.title("SMS Spam Detection Model")
st.write("*Made by Varun Tahiliani*")

# Input field for SMS
input_sms = st.text_input("Enter the SMS")

# Button to predict
if st.button('Predict'):
    if not input_sms.strip():
        st.warning("Please enter a valid SMS message.")
    else:
        # Preprocess the input
        transformed_sms = transform_text(input_sms)
        
        # Vectorize the processed text
        vector_input = tk.transform([transformed_sms])
        
        # Predict using the trained model
        result = model.predict(vector_input)[0]
        
        # Display the result
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")
