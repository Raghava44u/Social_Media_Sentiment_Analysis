# Install streamlit with: pip install streamlit
# Run the app with: streamlit run app.py

import streamlit as st
import pickle
import time

# Load the model
with open('C:/Users/pavani/Downloads/NLP-Project-3---Twitter-Sentiment-Analysis-with-Random-Forest-main/new/twitter_sentiment.pkl', 'rb') as file:
    model = pickle.load(file)

st.title('Twitter Sentiment Analysis')

# Input field for the tweet
tweet = st.text_input('Enter your tweet')

# Button to submit and get the prediction
submit = st.button('Predict')

if submit:
    start = time.time()
    prediction = model.predict([tweet])  # Assuming the model has a predict method that accepts a list
    end = time.time()
    
    # Display the prediction result and the time taken
    st.write('Prediction time taken:', round(end - start, 2), 'seconds')
    st.write('Sentiment Prediction:', prediction[0])
