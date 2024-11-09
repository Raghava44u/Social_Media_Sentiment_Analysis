import streamlit as st
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import pickle

# Load model architecture
with open("model_architecture.json", "r") as json_file:
    model_json = json_file.read()
classifier = model_from_json(model_json)

# Load model weights
with open("model_weights.pkl", "rb") as weights_file:
    model_weights = pickle.load(weights_file)
classifier.set_weights(model_weights)

# Define emotion class names
class_names = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']

# Streamlit App
st.title("Image Sentiment Analysis")
st.write("Upload an image, and the model will predict the emotion!")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Load the image for prediction
    img = image.load_img(uploaded_file, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Rescale the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Make a prediction
    prediction = classifier.predict(img_array)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_class_name = class_names[predicted_class_index]
    
    # Display the predicted emotion
    st.write(f"Predicted Emotion: **{predicted_class_name}**")
