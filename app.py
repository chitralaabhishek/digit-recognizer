import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import streamlit as st
from PIL import Image, ImageOps
import os

# Cache the model so it doesn't retrain every time the app loads
@st.cache_resource
def train_and_save_model():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize the data
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Reshape for CNN
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    # One-hot encode the labels
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)

    # Define the model architecture
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train_cat, epochs=3, validation_data=(x_test, y_test_cat))

    # Save the trained model
    model.save("digit_model.h5")
    return model

# Load the model if already trained, else train it
if os.path.exists("digit_model.h5"):
    model = load_model("digit_model.h5")
else:
    st.write("Training model for the first time...")
    model = train_and_save_model()

# Streamlit app UI
st.title("🧠 Handwritten Digit Recognizer")
st.write("Upload a 28x28 image of a handwritten digit (0–9).")

# File uploader for user to upload an image
uploaded_file = st.file_uploader("Choose a digit image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Open and process the uploaded image
    image = Image.open(uploaded_file).convert('L')
    image = ImageOps.invert(image)  # Invert the image color for better recognition
    st.image(image, caption='Uploaded Image', width=150)

    # Resize image to match model input size (28x28)
    image = image.resize((28, 28))
    img_array = np.array(image) / 255.0  # Normalize the image
    img_array = img_array.reshape(1, 28, 28, 1)  # Reshape to match model input

    # Predict the digit
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)

    # Display the predicted result
    st.subheader(f"Predicted Digit: {predicted_digit}")
    st.write(f"Confidence: {np.max(prediction) * 100:.2f}%")
