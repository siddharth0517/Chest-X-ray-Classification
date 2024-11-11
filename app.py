import streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load your trained model
# Ensure to specify the correct path to your saved model file
model_path = 'model.keras'  # Replace with your model file path
stacked_model = load_model(model_path)

# Define class names
class_names = ['Normal', 'Pneumonia']

# Streamlit app title and description
st.title("Chest X-ray Classification")
st.write("This application classifies chest X-ray images as **Normal** or **Pneumonia** using a deep learning model.")

# Upload an image file
uploaded_file = st.file_uploader("Please upload a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")

    # Preprocess the image
    img = image.resize((224, 224))  # Resize the image to match the input shape of the model
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize

    # Predict the class
    predictions = stacked_model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    # Output the result
    st.write(f"**Predicted class**: {class_names[predicted_class]}")
    st.write(f"**Confidence**: {confidence:.2f}")
