import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import openai

# Load API key from Streamlit secrets
openai.api_key = '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'


# Caching the model to improve performance
@st.cache_resource
def load_model():
    try:
        model = tf.keras.applications.MobileNetV2(weights="imagenet")
        return model
    except Exception as e:
        st.error(f"Failed to load the model. Error: {e}")
        st.stop()

model = load_model()

# Preprocess the image
def preprocess_image(image):
    try:
        st.write("Preprocessing image...")
        img = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        st.write("Image preprocessed successfully.")
        return img_array
    except Exception as e:
        st.error(f"Failed to preprocess the image. Error: {e}")
        return None

# Get image classification results with error handling
def classify_image(image):
    processed_image = preprocess_image(image)
    if processed_image is not None:
        try:
            st.write("Running model prediction...")
            predictions = model.predict(processed_image)
            decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)
            st.write(f"Predictions: {decoded_predictions}")
            return decoded_predictions[0][0][1]  # Return the most likely class name
        except Exception as e:
            st.error(f"Failed to classify the image. Error: {e}")
            return None
    else:
        return None

# Generate a response using GPT with error handling
def generate_response(text):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # You can also use "gpt-4" if available
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": text}
            ]
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        st.error(f"Failed to generate a response. Error: {e}")
        return None

# Streamlit UI
st.title("Image-Chatbot with Streamlit")
st.write("Upload an image and ask a question!")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    image_class = classify_image(image)
    if image_class:
        st.write(f"Image Classification: **{image_class}**")

        user_question = st.text_input("Ask a question based on the image:")

        if user_question and user_question.strip():
            context = f"The image shows a {image_class}. {user_question}"
            chatbot_response = generate_response(context)
            if chatbot_response:
                st.write(f"Chatbot: {chatbot_response}")
        else:
            st.warning("Please enter a valid question.")
