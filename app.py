import streamlit as st
from transformers import pipeline
from PIL import Image

# Load the Hugging Face model pipeline
@st.cache_resource
def load_model():
    model = pipeline("image-classification", model="dima806/cat_breed_image_detection")
    return model

model = load_model()

# Streamlit UI
st.title("🐱 Cat Breed Detector App")

uploaded_file = st.file_uploader("Upload an image of a cat", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Cat Image", use_column_width=True)

    # Predict button
    if st.button("Predict Cat Breed"):
        with st.spinner('Analyzing the image...'):
            result = model(image)

            # Sort predictions by score
            sorted_result = sorted(result, key=lambda x: x['score'], reverse=True)
            top_prediction = sorted_result[0]

            # Show result
            st.success(f"Predicted Breed: **{top_prediction['label']}** with confidence {top_prediction['score']:.2f}")
