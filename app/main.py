import os
import json
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the pre-trained model
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/plant_disease_prediction_model.h5"
model = tf.keras.models.load_model(model_path)
model.save('plant_disease_prediction_model.h5')


# Load the class names
class_indices = json.load(open(f"{working_dir}/class_indices.json"))

# Function to Load and Preprocess the Image
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    img_array = np.asarray(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices, top_k=5):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)[0]
    top_indices = np.argsort(predictions)[::-1][:top_k]
    top_classes = [class_indices[str(i)] for i in top_indices]
    top_probs = [predictions[i] for i in top_indices]
    return top_classes, top_probs

# Streamlit App UI
st.set_page_config(page_title='Plant Disease Classifier', page_icon='🌿', layout='wide')

st.title('🌿 Plant Disease Classifier')

# Sidebar for image upload
st.sidebar.title('📸 Upload Image')
uploaded_image = st.sidebar.file_uploader("Upload an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.sidebar.image(image, caption='Uploaded Image', use_column_width=True)

    if st.sidebar.button('Classify'):
        with st.spinner('Classifying...'):
            top_classes, top_probs = predict_image_class(model, uploaded_image, class_indices)
            
            # Debugging output
            st.write("Predicted Classes:", top_classes)
            st.write("Prediction Probabilities:", top_probs)

        if top_classes and top_probs:  # Check if predictions were made
            st.write('\n\n### Top Predictions:')
            
            # Create a figure and axes
            fig, ax = plt.subplots(figsize=(8, 6))

            # Plot the bar chart with a custom color palette
            ax.bar(top_classes, top_probs, color=['#6A5ACD', '#00FA9A', '#FF4500', '#1E90FF', '#FFD700'])
            ax.set_xlabel('Predicted Class', fontsize=14, color='#2F4F4F')
            ax.set_ylabel('Probability', fontsize=14, color='#2F4F4F')
            ax.set_title('Top Predictions', fontsize=16, color='#2F4F4F')
            ax.tick_params(axis='x', rotation=45)
            for label in ax.get_xticklabels():
                label.set_ha('right')

            # Display the plot in Streamlit
            st.pyplot(fig)

            # Add emojis (replace with your emoji dictionary)
            emojis = {
                'Healthy Plant': '🌱',
                'Disease 1': '🦠',
                'Disease 2': '🐛',
                # Add more emojis for other classes as needed
            }

            for i, (class_name, prob) in enumerate(zip(top_classes, top_probs), 1):
                emoji = emojis.get(class_name, '')  # Use '' for unknown classes
                st.write(f'{i}. {emoji} **{class_name}** (Probability: {prob:.4f})')

            # Add stickers or images corresponding to the classes
            
        else:
            st.write("No predictions made. Please check the model and input image.")
