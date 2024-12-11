import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import joblib
import streamlit as st
import pandas as pd
import seaborn as sns
import base64

# Image resizing dimensions
IMG_SIZE = (64, 64)
st.set_page_config(page_title="Skin Cancer Detection", layout="wide")

# Helper function to read and resize images
def read(image):
    return np.asarray(Image.open(image).resize(IMG_SIZE).convert("RGB"))

# Load saved PCA and SVM model
model = joblib.load('svm_skin_cancer_model.pkl')
pca = joblib.load('pca_skin_cancer.pkl')

# Helper function to display uploaded image
def display_uploaded_image(image):
    image.seek(0)  # Reset file pointer
    img_data = base64.b64encode(image.read()).decode('utf-8')
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center; align-items: center; margin-top: 20px;">
            <img src="data:image/jpeg;base64,{img_data}" 
                 style="width: 500px; height: 500px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);" 
                 alt="Uploaded Image">
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- Predict for a User-Specified Image ---
def predict_image(image):
    if image is None:
        st.error("No image uploaded")
        return

    # Display uploaded image
    display_uploaded_image(image)

    # Preprocess the image
    img = read(image)
    img_normalized = img.astype('float32') / 255  # Normalize
    img_flat = img_normalized.reshape(1, -1)     # Flatten
    img_pca = pca.transform(img_flat)            # Apply PCA

    # Make prediction
    prediction = model.predict(img_pca)[0]
    label = "Malignant" if prediction == 1 else "Benign"

    # Show prediction
    st.markdown(
        f"<h3 style='text-align: center; color: {'crimson' if label == 'Malignant' else 'green'};'>"
        f"The model predicts: {label}</h3>",
        unsafe_allow_html=True,
    )

    # Display effects
    if label == "Malignant":
        st.snow()
    else:
        st.balloons()

    # Recommendations
    st.subheader("Recommendations:")
    if label == "Malignant":
        st.warning(
            """
            ### Immediate Steps:
            - Seek immediate medical attention.
            - Schedule an appointment with a dermatologist.
            - Avoid prolonged sun exposure and use sunscreen.
            """
        )
        st.info(
            """
            **Further Actions:**
            - Maintain a healthy diet and lifestyle.
            - Monitor other skin abnormalities regularly.
            - Discuss possible biopsy options with your doctor.
            """
        )
    else:
        st.success(
            """
            ### Great News:
            - The model has predicted this image as benign.
            - Keep monitoring for any changes in size, shape, or color.
            - Consult a healthcare professional for annual skin checkups.
            """
        )
        st.info(
            """
            **Tips for Healthy Skin:**
            - Stay hydrated and use skin moisturizers.
            - Apply sunscreen daily to prevent UV damage.
            - Avoid harsh chemicals or irritants on your skin.
            """
        )

# --- Data Analytics Dashboard ---
def display_dashboard():
    # Load training data for visualization
    folder_benign_train = 'train/benign'
    folder_malignant_train = 'train/malignant'
    folder_benign_test = 'test/benign'
    folder_malignant_test = 'test/malignant'

    # Count training images
    benign_images_train = len([f for f in os.listdir(folder_benign_train) if f.endswith(('.jpg', '.png', '.jpeg'))])
    malignant_images_train = len([f for f in os.listdir(folder_malignant_train) if f.endswith(('.jpg', '.png', '.jpeg'))])
    total_train_images = benign_images_train + malignant_images_train

    # Count testing images
    benign_images_test = len([f for f in os.listdir(folder_benign_test) if f.endswith(('.jpg', '.png', '.jpeg'))])
    malignant_images_test = len([f for f in os.listdir(folder_malignant_test) if f.endswith(('.jpg', '.png', '.jpeg'))])
    total_test_images = benign_images_test + malignant_images_test

    # Display dataset distribution
    st.subheader("Training Dataset Overview")
    st.write(f"**Total Training Images:** {total_train_images}")
    st.write(f"- Benign Images: {benign_images_train}")
    st.write(f"- Malignant Images: {malignant_images_train}")
    data_train = {'Class': ['Benign', 'Malignant'], 'Count': [benign_images_train, malignant_images_train]}
    df_train = pd.DataFrame(data_train)
    st.bar_chart(df_train.set_index('Class'))

    st.subheader("Testing Dataset Overview")
    st.write(f"**Total Testing Images:** {total_test_images}")
    st.write(f"- Benign Images: {benign_images_test}")
    st.write(f"- Malignant Images: {malignant_images_test}")
    data_test = {'Class': ['Benign', 'Malignant'], 'Count': [benign_images_test, malignant_images_test]}
    df_test = pd.DataFrame(data_test)
    st.bar_chart(df_test.set_index('Class'))

    # Model performance
    st.subheader("Model Performance")
    accuracy = joblib.load('svm_skin_cancer_accuracy.pkl')
    st.metric("Model Accuracy", f"{accuracy * 100:.2f}%")

    # Confusion matrix
    st.subheader("Confusion Matrix")
    cm = joblib.load('confusion_matrix.pkl')
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', cbar=False,
        xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"]
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    st.pyplot(fig)

# --- Sidebar Navigation ---
st.sidebar.title("Skin Cancer Detection")
st.sidebar.markdown("---")
option = st.sidebar.radio(
    "Select an option:",
    ["🔍 Image Prediction", "📊 Data Insights Dashboard"]
)

# --- Main Content Based on Sidebar Selection ---
if option == "🔍 Image Prediction":
    st.header("Upload an Image for Prediction")
    st.markdown("**Instructions:** Upload an image of a skin lesion to classify it as benign or malignant.")

    uploaded_image = st.file_uploader(
        "Upload an image (JPG, PNG, JPEG):", type=["jpg", "png", "jpeg"]
    )

    if uploaded_image:
        predict_image(uploaded_image)

elif option == "📊 Data Insights Dashboard":
    st.header("Data Insights Dashboard")
    st.markdown("**Explore:** Analyze dataset distributions and visualize model performance metrics.")
    display_dashboard()

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.write("A Streamlit App for Skin Cancer Detection")