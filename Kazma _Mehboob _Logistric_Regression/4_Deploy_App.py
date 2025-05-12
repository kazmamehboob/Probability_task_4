import streamlit as st
import pickle
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

# Set paths
MODEL_PATH = r"E:\Project\models\model.pkl"
DATA_PATH = r"E:\Project\models\split.pkl"
TEST_IMAGE_FOLDER = r"E:\Project\data\archive\test\test_1"

# Load the model and data
@st.cache_resource
def load_model_and_data():
    model = pickle.load(open(MODEL_PATH, "rb"))
    X_train, X_test, y_train, test_filenames = pickle.load(open(DATA_PATH, "rb"))
    return model, X_test, test_filenames

# Predict labels
def predict(model, X):
    return model.predict(X)

# Title
st.title("🐶🐱 Cat vs Dog Image Classifier")

# Load model and test data
model, X_test, test_filenames = load_model_and_data()

# Show predictions for first 10 test images
st.subheader("🔍 Predictions on Sample Test Images")

cols = st.columns(5)
for i in range(10):
    img_path = os.path.join(TEST_IMAGE_FOLDER, test_filenames[i])
    image = Image.open(img_path)

    prediction = model.predict([X_test[i]])[0]
    label = "Dog 🐶" if prediction == 0 else "Cat 🐱"

    with cols[i % 5]:
        st.image(image, caption=label, use_column_width=True)

# Upload your own image
st.subheader("📤 Try with Your Own Image")

uploaded_file = st.file_uploader("Upload a JPG or PNG image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file).convert("RGB")
        img_resized = img.resize((64, 64))
        img_array = np.array(img_resized).flatten() / 255.0
        prediction = model.predict([img_array])[0]
        label = "Dog 🐶" if prediction == 0 else "Cat 🐱"

        st.image(img, caption=f"Predicted: {label}", use_column_width=True)
    except Exception as e:
        st.error(f"Error processing image: {e}")