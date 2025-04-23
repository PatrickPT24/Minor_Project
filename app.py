import streamlit as st
from PIL import Image
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from model_utils import load_models, predict_image

st.set_page_config(page_title="Alzheimer Detection App", layout="wide")

# Load models
class_names = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
model1, model2 = load_models(num_classes=len(class_names))

# Dark mode styling
def set_background():
    st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: white;
    }
    .stButton>button {
        color: white;
        background-color: #28a745;
        border-radius: 10px;
    }
    .stTextInput>div>div>input {
        background-color: #1e1e1e;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

set_background()

# Sidebar navigation
page = st.sidebar.selectbox("📄 Navigate", ["🏠 Welcome", "🔐 Login", "🧠 Classifier"])

# Welcome Screen
if page == "🏠 Welcome":
    st.title("🧠 Alzheimer's Detection using MRI")
    st.image("assets/background.jpg", use_column_width=True)
    st.markdown("""
    ### Vision Transformer Ensemble-based Classifier  
    Upload brain MRI images to detect the stage of Alzheimer's.  
    """, unsafe_allow_html=True)

# Login Screen
elif page == "🔐 Login":
    st.header("🔐 Login")
    username = st.text_input("👤 Enter your name")
    email = st.text_input("📧 Enter your email")
    age = st.number_input("🎂 Age", min_value=1, max_value=120)
    if st.button("Login"):
        if username and email and age:
            st.success(f"Welcome, {username}!")
        else:
            st.error("Please fill in all the details.")

# Classifier Screen
elif page == "🧠 Classifier":
    st.header("🖼️ Upload MRI Image for Classification")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded MRI", use_column_width=True)

        if st.button("🧠 Predict"):
            pred_class, probs = predict_image(image, model1, model2, class_names)
            st.success(f"Prediction: **{class_names[pred_class]}**")

            st.markdown("### 📊 Probability Chart")
            fig, ax = plt.subplots()
            sns.barplot(x=probs, y=class_names, palette="rocket", ax=ax)
            ax.set_xlim(0, 1)
            st.pyplot(fig)
