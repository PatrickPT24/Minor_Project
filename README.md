# Minor_Project
 
# 🧠 Alzheimer Detection using Vision Transformers (ViT)

This app uses an ensemble of ViT models to classify Alzheimer’s stages from MRI brain scans. Built with PyTorch and Streamlit.

## Features
- Ensemble of `google/vit-base-patch16-224` and `facebook/dino-vits16`
- Beautiful dark-themed GUI
- Upload MRI and classify into:
  - Mild Demented
  - Moderate Demented
  - Very Mild Demented
  - Non Demented

## How to Run

```bash
# Clone repo
git clone https://github.com/your-username/alzheimer-app.git
cd alzheimer-app

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
