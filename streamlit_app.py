import streamlit as st
import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CONFIGURATION ==========
MODEL_PATH = 'best_model_m1_notebook.pt'
IMG_SIZE = 224
NUM_CLASSES = 2
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
OPTIMAL_THRESHOLD = 0.5   # Updated threshold

# ========== LOAD MODEL (CACHED) ==========
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

# ========== IMAGE PREPROCESSING ==========
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ========== MAKE PREDICTION ==========
def predict(model, image):
    img_tensor = val_transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = nn.functional.softmax(outputs, dim=1)[0].cpu().numpy()
    
    prob_yes = probs[0]
    prob_no = probs[1]
    prediction = "TUMOR DETECTED" if prob_yes >= OPTIMAL_THRESHOLD else "NO TUMOR"
    confidence = max(prob_yes, prob_no)
    
    return {
        'prediction': prediction,
        'prob_yes': prob_yes,
        'prob_no': prob_no,
        'confidence': confidence,
        'threshold': OPTIMAL_THRESHOLD
    }

# ========== MAIN APP ==========
def main():
    # ---------- Title ----------
    st.markdown(
        "<h1 style='text-align:center; color:#0066CC; font-family:Arial;'>🧠 Brain Tumor Detection</h1>", 
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align:center; color:#666666; font-family:Arial;'>Upload MRI image to be evaluated by AI model</p>", 
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align:center; color:#666666; font-family:Arial;'>Model was trained to detect glioblastoma</p>", 
        unsafe_allow_html=True
    )
    
    # ---------- Sidebar ----------
    with st.sidebar:
        st.markdown("<h3 style='font-family:Arial;'>Model Information</h3>", unsafe_allow_html=True)
        st.metric("Model", "ResNet18")
        st.metric("Test F1-Score", "0.9651")
        st.metric("Independent Dataset F1", "0.9926")
        st.metric("ROC AUC", "0.9999")
        st.metric("Sensitivity (TPR)", "0.9883")
        st.metric("Specificity (TNR)", "0.9975")
        st.metric("Decision Threshold", f"{OPTIMAL_THRESHOLD}")
        st.divider()
        st.write("**Model Performance:**")
        st.write("- Catches 98.83% of tumors")
        st.write("- False alarm rate: 0.25%")
        st.write("- Trained on 1,356 images")

    model = load_model()

    col1, col2 = st.columns(2)

    # ---------- Upload / Example ----------
    with col1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a brain scan image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a grayscale or color brain scan image"
        )

        # Example images
        st.subheader("Select an example image")
        base_path = Path(__file__).parent / "images"
        tumor_path = base_path / "brain_tumor"
        normal_path = base_path / "normal"

        example_category = st.radio(
            "Example Category:",
            ["brain_tumor", "normal"],
            horizontal=True
        )

        if example_category == "brain_tumor":
            example_files = sorted(tumor_path.glob("*.jpg"))
        else:
            example_files = sorted(normal_path.glob("*.jpg"))

        selected_example = st.selectbox(
            "Choose an example image:",
            example_files,
            format_func=lambda x: x.name
        )
        example_image = Image.open(selected_example).convert("RGB")
        st.image(example_image, caption=f"Example: {selected_example.name}", use_column_width=True)

        if uploaded_file is None:
            uploaded_file = selected_example

    # ---------- Prediction ----------
    with col2:
        st.subheader("Prediction Result")
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            result = predict(model, image)

            if result['prediction'] == "TUMOR DETECTED":
                st.markdown(f"<h3 style='color:#FF4444; font-family:Arial;'>{result['prediction']}</h3>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h3 style='color:#44FF44; font-family:Arial;'>{result['prediction']}</h3>", unsafe_allow_html=True)

            st.metric(
                "Model Confidence",
                f"{result['confidence']:.2%}",
                help="How confident is the model in this prediction"
            )

    # ---------- Display image and probabilities ----------
    if uploaded_file is not None:
        col_img, col_details = st.columns([1, 1])
        with col_img:
            st.subheader("Uploaded Image")
            st.image(image, use_column_width=True)
        with col_details:
            st.subheader("Prediction Details")
            st.write("**Probability Scores:**")
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.metric("Tumor (Yes)", f"{result['prob_yes']:.4f}", help="Probability tumor is present")
            with col_p2:
                st.metric("No Tumor", f"{result['prob_no']:.4f}", help="Probability tumor is absent")

            st.write("**Classification Logic:**")
            st.write(f"- If Tumor probability >= {OPTIMAL_THRESHOLD} --> Predict TUMOR")
            st.write(f"- If Tumor probability < {OPTIMAL_THRESHOLD} --> Predict NO TUMOR")
            st.divider()

            # Bar chart
            fig, ax = plt.subplots(figsize=(8, 3))
            categories = ['Tumor', 'No Tumor']
            probabilities = [result['prob_yes'], result['prob_no']]
            colors = ['#FF6B6B', '#4ECDC4']
            bars = ax.barh(categories, probabilities, color=colors)
            ax.axvline(OPTIMAL_THRESHOLD, color='black', linestyle='--', linewidth=2, label=f'Threshold: {OPTIMAL_THRESHOLD}')
            ax.set_xlim([0, 1])
            ax.set_xlabel('Probability', fontsize=11, fontweight='bold')
            ax.set_title('Model Output Probabilities', fontsize=12, fontweight='bold')
            ax.legend()
            for i, (bar, prob) in enumerate(zip(bars, probabilities)):
                ax.text(prob + 0.02, i, f'{prob:.4f}', va='center', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

    # ---------- Batch Processing ----------
    st.divider()
    st.subheader("Batch Processing")
    uploaded_files = st.file_uploader(
        "Upload multiple images for batch predictions",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        key="batch_uploader"
    )

    if uploaded_files and len(uploaded_files) > 0:
        st.write(f"Processing {len(uploaded_files)} images...")
        results_list = []
        progress_bar = st.progress(0)
        for idx, file in enumerate(uploaded_files):
            image = Image.open(file).convert('RGB')
            result = predict(model, image)
            results_list.append({
                'Filename': file.name,
                'Prediction': result['prediction'],
                'Tumor Prob': f"{result['prob_yes']:.4f}",
                'Confidence': f"{result['confidence']:.2%}"
            })
            progress_bar.progress((idx + 1) / len(uploaded_files))

        # ---------- Styled Batch Table ----------
        df = pd.DataFrame(results_list)
        def highlight_tumor(row):
            return ['background-color: #FFCCCC; font-family: Arial;' if row.Prediction == 'TUMOR DETECTED' else 'font-family: Arial;' for _ in row]
        st.subheader("Batch Results")
        st.dataframe(df.style.apply(highlight_tumor, axis=1), use_container_width=True)

main()
