# app/streamlit_app.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import datetime
from fpdf import FPDF

from models.resnet_finetune import get_finetuned_resnet18

st.set_page_config(page_title="Bone Fracture Detection", layout="centered")

@st.cache_resource
def load_model(model_path="models/best_model.pth"):
    model = get_finetuned_resnet18(num_classes=2)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

st.title(" Bone Fracture Detection")

uploaded_file = st.file_uploader("Upload an X-ray Image", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray Image", use_container_width=True)

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing..."):
            img_tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                outputs = model(img_tensor)
                _, predicted = torch.max(outputs, 1)
                confidence = torch.softmax(outputs, dim=1)[0][predicted.item()].item()

            predicted_class = predicted.item()

            # Save prediction in session state
            st.session_state['prediction'] = predicted_class
            st.session_state['confidence'] = confidence

            if predicted_class == 1:
                st.error(f" Fracture Detected (Confidence: {confidence*100:.2f}%)")
            else:
                st.success(f" No Fracture Detected (Confidence: {confidence*100:.2f}%)")

# PDF Report Generator function
def generate_pdf_report(prediction, confidence):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="Bone Fracture Detection Report", ln=True, align='C')
    pdf.ln(10)

    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.cell(200, 10, txt=f"Date: {date_str}", ln=True)
    pdf.ln(5)

    if prediction == 1:
        pdf.set_text_color(200, 0, 0)
        pdf.cell(200, 10, txt=" Fracture Detected", ln=True)
    else:
        pdf.set_text_color(0, 150, 0)
        pdf.cell(200, 10, txt=" No Fracture Detected", ln=True)

    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)
    pdf.cell(200, 10, txt=f"Confidence Score: {confidence*100:.2f}%", ln=True)

    output_path = "fracture_report.pdf"
    pdf.output(output_path)
    return output_path

# Show Download PDF button only after prediction
if 'prediction' in st.session_state:
    if st.button("📄 Download Report as PDF"):
        report_path = generate_pdf_report(st.session_state['prediction'], st.session_state['confidence'])
        with open(report_path, "rb") as f:
            st.download_button("⬇️ Click to Download", f, file_name="fracture_report.pdf")
