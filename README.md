Automated Bone Fracture Detection using Deep Learning

Project Overview

This project presents an AI-powered tool for detecting bone fractures in X-ray images, focusing on the elbow region.
It utilizes a fine-tuned ResNet-18 convolutional neural network trained on the XR_ELBOW subset of the MURA v1.1 dataset. 
The goal is to assist clinicians by automating fracture detection with high accuracy and confidence.

Key Features

• Fine-tuned ResNet-18 model on elbow X-ray images
• Real-time prediction using a Streamlit-based UI
• Confidence score display (Fracture/No Fracture)
• Downloadable PDF report generation using FPDF
• Early stopping, data augmentation, and transfer learning integrated

How It Works

1. Upload an X-ray image through the Streamlit interface.
2. The image is preprocessed and passed through the ResNet-18 model.
3. The model classifies the image as 'Fracture' or 'No Fracture' with a confidence score.
4. Optionally, a PDF report with the diagnosis and timestamp is generated.

Folder Structure
• app/streamlit_app.py
• models/resnet_finetune.py
• utils/data_loader.py
• train_eval.py
• requirements.txt
• README.md
• .gitignore
• models/best_model.pth (generated after training)

Installation

1. Clone the repository and navigate into the project directory.
2. Create and activate a virtual environment (recommended):
   conda create -n fracture-env python=3.10
   conda activate fracture-env
3. Install required packages:
   pip install -r requirements.txt
4. Train the model before running the app:
   python train_eval.py
   (This will generate models/best_model.pth)
5. Launch the Streamlit app:
   streamlit run app/streamlit_app.py

Usage

Upload an X-ray image and click 'Predict'. The model will return the classification and confidence.
If the trained model file is missing, the app will run in demo mode with an untrained model (for UI testing).
Optionally, download the report as a PDF.

Model Details
- ResNet-18 model pretrained on ImageNet
- Fine-tuned by unfreezing layer2, layer3, layer4, and fc layers
- Loss: CrossEntropyLoss | Optimizer: Adam | Epochs: 30 | Patience: 7
- Training Accuracy: 91% | Validation F1 Score: ~80%

Acknowledgements
• MURA v1.1 dataset by Stanford ML Group
• PyTorch for model building
• Streamlit for interactive UI
• FPDF for report generation
