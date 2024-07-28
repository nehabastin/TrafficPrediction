import torch
from torchvision import transforms
from PIL import Image
import streamlit as st
from PIL import Image

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

#write a streamlit app 



st.title("Traffic Classifier")

page = st.sidebar.selectbox("Choose a page", ["Home", "Demo"])


if page == "Home":
    st.title("Home Page")
    with open('r&d.py', 'r') as file:
        code = file.read()
    st.code(code)
    # Add content for the home page here
elif page == "Demo":
    #upload button for the resnet model

    upload_resnet = st.file_uploader("Choose the resnet model...", type="pth")
    upload_model = st.file_uploader("Choose the traffic classifier model...", type="pth")
    
    # Check if both models are uploaded
    if upload_resnet is None or upload_model is None:
        st.write("Please upload both the resnet model and the traffic classifier model.")
    else:
        # Save the resnet model as a file
        with open('resnet50-weights.pth', 'wb') as f:
            f.write(upload_resnet.getvalue())
        st.write("Resnet model uploaded successfully")
        
        # Save the traffic classifier model as a file
        with open('traffic_classifier_model.pth', 'wb') as f:
            f.write(upload_model.getvalue())
        st.write("Traffic classifier model uploaded successfully")
        
        # Continue with the rest of the code
        uploaded_image = st.file_uploader("Choose an image...", type="jpg")
        if uploaded_image is not None:
            # Define the CNN model architecture
            class TrafficClassifier(nn.Module):
                def __init__(self, num_classes):
                    super(TrafficClassifier, self).__init__()
                    # Define the backbone (pre-trained ResNet50)
                    self.backbone = models.resnet50(pretrained=False)
                    # Load the pre-trained weights
                    self.backbone.load_state_dict(torch.load('resnet50-weights.pth'))
                    # Freeze the parameters of the backbone
                    for param in self.backbone.parameters():
                        param.requires_grad = False
                    # Replace the last fully connected layer with a new one
                    self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

                def forward(self, x):
                    # Forward pass through the backbone
                    x = self.backbone(x)
                    return x

            # Initialize the model
            # Load the trained model
            model = TrafficClassifier(num_classes=3)
            model.load_state_dict(torch.load('traffic_classifier_model.pth'))

            # Define the categories (classes)
            categories = ["High_Traffic", "Moderate_Traffic", "No_Traffic"]

            # Function to preprocess the image
            def preprocess_image(image_path):
                preprocess = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                image = Image.open(image_path)
                image = preprocess(image)
                image = image.unsqueeze(0)  # Add batch dimension
                return image

            # Function to predict the class of the image
            def predict_image(image_path):
                # Preprocess the image
                preprocessed_image = preprocess_image(image_path)
                # Predict using the model
                with torch.no_grad():
                    output = model(preprocessed_image)
                    _, predicted_class_index = torch.max(output, 1)
                    predicted_class = categories[predicted_class_index.item()]
                return predicted_class

            image = Image.open(uploaded_image)
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            st.write("")
            st.write("Classifying...")
            predicted_class = predict_image(uploaded_image)
            st.write(f"Predicted Class: {predicted_class}")
