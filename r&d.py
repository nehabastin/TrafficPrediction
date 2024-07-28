# -*- coding: utf-8 -*-
# NAME : NEHA BASTIN
# STUDENT ID : BAS22587831

# Dataset link: https://www.kaggle.com/datasets/shashwat9kumar/traffic-classificationdataset
#Colab link : https://colab.research.google.com/drive/1Vx0CGuDomFVBAwGHWCke8IqQFHJVjlWH#scrollTo=afa_ldL5baau
#Google drive link : https://drive.google.com/drive/folders/1otCjoZFJd_wCm6iGW0QZ00hFi38_-b1-?usp=sharing

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Define the directory containing the image dataset
dataset_dir = "/content/drive/MyDrive/Traffic_Dataset"

# Define the categories (classes)
categories = ["High_Traffic", "Moderate_Traffic", "No_Traffic"]

# Initialize lists to store images and their corresponding labels
data = []
labels = []

# Initialize dictionaries to store class counts
class_counts = {category: 0 for category in categories}

# Define the desired size for resizing the images
desired_size = (224, 224)  # Adjust the size as needed

# Loop through each category
for category in categories:
    # Construct the path to the category directory
    category_dir = os.path.join(dataset_dir, category)
    # Get the list of image files in the category directory
    image_files = os.listdir(category_dir)

    # Update class counts
    class_counts[category] = len(image_files)

    # Loop through each image file
    for image_file in image_files:
        # Read the image file using OpenCV
        image_path = os.path.join(category_dir, image_file)
        image = cv2.imread(image_path)
        # Preprocess the image (e.g., resize, normalize)
        image = cv2.resize(image, (224, 224))  # Resize image to a fixed size

        # Data Augmentation
        # Create an ImageDataGenerator instance for data augmentation
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest'
        )
        # Reshape the image to fit the expected shape for the generator
        image = np.expand_dims(image, axis=0)
        # Generate augmented images
        augmented_images = datagen.flow(image, batch_size=1)
        # Append augmented images to the data list
        data.extend(augmented_images[0])
        # Append the corresponding label to the labels list
        labels.append(categories.index(category))

# Convert the lists to NumPy arrays
data = np.array(data)
labels = np.array(labels)

# Display the shapes of the data and labels arrays
print("Data shape:", data.shape)
print("Labels shape:", labels.shape)

# Shuffle the data and labels to ensure randomness
data, labels = shuffle(data, labels, random_state=42)

# Split the dataset into training, validation, and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

# Print the shapes of the training, validation, and testing sets to verify preprocessing
print("Training data shape:", train_data.shape)
print("Training labels shape:", train_labels.shape)
print("Validation data shape:", val_data.shape)
print("Validation labels shape:", val_labels.shape)
print("Testing data shape:", test_data.shape)
print("Testing labels shape:", test_labels.shape)

# Print class counts
print("Class Counts:")
for category, count in class_counts.items():
    print(f"{category}: {count}")

# Basic EDA and visualizations
plt.figure(figsize=(10, 6))
sns.countplot(x=train_labels, palette='viridis')
plt.title('Distribution of Classes in Training Set')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1, 2], labels=categories)
plt.show()

# Display sample images from each class
plt.figure(figsize=(12, 6))
for i, category in enumerate(categories):
    plt.subplot(1, 3, i + 1)
    category_indices = np.where(train_labels == i)[0]
    sample_image_index = np.random.choice(category_indices)
    plt.imshow(train_data[sample_image_index].astype(np.uint8))  # Ensure the image is in the correct data type
    plt.title(category)
    plt.axis('off')
plt.show()

models.resnet50(pretrained=True)

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

# Define the CNN model architecture
class TrafficClassifier(nn.Module):
    def __init__(self, num_classes):
        super(TrafficClassifier, self).__init__()
        # Define the backbone (pre-trained ResNet50)
        self.backbone = models.resnet50(pretrained=True)
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
model = TrafficClassifier(num_classes=3)

# Define loss function (cross-entropy) and optimizer (Adam)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Print the model architecture
print(model)

# Define number of epochs and batch size
num_epochs = 10
batch_size = 32

# Convert NumPy arrays to PyTorch tensors
train_data_tensor = torch.tensor(train_data).permute(0, 3, 1, 2).float()
train_labels_tensor = torch.tensor(train_labels).long()
val_data_tensor = torch.tensor(val_data).permute(0, 3, 1, 2).float()
val_labels_tensor = torch.tensor(val_labels).long()

# Create PyTorch data loaders for training and validation sets
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(train_data_tensor, train_labels_tensor),
    batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(val_data_tensor, val_labels_tensor),
    batch_size=batch_size, shuffle=False)

# Training loop
for epoch in range(num_epochs):
    # Set model to training mode
    model.train()
    # Initialize variables for tracking loss and accuracy
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    # Iterate over training batches
    for inputs, targets in train_loader:
        # Zero the gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(inputs)
        # Calculate loss
        loss = criterion(outputs, targets)
        # Backward pass
        loss.backward()
        # Update weights
        optimizer.step()

        # Track loss and accuracy
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == targets).sum().item()
        total_predictions += targets.size(0)

    # Calculate average training loss and accuracy
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_accuracy = correct_predictions / total_predictions

    # Validate the model
    model.eval()
    val_loss = 0.0
    val_correct_predictions = 0
    val_total_predictions = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            val_correct_predictions += (predicted == targets).sum().item()
            val_total_predictions += targets.size(0)

    # Calculate average validation loss and accuracy
    val_epoch_loss = val_loss / len(val_loader.dataset)
    val_epoch_accuracy = val_correct_predictions / val_total_predictions

    # Print training and validation metrics
    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.4f}, "
          f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_accuracy:.4f}")

from sklearn.metrics import classification_report, confusion_matrix

# Set model to evaluation mode
model.eval()

# Initialize lists to store predictions and ground truth labels
all_predictions = []
all_targets = []

# Iterate over validation data
with torch.no_grad():
    for inputs, targets in val_loader:
        # Forward pass
        outputs = model(inputs)
        # Predicted class is the one with maximum probability
        _, predicted = torch.max(outputs, 1)
        # Append predictions and ground truth labels to lists
        all_predictions.extend(predicted.tolist())
        all_targets.extend(targets.tolist())

# Convert lists to NumPy arrays
all_predictions = np.array(all_predictions)
all_targets = np.array(all_targets)

# Generate classification report
print("Classification Report:")
print(classification_report(all_targets, all_predictions, target_names=categories))

# Generate confusion matrix
print("Confusion Matrix:")
conf_matrix = confusion_matrix(all_targets, all_predictions)
print(conf_matrix)

# Plot heatmap for confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=categories, yticklabels=categories)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Convert test data to PyTorch tensor
test_data_tensor = torch.tensor(test_data).permute(0, 3, 1, 2).float()
test_labels_tensor = torch.tensor(test_labels).long()

# Create PyTorch data loader for test set
test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(test_data_tensor, test_labels_tensor),
    batch_size=batch_size, shuffle=False)

# Initialize lists to store predictions and ground truth labels for test data
test_predictions = []
test_targets = []

# Iterate over test data
with torch.no_grad():
    for inputs, targets in test_loader:
        # Forward pass
        outputs = model(inputs)
        # Predicted class is the one with maximum probability
        _, predicted = torch.max(outputs, 1)
        # Append predictions and ground truth labels to lists
        test_predictions.extend(predicted.tolist())
        test_targets.extend(targets.tolist())

# Convert lists to NumPy arrays
test_predictions = np.array(test_predictions)
test_targets = np.array(test_targets)

# Display actual vs predicted results for each class
for i, category in enumerate(categories):
    category_indices = np.where(test_targets == i)[0]
    correct_predictions = np.sum(test_predictions[category_indices] == test_targets[category_indices])
    total_predictions = len(category_indices)
    accuracy = correct_predictions / total_predictions * 100
    print(f"Class: {category}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Total Images: {total_predictions}, Correct Predictions: {correct_predictions}, Incorrect Predictions: {total_predictions - correct_predictions}")
    print("-------------------------------------------")

import matplotlib.pyplot as plt
import numpy as np

# Set model to evaluation mode
model.eval()

# Define class names
class_names = ["High_Traffic", "Moderate_Traffic", "No_Traffic"]

# Function to display images with predictions
def display_images_with_predictions(images, labels, predictions):
    plt.figure(figsize=(15, 8))
    for i in range(min(9, len(images))):  # Display up to 9 images
        plt.subplot(3, 3, i + 1)
        # Normalize pixel values to be in the range [0, 1]
        plt.imshow(images[i] / 255.0)
        plt.title(f"True: {class_names[labels[i]]}\nPredicted: {class_names[predictions[i]]}")
        plt.axis('off')
    plt.show()

# Initialize lists to store images, true labels, and predicted labels
sample_images = []
true_labels = []
predicted_labels = []

# Iterate over test data and make predictions for a few images
with torch.no_grad():
    for i, (inputs, targets) in enumerate(test_loader):
        # Forward pass
        outputs = model(inputs)
        # Predicted class is the one with maximum probability
        _, predicted = torch.max(outputs, 1)
        # Convert tensors to NumPy arrays
        images_np = inputs.permute(0, 2, 3, 1).cpu().numpy()
        targets_np = targets.cpu().numpy()
        predicted_np = predicted.cpu().numpy()
        # Append a few images and their labels to the lists
        sample_images.extend(images_np)
        true_labels.extend(targets_np)
        predicted_labels.extend(predicted_np)
        # Display images with predictions for the first batch only
        if i == 0:
            display_images_with_predictions(images_np, targets_np, predicted_np)
            break

# prompt: Save the model

# Save the trained model
torch.save(model.state_dict(), '/content/drive/MyDrive/traffic_classifier_model.pth')

# prompt: Load the saved model and run a sample prediction from the. test data

import torch
import torchvision.models as models
import numpy as np

# Load the saved model
model = TrafficClassifier(num_classes=3)  # Initialize the model with the same architecture
model.load_state_dict(torch.load('/content/drive/MyDrive/traffic_classifier_model.pth'))
model.eval()  # Set the model to evaluation mode

# Select a sample image from the test data
sample_index = 10  # Choose any index within the range of test data
sample_image = test_data[sample_index]

# Convert the sample image to a PyTorch tensor and add a batch dimension
sample_image_tensor = torch.tensor(sample_image).unsqueeze(0).permute(0, 3, 1, 2).float()

# Make a prediction
with torch.no_grad():
  output = model(sample_image_tensor)
  _, predicted_class = torch.max(output, 1)

# Get the predicted class label
predicted_label = predicted_class.item()
print(predicted_label)

import torch
import torchvision.models as models

# Load the pre-trained ResNet50 model
resnet50 = models.resnet50(pretrained=True)

# Save the model's state dictionary to a file
torch.save(resnet50.state_dict(), '/content/drive/MyDrive/resnet50-weights.pth')

print('ResNet50 model weights saved to resnet50-weights.pth')

test_data[0]

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image



model.load_state_dict(torch.load('/content/drive/MyDrive/traffic_classifier_model.pth'))
model.eval()

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

# Example usage
image_path = '/content/drive/MyDrive/Traffic_Dataset/Moderate_Traffic/1.jpg'
predicted_class = predict_image(image_path)
print(f'The predicted class for the image is: {predicted_class}')

