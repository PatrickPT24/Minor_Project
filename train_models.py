# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# from torch.utils.data import DataLoader
# from torch.utils.data import random_split
# from transformers import ViTForImageClassification, ViTFeatureExtractor
# from torch.amp import GradScaler

# # Define device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load Alzheimer’s dataset
# data_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5], [0.5])
# ])

# # Load Dataset
# dataset_path = r"C:\Users\taral\Minor_Project\OriginalDataset"
# if not os.path.exists(dataset_path):
#     raise FileNotFoundError(f"Dataset path {dataset_path} does not exist. Please check the path.")


# dataset = datasets.ImageFolder(root=dataset_path, transform=data_transform)
# train_size = int(0.8 * len(dataset))
# test_size = len(dataset) - train_size
# train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
# num_classes = len(dataset.classes)
# print(f"Detected {num_classes} classes in dataset: {dataset.classes}")

# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
# test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

# # Load pre-trained ViT model
# model1 = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", num_labels=num_classes, ignore_mismatched_sizes=True).to(device)
# model2 = ViTForImageClassification.from_pretrained("facebook/dino-vits16", num_labels=num_classes, ignore_mismatched_sizes=True).to(device)

# # Define loss and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer1 = optim.Adam(model1.parameters(), lr=3e-5)
# optimizer2 = optim.Adam(model2.parameters(), lr=3e-5)

# print("Unique labels in dataset:", set(dataset.targets))

# # Training function
# scaler = GradScaler()
# def train_model(model, optimizer, train_loader, model_name, num_epochs=5):
#     model.train()
#     for epoch in range(num_epochs):
#         total_loss = 0
#         for images, labels in train_loader:
#             images, labels = images.to(device), labels.to(device)

#             # Ensure labels are in range [0, num_labels - 1]
#             labels = labels.long()  # Convert to LongTensor if needed
#             optimizer.zero_grad()

#             if device.type == "cuda":  # Use mixed precision only for GPUs
#                 with torch.amp.autocast(device_type="cuda"):
#                     outputs = model(images).logits
#                     loss = criterion(outputs, labels)
#             else:
#                 outputs = model(images).logits
#                 loss = criterion(outputs, labels)

#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")
    
#     # Save trained model
#     save_path = f"saved_model/{model_name}.pth"
#     torch.save(model.state_dict(), save_path)
#     print(f"✅ {model_name} saved at {save_path}")

# # Train and save both models
# if __name__ == '__main__':
#     print("🚀 Training Model 1 (ViT Base)")
#     train_model(model1, optimizer1, train_loader, "model1")

#     print("🚀 Training Model 2 (DINO-ViT)")
#     train_model(model2, optimizer2, train_loader, "model2")

# # Ensemble Model Evaluation
# def evaluate_ensemble(models, test_loader):
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for images, labels in test_loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = [model(images).logits for model in models]
#             avg_output = torch.mean(torch.stack(outputs), dim=0)
#             _, predicted = torch.max(avg_output, 1)
#             correct += (predicted == labels).sum().item()
#             total += labels.size(0)
#     print(f"Ensemble Model Accuracy: {100 * correct / total:.2f}%")

# # Evaluate the ensemble model
# evaluate_ensemble([model1, model2], test_loader)


import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
from transformers import ViTForImageClassification
from torch.amp import GradScaler

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define constants
DATASET_PATH = r"C:\Users\taral\Minor_Project\OriginalDataset"
BATCH_SIZE = 16
NUM_WORKERS = 4
NUM_EPOCHS = 5
LEARNING_RATE = 3e-5

# Data Transform
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Load Dataset
def load_datasets(dataset_path):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")

    dataset = datasets.ImageFolder(root=dataset_path, transform=data_transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    print(f"Detected {len(dataset.classes)} classes: {dataset.classes}")
    return train_dataset, test_dataset, len(dataset.classes)

# Training Function
scaler = GradScaler()
def train_model(model, optimizer, train_loader, model_name, num_epochs=NUM_EPOCHS):
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.long()
            optimizer.zero_grad()

            if device.type == "cuda":
                with torch.amp.autocast(device_type="cuda"):
                    outputs = model(images).logits
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images).logits
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}")
    
    # Save model
    os.makedirs("saved_model", exist_ok=True)
    save_path = f"saved_model/{model_name}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"✅ {model_name} saved at {save_path}")

# Ensemble Evaluation
def evaluate_ensemble(models, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = [model(images).logits for model in models]
            avg_output = torch.mean(torch.stack(outputs), dim=0)
            _, predicted = torch.max(avg_output, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    print(f"🧠 Ensemble Accuracy: {100 * correct / total:.2f}%")

# Main Script
def main():
    print("📦 Loading dataset...")
    train_dataset, test_dataset, num_classes = load_datasets(DATASET_PATH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print("🔍 Unique labels:", set(datasets.ImageFolder(root=DATASET_PATH).targets))

    print("📥 Initializing models...")
    model1 = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", num_labels=num_classes, ignore_mismatched_sizes=True).to(device)
    model2 = ViTForImageClassification.from_pretrained("facebook/dino-vits16", num_labels=num_classes, ignore_mismatched_sizes=True).to(device)

    optimizer1 = optim.Adam(model1.parameters(), lr=LEARNING_RATE)
    optimizer2 = optim.Adam(model2.parameters(), lr=LEARNING_RATE)

    print("🚀 Training Model 1 (ViT Base)")
    train_model(model1, optimizer1, train_loader, "model1")

    print("🚀 Training Model 2 (DINO-ViT)")
    train_model(model2, optimizer2, train_loader, "model2")

    print("🔎 Evaluating Ensemble...")
    evaluate_ensemble([model1, model2], test_loader)

if __name__ == '__main__':
    main()
