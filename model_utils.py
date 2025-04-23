import torch
from torchvision import transforms
from transformers import ViTForImageClassification
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models(num_classes=4):
    model1 = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", num_labels=num_classes, ignore_mismatched_sizes=True)
    model2 = ViTForImageClassification.from_pretrained("facebook/dino-vits16", num_labels=num_classes, ignore_mismatched_sizes=True)
    
    model1.load_state_dict(torch.load("saved_model/model1.pth", map_location=device))
    model2.load_state_dict(torch.load("saved_model/model2.pth", map_location=device))
    
    model1.to(device).eval()
    model2.to(device).eval()
    return model1, model2

def predict_image(image, model1, model2, class_names):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output1 = model1(image_tensor).logits
        output2 = model2(image_tensor).logits
        avg_output = torch.mean(torch.stack([output1, output2]), dim=0)
        probabilities = torch.nn.functional.softmax(avg_output, dim=1)
        predicted_class = torch.argmax(probabilities).item()

    return predicted_class, probabilities.squeeze().cpu().numpy()
