import torch
import timm
from PIL import Image
import torchvision.transforms as transforms

# Load pretrained model
model = timm.create_model('resnet18', pretrained=True)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load image
img = Image.open("test.jpg")
img = transform(img)
img = img.unsqueeze(0)

# Prediction (dummy for now)
with torch.no_grad():
    output = model(img)

print("Model output shape:", output.shape)