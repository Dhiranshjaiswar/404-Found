import torch
import timm
from PIL import Image
from torchvision import transforms

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = timm.create_model('resnet18', pretrained=False, num_classes=2)

model.load_state_dict(torch.load("ai_detector.pth", map_location=device))

model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Input image path
image_path = input("Enter image path: ")

# Load image
img = Image.open(image_path).convert("RGB")

img = transform(img).unsqueeze(0).to(device)

# Prediction
with torch.no_grad():
    outputs = model(img)
    prediction = torch.argmax(outputs, dim=1).item()

if prediction == 0:
    print("⚠ AI GENERATED IMAGE")
else:
    print("✅ REAL IMAGE")