import torch
import timm
import numpy as np
import pickle
from PIL import Image

# Load CNN feature extractor
model = timm.create_model('resnet18', pretrained=True, num_classes=0)
model.eval()

# Load trained ML model
clf = pickle.load(open("fake_detector.pkl", "rb"))

def extract_features_from_image(image_path):
    img = Image.open(image_path).convert("RGB")

    img = img.resize((224, 224))

    img = torch.tensor(np.array(img)) \
        .permute(2, 0, 1) \
        .unsqueeze(0) \
        .float() / 255.0

    with torch.no_grad():
        feat = model(img)

    return feat.numpy().flatten()

# Ask user for image path
image_path = input("Enter image path: ")

# Extract features
features = extract_features_from_image(image_path)

# Predict
prediction = clf.predict([features])[0]

if prediction == 0:
    print("✅ REAL IMAGE")
else:
    print("⚠️ FAKE IMAGE")