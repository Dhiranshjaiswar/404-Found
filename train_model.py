import os
import torch
import timm
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
import pickle

# -----------------------------
# Load pretrained CNN (feature extractor)
# -----------------------------
model = timm.create_model('resnet18', pretrained=True, num_classes=0)
model.eval()

# -----------------------------
# Function to extract features
# -----------------------------
def extract_features(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))

    img = torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    with torch.no_grad():
        features = model(img)

    return features.numpy().flatten()

# -----------------------------
# Load dataset
# -----------------------------
X = []
y = []

real_path = "dataset_small/real"
fake_path = "dataset_small/fake"

# REAL images = label 0
for file in os.listdir(real_path):
    img_path = os.path.join(real_path, file)
    X.append(extract_features(img_path))
    y.append(0)

# FAKE images = label 1
for file in os.listdir(fake_path):
    img_path = os.path.join(fake_path, file)
    X.append(extract_features(img_path))
    y.append(1)

X = np.array(X)
y = np.array(y)

# -----------------------------
# Train ML model
# -----------------------------
clf = LogisticRegression(max_iter=1000)
clf.fit(X, y)

# -----------------------------
# Save model
# -----------------------------
with open("fake_detector.pkl", "wb") as f:
    pickle.dump(clf, f)

print("Model trained successfully!")