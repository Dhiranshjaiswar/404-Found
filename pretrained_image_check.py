import torch
import timm
from PIL import Image
import torchvision.transforms as transforms

model = timm.create_model('resnet18', pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

img = Image.open("test.jpg").convert("RGB")
img = transform(img).unsqueeze(0)

import torch.nn.functional as F

with torch.no_grad():
    output = model(img)
    probs = F.softmax(output, dim=1)

# -------------------------
# CV SIGNALS
# -------------------------
confidence = torch.max(probs).item()
entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()

# -------------------------
# SIMPLE FUSION SCORE
# -------------------------
fake_score = 0

# low confidence → suspicious
if confidence < 0.5:
    fake_score += 1

# high uncertainty → suspicious
if entropy > 3.0:
    fake_score += 1

print("\n======================")
print("CV ANALYSIS")
print("Confidence:", confidence)
print("Entropy:", entropy)
print("Fake Score:", fake_score)

# -------------------------
# FINAL DECISION
# -------------------------
if fake_score >= 2:
    print("⚠ FINAL RESULT: POSSIBLY FAKE IMAGE")
else:
    print("✅ FINAL RESULT: LIKELY REAL IMAGE")

print("======================")

