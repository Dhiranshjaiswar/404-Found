import torch
import timm
import easyocr
from PIL import Image
import torchvision.transforms as transforms

# -----------------------------
# 1. OCR SETUP
# -----------------------------
reader = easyocr.Reader(['en'])

# -----------------------------
# 2. PRETRAINED IMAGE MODEL
# -----------------------------
model = timm.create_model('resnet18', pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -----------------------------
# 3. LOAD IMAGE
# -----------------------------
img_path = "test.jpg"
img = Image.open(img_path).convert("RGB")

# -----------------------------
# 4. OCR PROCESSING
# -----------------------------
ocr_result = reader.readtext(img_path, detail=0)
extracted_text = " ".join(ocr_result)

# -----------------------------
# 5. IMAGE FEATURE EXTRACTION
# -----------------------------
img_tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    features = model(img_tensor)

# -----------------------------
# 6. OUTPUT
# -----------------------------
print("\n==============================")
print("OCR EXTRACTED TEXT:")
print(extracted_text)

print("\nIMAGE FEATURE SHAPE:")
print(features.shape)

print("\n==============================")
print("NOTE: This is CV + OCR pipeline (no final fake decision yet)")
# -----------------------------
# 7. SIMPLE DECISION LOGIC
# -----------------------------

text_len = len(extracted_text)

# Fake indicators (very basic baseline logic)
fake_score = 0

# Rule 1: very little or no text
if text_len < 10:
    fake_score += 1

# Rule 2: image exists but no meaningful OCR
if text_len < 20:
    fake_score += 1

# Rule 3: always add small uncertainty from image model
fake_score += 1  # placeholder CV signal

print("\n==============================")

if fake_score >= 2:
    print("FINAL RESULT: ⚠ POSSIBLY FAKE / SUSPICIOUS")
else:
    print("FINAL RESULT: ✅ LIKELY REAL")

print("==============================")