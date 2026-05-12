import cv2
import easyocr
import matplotlib.pyplot as plt

# Load image
image = cv2.imread("test.jpg")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Resize image
resized = cv2.resize(gray, (800, 800))

# Apply thresholding
_, threshold = cv2.threshold(resized, 150, 255, cv2.THRESH_BINARY)

# Show processed image
plt.imshow(threshold, cmap='gray')
plt.axis('off')
plt.show()

# OCR Reader
reader = easyocr.Reader(['en'])

# Read text
result = reader.readtext(threshold)

# Store extracted text
text = ""

for detection in result:
    text += detection[1] + " "

print("\nExtracted Text:")
print(text)