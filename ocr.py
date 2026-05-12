import easyocr

# Create OCR reader
reader = easyocr.Reader(['en'])

# Image path
image_path = "test.jpg"

# Read text from image
result = reader.readtext(image_path)

# Store text
text = ""

for detection in result:
    text += detection[1] + " "

print("Extracted Text:")
print(text)