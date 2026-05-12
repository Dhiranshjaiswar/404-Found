import cv2
import matplotlib.pyplot as plt

# Read image
image = cv2.imread("test.jpg")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Resize image
resized = cv2.resize(gray, (500, 500))

# Display image
plt.imshow(resized, cmap='gray')

# Hide axes
plt.axis('off')

# Show image
plt.show()