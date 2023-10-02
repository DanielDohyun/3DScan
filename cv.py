import cv2
import numpy as np

# Load an example image (you should replace this with your 3D scan)
image_path = "cartCase1.jpeg"
image = cv2.imread(image_path)

# Convert the image to grayscale (assuming it's not already)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Define a threshold to create a mask
threshold = 128
ret, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

# Invert the mask if needed
# mask = cv2.bitwise_not(mask)

# Apply the mask to the original image
masked_image = cv2.bitwise_and(image, image, mask=mask)

# Display the original and masked images
cv2.imshow("Original Image", image)
cv2.imshow("Masked Image", masked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# import cv2
# import numpy as np

# # Load the 3D scan image of the cartridge case
# image_path = "cartridge_case.jpg"
# image = cv2.imread(image_path)

# # Convert the image to grayscale (assuming it's not already)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Apply Gaussian blur to reduce noise and improve thresholding
# blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# # Perform adaptive thresholding to create a mask
# mask = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

# # Invert the mask if needed
# # mask = cv2.bitwise_not(mask)

# # Apply the mask to the original image
# masked_image = cv2.bitwise_and(image, image, mask=mask)

# # Display the original and masked images
# cv2.imshow("Original Image", image)
# cv2.imshow("Masked Image", masked_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
