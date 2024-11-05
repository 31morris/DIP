import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.morphology as morp
from skimage.filters import rank

# Load the image
img = cv2.imread('hidden_object_2.jpg', cv2.IMREAD_GRAYSCALE)

# Local Equalization, disk shape kernel
kernel = morp.square(3)
img_local = rank.equalize(img, footprint=kernel)
clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(10,10))

# Apply CLAHE to the grayscale image to enhance contrast
equalized = clahe.apply(img)

# Normalize the enhanced image to 8-bit range
enhanced = cv2.normalize(equalized, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Display the original and enhanced images using plt
fig, (ax_img, ax_enhanced) = plt.subplots(1, 2)

ax_img.imshow(img, cmap=plt.cm.gray)
ax_img.set_title('Low contrast image')
ax_img.set_axis_off()

ax_enhanced.imshow(enhanced, cmap=plt.cm.gray)
ax_enhanced.set_title('Local equalization')
ax_enhanced.set_axis_off()

plt.show()