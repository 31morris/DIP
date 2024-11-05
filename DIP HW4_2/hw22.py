import cv2
import numpy as np
import matplotlib.pyplot as plt

def notch_reject_filter(shape, d0=9, u_k=0, v_k=0):
    P, Q = shape
    # Initialize filter with zeros
    H = np.zeros((P, Q))

    # Traverse through filter
    for u in range(0, P):
        for v in range(0, Q):
            # Get euclidean distance from point D(u,v) to the center
            D_uv = np.sqrt((u - P / 2 + u_k) ** 2 + (v - Q / 2 + v_k) ** 2)
            D_muv = np.sqrt((u - P / 2 - u_k) ** 2 + (v - Q / 2 - v_k) ** 2)

            if D_uv <= d0 or D_muv <= d0:
                H[u, v] = 0.0
            else:
                H[u, v] = 1.0

    return H

img = cv2.imread('car-moire-pattern.tif', 0)
print(img.shape)

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
phase_spectrumR = np.angle(fshift)
magnitude_spectrum = 20*np.log(np.abs(fshift))

img_shape = img.shape

H1 = notch_reject_filter(img_shape, 4, 38, 30)
H2 = notch_reject_filter(img_shape, 4, -42, 27)
H3 = notch_reject_filter(img_shape, 2, 80, 30)
H4 = notch_reject_filter(img_shape, 2, -82, 28)

NotchFilter = H1*H2*H3*H4
NotchRejectCenter = fshift * NotchFilter 
NotchReject = np.fft.ifftshift(NotchRejectCenter)
inverse_NotchReject = np.fft.ifft2(NotchReject)  # Compute the inverse DFT of the result

Result = np.abs(inverse_NotchReject)

# Display the original
plt.figure(figsize=(18, 8))
plt.subplot(1, 4, 1)
plt.title("Original Image")
plt.imshow(img, cmap='gray')


# frequency domain
plt.subplot(1, 4, 2)
plt.title("Frequency Domain Image")
plt.imshow(magnitude_spectrum, cmap='gray')

plt.subplot(1, 4, 3)
plt.title("Notch  Reject Filter in Frequency Domain")
plt.imshow(magnitude_spectrum*NotchFilter, "gray") 

plt.subplot(1, 4, 4)
plt.title("Filtered Image (Notch Filter)")
plt.imshow(Result, "gray") 

plt.show()