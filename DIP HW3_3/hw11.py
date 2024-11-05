import numpy as np
import cv2
import matplotlib.pyplot as plt

def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    生成高斯滤波核
    :param size: 核的大小（应为奇数）
    :param sigma: 高斯分布的标准差
    :return: 高斯核矩阵
    """
    k = size // 2
    x, y = np.mgrid[-k:k+1, -k:k+1]
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= kernel.sum()  # 归一化
    return kernel

# 设置核大小和sigma
kernel_size = 255  # 核大小为5x5
sigma = 64      # 高斯标准差

# 生成高斯滤波核
gaussian_kernel_matrix = gaussian_kernel(kernel_size, sigma)

# 生成高斯kernel
gaussian_kernel_result = gaussian_kernel(kernel_size, sigma)

# read the image
image = cv2.imread('checkerboard1024-shaded.tif', cv2.IMREAD_GRAYSCALE)
# print(np.shape(image))
# cv2.imwrite('checkerboard.png',image)


cvfilter = cv2.filter2D(image, -1, gaussian_kernel_result)
image2 = image/cvfilter


# show the original image and the processed image
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.subplot(1, 3, 2)
plt.title("Shaded Pattern")
plt.imshow(cvfilter,cmap='gray')
plt.subplot(1, 3, 3)
plt.title("Shaded Pattern")
plt.imshow(image2,cmap='gray')
plt.show()  



