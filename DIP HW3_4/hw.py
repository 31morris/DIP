import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 读取图像
image = cv2.imread('Bodybone.bmp', cv2.IMREAD_GRAYSCALE)

# 检查图像是否成功读取
if image is None:
    print("Error: Unable to read the image. Please check the file path.")
    exit()

# 2. 应用 Sobel 滤波器
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # 水平方向边缘
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # 垂直方向边缘
sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)  # 边缘强度

# 3. 应用 Laplacian 滤波器
laplacian = cv2.Laplacian(image, cv2.CV_64F)

# 4. 设计高提升滤波器（使用原图和增强细节）
alpha = 1.5  # 高提升因子，控制增强程度
high_boost = image + alpha * (sobel_magnitude + laplacian)

# 5. 结果转换为可显示的格式
high_boost = np.clip(high_boost, 0, 255).astype(np.uint8)

# 6. 显示原图与增强后的图像
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.subplot(1, 2, 2)
plt.title('High Boost Enhanced Image')
plt.imshow(high_boost, cmap='gray')
plt.show()
