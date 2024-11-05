import cv2
import numpy as np
import matplotlib.pyplot as plt

# 定義 Sobel 濾波器
def sobel_filter(image):
    # 定義 Sobel 區域
    sobel_x = np.array([[2,  1, 0, -1, -2],
                    [4,  2, 0, -2, -4],
                    [6,  0, 0,  0, -6],
                    [4, -2, 0,  2,  4],
                    [2, -1, 0,  1,  2]], dtype=np.float32)

    sobel_y = np.array([[2,  4, 6,  4, 2],
                    [1,  2, 0, -2, -1],
                    [0,  0, 0,  0, 0],
                    [-1, -2, 0,  2,  1],
                    [-2, -4, -6, -4, -2]], dtype=np.float32)
    # 應用濾波器
    grad_x = cv2.filter2D(image, cv2.CV_64F, sobel_x)
    grad_y = cv2.filter2D(image, cv2.CV_64F, sobel_y)

    # 計算梯度幅值
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return magnitude

# 定義 Laplacian 濾波器 (4x4 核)
def laplacian_filter(image):
    # 定義 Laplacian 區域 (4x4 核)
    laplacian = np.array([[0,  1, 0,  1, 0],
                      [1, -4, 1, -4, 1],
                      [0,  1, 0,  1, 0],
                      [1, -4, 1, -4, 1],
                      [0,  1, 0,  1, 0]], dtype=np.float32)

    # 應用濾波器
    laplacian_result = cv2.filter2D(image, cv2.CV_64F, laplacian)
    
    return laplacian_result

# 定義高增強方法
def high_boost(image, alpha=1.0):


     # 使用高斯濾波進行去噪
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    # 轉換為灰度圖
    gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)

    # 使用 Sobel 濾波器和 Laplacian 濾波器
    sobel_edges = sobel_filter(gray_image)
    laplacian_edges = laplacian_filter(gray_image)

    # 將 Sobel 和 Laplacian 邊緣結合
    edges = np.clip(sobel_edges + laplacian_edges, 0, 255).astype(np.uint8)

    # 高增強圖像
    high_boost_image = cv2.addWeighted(gray_image, 1 + alpha, edges, -alpha, 0)
    return gray_image, sobel_edges, laplacian_edges, high_boost_image

# 主函數
if __name__ == "__main__":
    # 讀取圖像
    image = cv2.imread('Bodybone.bmp')  # 替換為您的圖像路徑

    if image is None:
        print("無法讀取圖像，請檢查文件路徑。")
    else:
        # 應用高增強方法
        gray_image, sobel_edges, laplacian_edges, high_boost_image = high_boost(image)

        # 顯示圖像
        plt.figure(figsize=(12, 8))

        # 原始圖像
        plt.subplot(2, 2, 1)
        plt.title('Original Image')
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # 轉換顏色通道
        plt.axis('off')

        # Sobel 濾波結果
        plt.subplot(2, 2, 2)
        plt.title('Sobel Filter Result')
        plt.imshow(sobel_edges, cmap='gray')
        plt.axis('off')

        # Laplacian 濾波結果
        plt.subplot(2, 2, 3)
        plt.title('Laplacian Filter Result')
        plt.imshow(laplacian_edges, cmap='gray')
        plt.axis('off')

        # 高增強圖像
        plt.subplot(2, 2, 4)
        plt.title('High Boost Enhanced Image')
        plt.imshow(high_boost_image, cmap='gray')
        plt.axis('off')

        plt.tight_layout()
        plt.show()