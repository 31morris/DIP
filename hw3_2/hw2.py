import cv2
import numpy as np

# 讀取圖像
image = cv2.imread('hidden_object_2.jpg', cv2.IMREAD_GRAYSCALE)

# 創建 CLAHE 對象，設置對比度限制和格網大小
clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(16,16))
enhanced_image = clahe.apply(image)

# 顯示增強後的圖像
cv2.imshow('Enhanced Image - Local Enhancement', enhanced_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
#save the enhanced image
cv2.imwrite('enhanced_image.jpg', enhanced_image)