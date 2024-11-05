import cv2
import numpy as np

if __name__ == '__main__':
    # 载入 Histogram Statistics 程序保存的图像数据
    img_histogram = np.load('histogram_statistics_data.npy')

    # Local enhancement method
    clahe = cv2.createCLAHE()
    clahe_img = clahe.apply(img_histogram)
    
    #show the result
    cv2.imshow('CLAHE', clahe_img)
    cv2.waitKey(0)
