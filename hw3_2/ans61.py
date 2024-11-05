import cv2
import numpy as np

k0 = 0
k1 = 0.3
k2 = 0
k3 = 0.049

def statistical_measures(img, kernal_size):
    if kernal_size == 0:
        return cv2.meanStdDev(img)[0], cv2.meanStdDev(img)[1]
    else:
        i = kernal_size
        kernal = np.ones((i,i), np.float32)/(i**2)
        mean = cv2.filter2D(img.astype(np.float32), -1, kernal)
        var = cv2.filter2D((img.astype(np.float32)-mean)**2, -1, kernal)
        return mean, np.sqrt(var)

def boundary(a,b):
    a_lower_bound = k0*a
    a_upper_bound = k1*a
    b_lower_bound = k2*b
    b_upper_bound = k3*b
    return [a_lower_bound, a_upper_bound, b_lower_bound, b_upper_bound] 

def histogram_statistics(img, local_ms, max, b):
    width, height = img.shape

    for i in range(1, width):
        for j in range(1, height):
            local = img[i-1:i+2, j-1:j+2]
            local_max = np.max(local)
            if (b[0] < local_ms[0][i,j] < b[1]) and (b[2] < local_ms[1][i-1,j-1] < b[3]):
                c = round(max/local_max)
                img[i,j] = round(c*img[i,j])
    return img

if __name__ == '__main__':
    img = cv2.imread('hidden_object_2.jpg', cv2.IMREAD_GRAYSCALE)
    local = statistical_measures(img, 3)
    max = np.max(img)
    mean_std = statistical_measures(img, 0)
    
    # 使用 boundary 函数计算区间
    b = boundary(mean_std[0], mean_std[1])
    
    # 调用 histogram_statistics 方法处理图像
    img_histogram = histogram_statistics(img, local, max, b)
    
    # 保存处理后的图像
    cv2.imwrite('histogram_statistics_img.png', img_histogram)
    #show image
    cv2.imshow('image', img_histogram)
    cv2.waitKey(0)
    
    # 将结果传递给 Local Enhancement
    np.save('histogram_statistics_data.npy', img_histogram)  # 保存图像供下一步使用
    