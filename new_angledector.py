import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import math

def calc_gray_hist(image):
    rows, cols = image.shape[:2]
    gray_hist = np.zeros([256], np.uint64)
    for i in range(rows):
        for j in range(cols):
            gray_hist[image[i][j]] += 1
    return gray_hist

def otsu_thresh(image):
    rows, cols = image.shape[:2]
    # 计算灰度直方图
    gray_hist = calc_gray_hist(image)
    # 归一化灰度直方图
    norm_hist = gray_hist / float(rows*cols)
    # 计算零阶累积矩, 一阶累积矩
    zero_cumu_moment = np.zeros([256], np.float32)
    one_cumu_moment = np.zeros([256], np.float32)
    for i in range(256):
        if i == 0:
            zero_cumu_moment[i] = norm_hist[i]
            one_cumu_moment[i] = 0
        else:
            zero_cumu_moment[i] = zero_cumu_moment[i-1] + norm_hist[i]
            one_cumu_moment[i] = one_cumu_moment[i - 1] + i * norm_hist[i]
    # 计算方差，找到最大的方差对应的阈值
    mean = one_cumu_moment[255]
    thresh = 0
    sigma = 0
    for i in range(256):
        if zero_cumu_moment[i] == 0 or zero_cumu_moment[i] == 1:
            sigma_tmp = 0
        else:
            sigma_tmp = math.pow(mean*zero_cumu_moment[i] - one_cumu_moment[i], 2) / (zero_cumu_moment[i] * (1.0-zero_cumu_moment[i]))
        if sigma < sigma_tmp:
            thresh = i
            sigma = sigma_tmp
    # 阈值分割
    thresh_img = image.copy()
    thresh_img[thresh_img>thresh] = 255
    thresh_img[thresh_img<=thresh] = 0
    return thresh, thresh_img

img=cv.imread(r"D:\Learn_project\matcher\feture1.jpg")
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY) #（800×800）的图像
# 平滑处理
gray = cv.medianBlur(gray,3)
# 阈值分割
thresh, gray = otsu_thresh(gray)
kernel = np.ones((5,5),np.uint8)
opening = cv.morphologyEx(gray, cv.MORPH_OPEN, kernel)
# print(thresh)

canny = cv.Canny(gray, 50, 150)

# 形态学：边缘检测
_, Thr_img = cv.threshold(img, 210, 255, cv.THRESH_BINARY)  # 设定红色通道阈值210（阈值影响梯度运算效果）
kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))  # 定义矩形结构元素
gradient = cv.morphologyEx(Thr_img, cv.MORPH_GRADIENT, kernel)  # 梯度

cv.imshow("original_img", img)
cv.imshow("gradient", gradient)
cv.imshow('Canny', canny)

cv.waitKey(0)
cv.destroyAllWindows()


# corners=cv.goodFeaturesToTrack(canny,4,0.1,220)
# key_point=[]  #存储关键特征点的列表
# new_key_point=[]
# #int0就是int64
# corners=np.int0(corners)
# #标出来
# # gn = torch.tensor(gray.shape)[[1, 0, 1, 0]]
# for i in corners:
#     x,y=i.ravel()
#     # x, y = (torch.tensor(i.ravel()).view(1,2)/gn).view(-1).tolist
#     # print(x,y)
#     key_point.append((x,y))  # 存储关键特征点的像素坐标
#     # key_point.append((float(format(x, '.4f')), float(format(y, '.4f'))))  # 存储关键特征点的像素坐标
#     # cv.circle(img,(x,y),3,255,-1)
#     cv.circle(img, (x, y), 3, 255, -1)
# # cv.circle(img, (400, 400), 20, 255, -1)
# plt.imshow(img),plt.show()

# cv.imshow('thresh', gray)
# cv.waitKey()



