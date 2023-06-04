import cv2 as cv
import numpy as np

def guideFilter(I, p, winSize, eps, s):
    # 输入图像的高、宽
    h, w = I.shape[:2]

    # 缩小图像
    size = (int(round(w * s)), int(round(h * s)))
    small_I = cv.resize(I, size, interpolation=cv.INTER_CUBIC)
    small_p = cv.resize(I, size, interpolation=cv.INTER_CUBIC)

    # 缩小滑动窗口
    X = winSize[0]
    small_winSize = (int(round(X * s)), int(round(X * s)))

    # I的均值平滑 p的均值平滑
    mean_small_I = cv.blur(small_I, small_winSize)
    mean_small_p = cv.blur(small_p, small_winSize)

    # I*I和I*p的均值平滑
    mean_small_II = cv.blur(small_I * small_I, small_winSize)
    mean_small_Ip = cv.blur(small_I * small_p, small_winSize)

    # 方差、协方差
    var_small_I = mean_small_II - mean_small_I * mean_small_I
    cov_small_Ip = mean_small_Ip - mean_small_I * mean_small_p

    small_a = cov_small_Ip / (var_small_I + eps)
    small_b = mean_small_p - small_a * mean_small_I

    # 对a、b进行均值平滑
    mean_small_a = cv.blur(small_a, small_winSize)
    mean_small_b = cv.blur(small_b, small_winSize)

    # 放大
    size1 = (w, h)
    mean_a = cv.resize(mean_small_a, size1, interpolation=cv.INTER_LINEAR)
    mean_b = cv.resize(mean_small_b, size1, interpolation=cv.INTER_LINEAR)

    q = mean_a * I + mean_b

    return q


img2=cv.imread(r"D:\Learn_project\matcher\feture1.jpg",0) #读取第二个图像（大图像）
# cat_face=img2[139:419,832:1130]
# 1.中值平滑
img_ret1 = cv.medianBlur(img2,3)
# img_ret2 = cv.medianBlur(img2,5)
# img_ret3 = cv.medianBlur(img2,11)
cv.imshow('1',img2)
# cv.imshow('2',img_ret1)
# cv.imshow('3',img_ret2)
# cv.imshow('4',img_ret3)
# 2.双边滤波
result1 = cv.bilateralFilter(img2,33,50,25/2)
result2 = cv.bilateralFilter(img2,5,200,200)
cv.imshow('5',result1)
cv.imshow('6',result2)
# 3.导向滤波
I = img2/255.0
p = I
guideFilter_img = guideFilter(I, p, (5,5), 0.1,3)
# cv.imshow('7',guideFilter_img)

cv.waitKey(0)