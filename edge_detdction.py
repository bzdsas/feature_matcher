import cv2 as cv
import numpy as np
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


# original_img = cv2.imread(r"D:\Learn_project\matcher\save_img4000\feture-14-09-57.jpg", 0)
#
# # canny(): 边缘检测
# img1 = cv2.GaussianBlur(original_img, (3, 3), 0)
# canny = cv2.Canny(img1, 50, 150)
#
# # 形态学：边缘检测
# _, Thr_img = cv2.threshold(original_img, 210, 255, cv2.THRESH_BINARY)  # 设定红色通道阈值210（阈值影响梯度运算效果）
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 定义矩形结构元素
# gradient = cv2.morphologyEx(Thr_img, cv2.MORPH_GRADIENT, kernel)  # 梯度
#
# cv2.imshow("original_img", original_img)
# cv2.imshow("gradient", gradient)
# cv2.imshow('Canny', canny)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# # 2.亚像素角检测
img=cv.imread(r"D:\Learn_project\matcher\new_imgs\sibianxingimg\feature-09-44-29.jpg")
gray_img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
result_img=img.copy()
# gray_img = cv.bilateralFilter(gray_img,33,50,25/2)
# gray_img = cv.bilateralFilter(gray_img,5,200,200)
gray_img = cv.medianBlur(gray_img,5)
# gray_img = cv.GaussianBlur(gray_img, (3, 3), 0)
# kernel = np.ones((5,5),np.uint8)
# gray_img = cv.erode(gray_img,kernel,iterations = 1)

# thresh, gray_img = otsu_thresh(gray_img)

# gray_img = cv.Canny(gray_img, 50, 150)

#Shi-Tomasi角点检测
# corners=cv.goodFeaturesToTrack(gray_img,4,0.01,100,blockSize=3,useHarrisDetector=False,k=0.04)
corners=cv.goodFeaturesToTrack(gray_img,4,0.01,100,blockSize=11,useHarrisDetector=False,k=0.04)
#迭代算法模板（1.类型 2.迭代次数 3.阈值）
criteria=(cv.TermCriteria_EPS+cv.TermCriteria_MAX_ITER,30,0.01)
#亚像素角点检测
corners2=cv.cornerSubPix(gray_img,corners,(5,5),(-1,-1),criteria)
j=0
for i in corners2:
    x,y=i.ravel()
    j=j+1
    cv.circle(result_img,(x,y),2,(0,0,255),2)
    print("角点坐标"+str(j)+":",(x,y))
# cv.imshow("original image", img)
cv.imshow("result image",result_img)
# cv.imshow("gray image",gray_img)
#cv.imwrite("D:/testimage/result ma.jpg",result_img)
cv.waitKey(0)
cv.destroyAllWindows()



# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt
#
# img = cv.imread(r'D:\Learn_project\matcher\sanjiaoxing.jpg',0)
# # 使用默认值初始化FAST对象
# fast = cv.FastFeatureDetector_create()
#
# # 查找并绘制关键点
# kp = fast.detect(img,None)
# img2 = cv.drawKeypoints(img, kp, None, color=(255,0,0))
#
# # 打印所有默认参数
# print( "Threshold: {}".format(fast.getThreshold()) )
# print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
# print( "neighborhood: {}".format(fast.getType()) )
# print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp)) )
# # cv.imshow('fast_true.png',img2)
#
# # 禁用nonmaxSuppression
# fast.setNonmaxSuppression(0)
# kp = fast.detect(img,None)
# print( "Total Keypoints without nonmaxSuppression: {}".format(len(kp)) )
#
# img3 = cv.drawKeypoints(img, kp, None, color=(255,0,0))
# cv.imshow('fast_false.png',img3)
# cv.imshow('fast_true.png',img2)
# cv.waitKey(0)

