import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
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

# # 1.哈里斯角检测
# filename="D:\Learn_project\matcher\sanjiaoxing.jpg"
# img=cv.imread(filename)
# img1=cv.imread(filename)
# gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# #需要float32精度才能在cv.cornerHarris中运行
# gray=np.float32(gray)
# #哈里斯角点检测（输入灰度图像，拐角的邻域大小，Sobel导数的光圈参数，哈里斯检测器自由参数）
# dst=cv.cornerHarris(gray,2,9,0.01)
# #对角点进行膨胀
# dst=cv.dilate(dst,None)
# #标记
# img[dst>0.01*dst.max()]=[0,0,255]
# plt.imshow(img),plt.title("dst"),plt.xticks([]),plt.yticks([])
# plt.show()


# # 2.亚像素角检测
# filename="D:\Learn_project\matcher\sanjiaoxing.jpg"
# img=cv.imread(filename)
# img1=cv.imread(filename)
# gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# gray=np.float32(gray)
# dst=cv.cornerHarris(gray,2,9,0.01)
# dst=cv.dilate(dst,None)
# ret,dst=cv.threshold(dst,0.01*dst.max(),255,0)
# #上面是哈里斯角检测
# dst=np.uint8(dst)
# #找质心
# ret,labels,stats,centroids=cv.connectedComponentsWithStats(dst)
# #定义停止和完善拐角的条件
# criteria=(cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER,100,0.001)
# corners=cv.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
# res=np.hstack((centroids,corners))
# res=np.int0(res)
# #绘制，在matplotlib中显示会按照RGB顺序，CV2中是BGR
# img[res[:,1],res[:,0]]=[0,0,255]
# img[res[:,3],res[:,2]]=[0,255,0]
# plt.imshow(img),plt.title("dst"),plt.xticks([]),plt.yticks([])
# plt.show()

# new_imgs2第三张和第五张图片识别有问题
# 3.Shi-tomas拐角检测器
img=cv.imread(r"D:\Learn_project\matcher\new_imgs3\feature-16-31-38.jpg")
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY) #（800×800）的图像
#（灰度图像，检测的角个数，0-1之间角的质量阈值，两个角之间的最小欧式距离）
'''平滑处理'''
# gray = cv.GaussianBlur(gray, (3, 3), 0)
gray = cv.medianBlur(gray,5)
# gray = cv.bilateralFilter(gray,33,50,25/2)
# gray = cv.bilateralFilter(gray,5,200,200)
'''对比度增强'''
#
# clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# 限制对比度的自适应阈值均衡化
# gray = clahe.apply(gray)


# fi = gray / 255.0
# # 伽马变换
# gamma = 0.4
# gray = np.power(fi, gamma)

# Imin, Imax = cv.minMaxLoc(gray)[:2]
# # 使用numpy计算
# # Imax = np.max(img)
# # Imin = np.min(img)
# Omin, Omax = 0, 255
# # 计算a和b的值
# a = float(Omax - Omin) / (Imax - Imin)
# b = Omin - a * Imin
# gray = a * gray + b
# gray = gray.astype(np.uint8)
'''阈值分割'''

# thresh, gray = otsu_thresh(gray)
# print(thresh)
# cv.imshow('thresh', gray)
# cv.waitKey()
# gray = cv.Canny(gray, 50, 150)

# corners=cv.goodFeaturesToTrack(gray,4,0.01,10)
corners=cv.goodFeaturesToTrack(gray,4,0.01,100)
key_point=[]  #存储关键特征点的列表
new_key_point=[]
#int0就是int64
corners=np.int0(corners)
#标出来
# gn = torch.tensor(gray.shape)[[1, 0, 1, 0]]
for i in corners:
    x,y=i.ravel()
    # x, y = (torch.tensor(i.ravel()).view(1,2)/gn).view(-1).tolist
    # print(x,y)
    key_point.append((x,y))  # 存储关键特征点的像素坐标
    # key_point.append((float(format(x, '.4f')), float(format(y, '.4f'))))  # 存储关键特征点的像素坐标
    # cv.circle(img,(x,y),3,255,-1)
    cv.circle(img, (x, y), 10, 255, -1)
# cv.circle(img, (400, 400), 20, 255, -1)
plt.imshow(img),plt.show()
# key_point.sort(key=lambda x: x[1], reverse=False)
# a=0
# while a <=11:
#     x=key_point[a][0]
#     x+=100
#     y=key_point[a][1]
#     y+=100
#     new_key_point.append((x,y))
#     a+=1
# print(key_point)
# print(new_key_point)
# for k in new_key_point:
#     x = k[0]
#     y = k[1]
#     # cv.circle(img, (x, y), 3, 255, -1)
#     cv.circle(img, (int(x), int(y)), 3, 255, -1)
# plt.imshow(img),plt.show()