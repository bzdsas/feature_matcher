import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import math

# 阈值分割中使用的均值平滑
def adaptive_thresh(image, win_size, ratio=0.15):
    # 对图像矩阵进行均值平滑
    image_mean = cv.blur(image, win_size)
    # 原图像矩阵与平滑结果做差
    out = image - (1.0-ratio) * image_mean
    # 当差值大于或等于0时，输出值为255，反之输出值为0
    out[out >= 0] = 255
    out[out < 0] = 0
    out = out.astype(np.uint8)
    return out

MIN_MATCH_COUNT=10 # 设置最低匹配数量为10
img1=cv.imread("D:\Learn_project\matcher\sanjiaoxing22.jpg",0) #读取第一个图像（小图像）
img2=cv.imread("D:\Learn_project\matcher\sanjiaoxing.jpg",0) #读取第二个图像（大图像）


'''----------1.特征匹配，找到特征所在像素坐标，并设为感兴趣区域ROI----------'''
sift=cv.xfeatures2d.SIFT_create() #创建sift检测器
kp1,des1=sift.detectAndCompute(img1,None)
kp2,des2=sift.detectAndCompute(img2,None)
# 创建设置FLAAN匹配
FLANN_INDEX_KDTREE=1
index_params=dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params=dict(checks=50)
flann=cv.FlannBasedMatcher(index_params,search_params)
mathces=flann.knnMatch(des1,des2,k=2)
good=[]
# 过滤不合格的匹配结果，大于0.7的都舍弃
for m,n in mathces:
    if m.distance<0.7*n.distance:
        good.append(m)
# 如果匹配结果大于10，则获取关键点的坐标，用于计算变换矩阵
if len(good)>MIN_MATCH_COUNT:
    src_pts=np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts =np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
# 计算变换矩阵和掩膜
    M,mask=cv.findHomography(src_pts,dst_pts,cv.RANSAC,10.0)
    matchesMask=mask.ravel().tolist()
# 根据变换矩阵进行计算，找到小图像在大图像中的位置
    h,w=img1.shape
    pts=np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
    dst=cv.perspectiveTransform(pts,M)
    # 画出特征所在区域
    # cv.polylines(img2,[np.int32(dst)],True,255,5,cv.LINE_AA)
else:
     print(" Not Enough matches are found")
     matchesMask=None
# #画出特征匹配线
# draw_params=dict(matchColor=(0,255,0),singlePointColor=None,
# matchesMask=matchesMask,flags=2)
# #plt展示最终的结果
# img3=cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
# # img3=cv.drawMatches(img1,kp1,img2,kp2,good,None)
# plt.imshow(img3),plt.show()
# print(dst)
x_min,y_min=dst.min(axis=0)[0,0],dst.min(axis=0)[0,1]   #axis=0(按列)
x_max,y_max=dst.max(axis=0)[0,0],dst.max(axis=0)[0,1]
x_min,y_min=math.floor(x_min),math.floor(y_min) #向下取整
x_max,y_max=math.ceil(x_max),math.ceil(y_max)   #向上取整
ROI=img2[y_min:y_max,x_min:x_max]   #感兴趣区域

'''----------2.开始处理感兴趣区域部分的图像效果----------'''
'''----------2.1图像平滑----------'''
ROI = cv.medianBlur(ROI,5)

'''----------2.2对比度增强----------'''
ROI = cv.resize(ROI, None, fx=0.5, fy=0.5)
# 创建CLAHE对象
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# 限制对比度的自适应阈值均衡化
ROI = clahe.apply(ROI)
# 分别显示原图，CLAHE
# cv.imshow("img", ROI)
# cv.imshow("dst", dst)
# cv.waitKey()

'''----------2.3阈值分割----------'''
# 目前先不使用阈值分割，效果有点差
# threshImg = adaptive_thresh(ROI, (3,3))
# cv.imshow('thresh', threshImg)
# cv.waitKey()


'''----------Shi-tomas拐角检测器----------'''
corners=cv.goodFeaturesToTrack(ROI,12,0.01,10)
#int0就是int64
corners=np.int0(corners)
three_key_point=[]  #存储三个关键特征点的列表
#标出来
for i in corners:
    x,y=i.ravel()
    three_key_point.append((x,y))   #存储关键特征点的像素坐标
    # print(x,y)
    cv.circle(ROI,(x,y),3,0,-1)     #将识别到的特征点画出来
# print(three_key_point)
# plt.imshow(ROI),plt.show()

# 查看识别到的特征点
cv.imshow("0",ROI)
cv.waitKey()

'''----------由三角形的三个特征点进行计算螺栓的坐标点----------'''

#处理计算完坐标后就跟处理三个坐标是一样的了