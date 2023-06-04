import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# img2=cv.imread("D:\Learn_project\matcher\cat1.jpg",0) #读取第二个图像（大图像）
# cat_face=img2[139:419,832:1130]
# black_img=np.zeros((853,1280),dtype=np.uint8) #创建一个全黑图像
# black_img[139:419,832:1130]=cat_face
# cv.imshow('1',black_img)
# cv.waitKey(0)


# MIN_MATCH_COUNT=10 # 设置最低匹配数量为10
img1=cv.imread("D:\Learn_project\matcher\sanjiaoxing22.jpg") #读取第一个图像（小图像）
img2=cv.imread("D:\Learn_project\matcher\sanjiaoxing.jpg") #读取第二个图像（大图像）
img1=cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
gray2=cv.cvtColor(img2,cv.COLOR_BGR2GRAY)



corners=cv.goodFeaturesToTrack(gray2,12,0.01,10)
#int0就是int64
corners=np.int0(corners)
#标出来
for i in corners:
    x,y=i.ravel()
    # print(x,y)
    cv.circle(img2,(x,y),3,255,-1)
plt.imshow(img2),plt.show()