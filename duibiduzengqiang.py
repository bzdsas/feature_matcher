import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


img2=cv.imread("D:\Learn_project\matcher\cat1.jpg",0) #读取第二个图像（大图像）
cat_face=img2[139:419,832:1130]


# 1.直方图正规化--对猫脸处理没变化

# 计算原图中出现的最小灰度级和最大灰度级
# 使用函数计算
Imin, Imax = cv.minMaxLoc(cat_face)[:2]
# 使用numpy计算
# Imax = np.max(img)
# Imin = np.min(img)
Omin, Omax = 0, 255
# 计算a和b的值
a = float(Omax - Omin) / (Imax - Imin)
b = Omin - a * Imin
out = a * cat_face + b
out = out.astype(np.uint8)
cv.imshow("img", cat_face)
cv.imshow("out", out)

#原图像灰度图
h1, w1 = cat_face.shape[:2]
pixelSequence1 = cat_face.reshape([h1 * w1, ])
numberBins1 = 256
histogram1, bins1, patch1 = plt.hist(pixelSequence1, numberBins1,
                                  facecolor='black', histtype='bar')
plt.xlabel("1gray label")
plt.ylabel("1number of pixels")
plt.axis([0, 255, 0, np.max(histogram1)])
plt.show()

# 现图像灰度图
h2, w2 = out.shape[:2]
pixelSequence2 = out.reshape([h2 * w2, ])
numberBins2 = 256
histogram2, bins2, patch2 = plt.hist(pixelSequence2, numberBins2,
                                  facecolor='black', histtype='bar')
plt.xlabel("2gray label")
plt.ylabel("2number of pixels")
plt.axis([0, 255, 0, np.max(histogram2)])
plt.show()

cv.waitKey(0)

