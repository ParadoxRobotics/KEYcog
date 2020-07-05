
import numpy as np
import cv2

img = cv2.imread('target.jpg',1)
gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

ret, result_bin = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)
# erosion + dilation => opening suppression
result_bin = cv2.morphologyEx(result_bin, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
# CV_8U conversion# UDP Communication thread with Daemon form
thresh_img = np.uint8(result_bin)
#Contours
contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

img2 = np.full((img.shape), 255)
img2 = img2.astype(np.uint8)
index = -1 #if negative all contours are drawn, index of contour to be drawn
thickness = 8
color = (0,0,255)
contours = max(contours, key = cv2.contourArea)
contours = np.delete(contours, 0, 0)
for i in range(len(contours)):
    cv2.circle(img2, (contours[i,0,0],contours[i,0,1]), radius=8, color=(0,0,255), thickness=-1, lineType=8, shift=0)
cv2.imwrite('/contour.jpg', img2)
cv2.imshow('Contours', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
