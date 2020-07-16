# Robust, efficient and simple single objet recognition for robot application
# this work is based on covariant matrix descriptor and deformable template matching (DTM)
# Author : Munch Quentin, 2020.

"""
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
# Robust, efficint and simple single objet recognition for robot application
# Author : Munch Quentin, 2020.
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import pyplot

# compute the region feature RF = [Hue, Sat, magnitude, angle]
def computeFeature(region):
    # compute Hue and Saturation
    hsvImg = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsvImg)
    # compute gradient
    dx = cv2.Sobel(cv2.cvtColor(region, cv2.COLOR_RGB2GRAY), cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(cv2.cvtColor(region, cv2.COLOR_RGB2GRAY), cv2.CV_32F, 0, 1)
    # compute gradient magnitude and angle (in degree)
    mag = cv2.magnitude(dx, dy)
    # thresholding the magnitude value
    thresh = 50
    ret, magMask = cv2.threshold(mag, thresh, 255, cv2.THRESH_BINARY)
    magMask = magMask.astype(np.uint8)
    # apply the mask to the magnitude
    mag = cv2.bitwise_and(mag, mag, mask=magMask)
    # compute the angle given the thresholding magnitude mask
    angle = cv2.phase(dx, dy, angleInDegrees=True)
    angle = cv2.bitwise_and(angle, angle, mask=magMask)
    # region feature shape:  RF = [RH, RW, nbFeature]
    return np.dstack((H, S, mag, angle))

def computeConvariance(RF):
    # compute mean for each channel
    featureMean = np.sum(RF,(0,1))/(RF.shape[0]*RF.shape[1])
    # compute convariance
    Cov = np.zeros((RF.shape[2], RF.shape[2]))
    for i in range(RF.shape[0]):
        for j in range(RF.shape[1]):
            m=np.mat(RF[i,j,:]-featureMean)
            Cov=Cov+np.dot(m.T,m)
    return Cov/(RF.shape[0]*RF.shape[1]-1)

def computeConvarianceDist(CRef, CCur):
    return np.linalg.norm(logm(CRef) - logm(CCur), ord='fro')

# get image and resize it
img = cv2.imread('target_.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (640,480))

# get ROI
roi = cv2.selectROI(img)
cv2.destroyAllWindows()
region = img[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]

# compute feature in region
RegionFeature = computeFeature(region)
# compute the covariance matrix of the region feature
RegionCovariance = computeConvariance(RegionFeature)

plt.imshow(RegionCovariance)
plt.show()

# compute image integral
#imgSum, imgSqsum, imgTilted = cv2.integral3(imgGray)
