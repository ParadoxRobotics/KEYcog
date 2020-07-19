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
from scipy.linalg import logm, norm
import cv2
from matplotlib import pyplot as plt
from matplotlib import pyplot

# compute feature tesor F = [Hue, Sat, magnitude, angle] and the 1st/2nd order image integral
def computeFeature(img):
    # compute Hue and Saturation
    hsvImg = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsvImg)
    # compute gradient
    dx = cv2.Sobel(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.CV_32F, 0, 1)
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
    # construct feature tensor
    F = np.dstack((H, S, mag, angle))
    # compute P matrix H+1xW+1xD (first order integral image)
    HInt = cv2.integral(H, sdepth=cv2.CV_64F)
    SInt = cv2.integral(S, sdepth=cv2.CV_64F)
    magInt = cv2.integral(mag, sdepth=cv2.CV_64F)
    angleInt = cv2.integral(angle, sdepth=cv2.CV_64F)
    Pint = np.dstack((HInt, SInt, magInt, angleInt))
    # compute Q matrix H+1xW+1xDxD (second order integral image)
    Qint = np.zeros((F.shape[0]+1, F.shape[1]+1, int((F.shape[2] * (F.shape[2] + 1)) / 2 )))
    idx = 0
    for i in range(F.shape[2]):
        for j in range(i, F.shape[2]):
            mult = F[:,:,i]*F[:,:,j]
            FeatureInt = cv2.integral(mult, sdepth=cv2.CV_64F)
            Qint[:,:,idx] = FeatureInt
            idx = idx+1
    # F = [H, W, nbFeature], Q matrix anf P matrix
    return F, Pint, Qint

def computeConvariance(Pint, Qint, roi):
    # region info
    x0 = roi[0]
    y0 = roi[1]
    W = roi[2]
    H = roi[3]
    # init covariance matrix
    covariance = np.zeros((Pint.shape[2], Pint.shape[2]))
    n0 = (1/((H*W)-1))
    n1 = (1/(H*W))
    # index init
    idx = 0
    # compute covariance matrix using integral image representation
    for i in range(Pint.shape[2]):
        for j in range(i, Pint.shape[2]):
            q = Qint[y0, x0, idx] + Qint[y0+H, x0+W, idx] - Qint[y0, x0+W, idx] - Qint[y0+H, x0, idx]
            p0 = (Pint[y0, x0, i] + Pint[y0+H, x0+W, i] - Pint[y0, x0+W, i] - Pint[y0+H, x0, i])/H*W
            p1 = (Pint[y0, x0, j] + Pint[y0+H, x0+W, j] - Pint[y0, x0+W, j] - Pint[y0+H, x0, j])/H*W
            covariance[i,j] = n0*(q-(n1*p0*p1))
            idx = idx+1
            if(i!=j):
                covariance[j,i]=covariance[i,j]

    return covariance

def computeConvarianceRegion(RF):
    # compute mean for each channel
    featureMean = np.sum(RF,(0,1))/(RF.shape[0]*RF.shape[1])
    # compute convariance (dxd)
    Cov = np.zeros((RF.shape[2], RF.shape[2]))
    for i in range(RF.shape[0]):
        for j in range(RF.shape[1]):
            m=np.mat(RF[i,j,:]-featureMean)
            Cov=Cov+np.dot(m.T,m)
    return Cov/(RF.shape[0]*RF.shape[1]-1)

def computeConvarianceDist(CRef, CCur):
    return norm(logm(CRef) - logm(CCur), ord='fro')

# get image and resize it
img = cv2.imread('target.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (640,480))

# get ROI
roi = cv2.selectROI(img)
print(roi)
cv2.destroyAllWindows()

# compute feature in region
Feature, Pint, Qint = computeFeature(img)

# compute covariance on the region
region = Feature[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2]), :]
regionCovariance = computeConvarianceRegion(region)
plt.imshow(regionCovariance)
plt.show()

# compute region covariance matrix of the region feature with the integral representation
regionCovarianceInt = computeConvariance(Pint, Qint, roi)
plt.imshow(regionCovarianceInt)
plt.show()

# compute distance
dist = computeConvarianceDist(regionCovariance, regionCovarianceInt)
print(dist)
