
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
from scipy.linalg import logm, norm, eigvalsh, det, sqrtm
import cv2
from matplotlib import pyplot as plt
from matplotlib import pyplot

import time

# speed-up using multithreads
cv2.setUseOptimized(True);
cv2.setNumThreads(4);

# compute feature tesor F = [Hue, Sat, magnitude, angle] and the 1st/2nd order image integral
def computeFeature(img):
    # get intensity
    I = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # get RGB component
    R, G, B = cv2.split(img)
    # compute Hue and Saturation
    hsvImg = cv2.cvtColor(cv2.blur(img,(3,3)), cv2.COLOR_RGB2HSV)
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
    F = np.dstack((I, R, G, B, H, S, dx, dy, mag, angle))
    # compute P matrix H+1xW+1xD (first order integral image)
    IInt = cv2.integral(I, sdepth=cv2.CV_64F)
    RInt = cv2.integral(R, sdepth=cv2.CV_64F)
    GInt = cv2.integral(G, sdepth=cv2.CV_64F)
    BInt = cv2.integral(B, sdepth=cv2.CV_64F)
    HInt = cv2.integral(H, sdepth=cv2.CV_64F)
    SInt = cv2.integral(S, sdepth=cv2.CV_64F)
    dxInt = cv2.integral(dx, sdepth=cv2.CV_64F)
    dyInt = cv2.integral(dy, sdepth=cv2.CV_64F)
    magInt = cv2.integral(mag, sdepth=cv2.CV_64F)
    angleInt = cv2.integral(angle, sdepth=cv2.CV_64F)
    Pint = np.dstack((IInt, RInt, GInt, BInt, HInt, SInt, dxInt, dyInt, magInt, angleInt))
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

def computeConvarianceDistLogEuclide(CRef, CCur):
    return norm(logm(CRef) - logm(CCur), ord='fro')

def computeScale(roi, Pint, nbDim):
    # image size
    H = Pint.shape[1]
    W = Pint.shape[0]
    # roi shape
    x0 = roi[0]
    y0 = roi[1]
    Wr = roi[2]
    Hr = roi[3]
    # init scale matrix
    scale = np.zeros((nbDim, 2))
    # compute scale
    if H == min(H,W):
        ratio = Wr/Hr
        for i in range(0, nbDim):
            scale[i,:] = i*np.ceil(H/nbDim), i*np.ceil(ratio*H/nbDim)

    elif W == min(H,W):
        ratio = Hr/Wr
        for i in range(0, nbDim):
            scale[i,:] = i*np.ceil(ratio*W/nbDim), i*np.ceil(W/nbDim)
    return scale

def searchDescriptor(targetCov, targetRoi, PintTest, QintTest, nbDim, windowSize, stepSize):
    # location and cost of the windows
    cost = []
    # Test Pint size
    Wt = PintTest.shape[1]
    Ht = PintTest.shape[0]
    # Get windows size
    windowsSizeH = windowSize[0]
    windowsSizeW = windowSize[1]
    for D in range(1, nbDim):
        # for each spatial dimension recompute windows size
        windowsSizeW = int(windowSize[D,1])
        windowsSizeH = int(windowSize[D,0])
        # search over the image
        for H in range(0, Ht-stepSize, stepSize):
            for W in range(0, Wt-stepSize, stepSize):
                if W+windowsSizeW > Wt-1:
                    EW = abs(W-Wt)-1
                else:
                    EW = windowsSizeW
                if H+windowsSizeH > Ht-1:
                    EH = abs(H-Ht)-1
                else:
                    EH = windowsSizeH
                print([W, H, EW, EH])
                # compute test covariance
                testCov = computeConvariance(Pint=PintTest, Qint=QintTest, roi=[W, H, EW, EH])
                # if non null matrix compute distance
                if np.all(testCov != 0):
                    dist = computeConvarianceDistLogEuclide(CRef=targetCov[:,:,0], CCur=testCov)
                    cost.append([dist, W, H, EW, EH])
    return min(cost)

def RegionProposalSearchDescriptor(img, targetCov, PintTest, QintTest):
    # init cost
    cost = []
    # init regio proposal
    RP = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    # set input image on which we will run segmentation
    RP.setBaseImage(img)
    RP.switchToSelectiveSearchFast()
    # run selective search segmentation on input image
    roi = RP.process()
    # perform covariance recognition
    for i, R in enumerate(roi):
        testCov = computeConvariance(Pint=PintTest, Qint=QintTest, roi=R)
        # if non null matrix compute distance
        if np.all(testCov != 0):
            dist = computeConvarianceDistLogEuclide(CRef=targetCov[:,:,0], CCur=testCov)
            cost.append([dist, R[0], R[1], R[2], R[3]])
    return min(cost)

# get image and resize it
target = cv2.imread('02.jpg')
target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
target = cv2.resize(target, (320,240))

# get image and resize it
test = cv2.imread('01.jpg')
test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
test = cv2.resize(test, (320,240))

# get ROI
roi = cv2.selectROI(target) # ROI = [W0, H0, W, H]
roiLeft = [roi[0], roi[1], int(roi[2]/2), roi[3]]
roiRight = [roi[0]+int(roi[2]/2), roi[1], int(roi[2]/2), roi[3]]
roiUp = [roi[0], roi[1], roi[2], int(roi[3]/2)]
roiDown = [roi[0], roi[1]+int(roi[3]/2), roi[2], int(roi[3]/2)]
print(roi)
print(roiLeft)
print(roiRight)
print(roiUp)
print(roiDown)
cv2.destroyAllWindows()

# compute feature in region
FeatureTarget, PintTarget, QintTarget = computeFeature(img=target)
"""
print(FeatureTarget.shape)
plt.imshow(FeatureTarget[:,:,0])
plt.show()
plt.imshow(FeatureTarget[:,:,1])
plt.show()
plt.imshow(FeatureTarget[:,:,2])
plt.show()
plt.imshow(FeatureTarget[:,:,3])
plt.show()
plt.imshow(FeatureTarget[:,:,4])
plt.show()
plt.imshow(FeatureTarget[:,:,5])
plt.show()
plt.imshow(FeatureTarget[:,:,6])
plt.show()
plt.imshow(FeatureTarget[:,:,7])
plt.show()
plt.imshow(FeatureTarget[:,:,8])
plt.show()
plt.imshow(FeatureTarget[:,:,9])
plt.show()
"""
# compute feature in region
FeatureTest, PintTest, QintTest = computeFeature(img=test)
"""
print(FeatureTest.shape)
plt.imshow(FeatureTest[:,:,0])
plt.show()
plt.imshow(FeatureTest[:,:,1])
plt.show()
plt.imshow(FeatureTest[:,:,2])
plt.show()
plt.imshow(FeatureTest[:,:,3])
plt.show()
plt.imshow(FeatureTest[:,:,4])
plt.show()
plt.imshow(FeatureTest[:,:,5])
plt.show()
plt.imshow(FeatureTest[:,:,6])
plt.show()
plt.imshow(FeatureTest[:,:,7])
plt.show()
plt.imshow(FeatureTest[:,:,8])
plt.show()
plt.imshow(FeatureTest[:,:,9])
plt.show()
"""
# compute region covariance matrix of the region feature with the integral representation
targetCovFull = computeConvariance(Pint=PintTarget, Qint=QintTarget, roi=roi)
targetCovLeft = computeConvariance(Pint=PintTarget, Qint=QintTarget, roi=roiLeft)
targetCovRight = computeConvariance(Pint=PintTarget, Qint=QintTarget, roi=roiRight)
targetCovUp = computeConvariance(Pint=PintTarget, Qint=QintTarget, roi=roiUp)
targetCovDown = computeConvariance(Pint=PintTarget, Qint=QintTarget, roi=roiDown)
targetCov = np.dstack((targetCovFull, targetCovLeft, targetCovRight, targetCovUp, targetCovDown))

plt.imshow(targetCov[:,:,0])
plt.show()
plt.imshow(targetCov[:,:,1])
plt.show()
plt.imshow(targetCov[:,:,2])
plt.show()
plt.imshow(targetCov[:,:,3])
plt.show()
plt.imshow(targetCov[:,:,4])
plt.show()

# perform Brute Force search on the test image
windowSize = computeScale(roi=roi, Pint=PintTarget, nbDim=9)
print(windowSize)

start_time = time.time()
pose = searchDescriptor(targetCov=targetCov, targetRoi=roi, PintTest=PintTest, QintTest=QintTest, nbDim=9, windowSize=windowSize, stepSize=10)
print((time.time() - start_time))
cv2.rectangle(test, (pose[1], pose[2]), (pose[1]+pose[3], pose[2]+pose[4]), (0,0,255), 2)

start_time = time.time()
pose = RegionProposalSearchDescriptor(img=cv2.cvtColor(test, cv2.COLOR_BGR2RGB), targetCov=targetCov, PintTest=PintTest, QintTest=QintTest)
print((time.time() - start_time))
cv2.rectangle(test, (pose[1], pose[2]), (pose[1]+pose[3], pose[2]+pose[4]), (255,0,0), 2)

plt.imshow(test)
plt.show()
