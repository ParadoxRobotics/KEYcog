# ORBcog is object template matching method for fast object recognition.
# This code is based Li Yang Ku SURF object matching

import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import pyplot
from scipy.spatial import Delaunay, delaunay_plot_2d

# create base model
class targetModel():
    def __init__(self, nbPointMax):
        super(targetModel, self).__init__()
        # input
        self.targetImage = None # image to train on
        self.targetMask = None # target binary mask
        # image keypoint extractor
        self.orb = cv2.ORB_create(nfeatures=nbPointMax)
        # parameters for the probabilistic model
        self.pt = [] # keypoint [x,y] -> [W,H]
        self.angle = [] # angle of the keypoint
        self.scale = [] # size response of the keypoint (scale)
        self.response = [] # response from the descriptor
        self.angleCenter = [] # angle between the ORB keypoint and the center of the image
        self.scaleCenter = [] # length of the line between the ORB point and the middle point devide by the scale of the ORB descriptor
        self.descriptor = None

    def createModel(self, img, mask, imgCenter):
        # store target image and mask
        self.targetImage = img
        self.targetMask = mask
        # compute ORB keypoint and descriptor
        kp, des = self.orb.detectAndCompute(self.targetImage, None)
        # store the descriptor
        self.descriptor = des
        # draw keypoint
        kp_img = cv2.drawKeypoints(img, kp, None,  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.imshow(kp_img)
        plt.show()

        if self.targetMask is not None:
            # get point, angle and size from keypoints in the binary mask
            Hm, Wm = np.where(self.targetMask == 255)
            maskpoint = np.array((Wm, Hm)).T
            for i in range(0,len(kp)):
                if np.round(kp[i].pt) in maskpoint:
                    self.pt.append(kp[i].pt)
                    self.angle.append(kp[i].angle)
                    self.scale.append(kp[i].size)
                    self.response.append(kp[i].response)
        else:
            # get point, angle and size from keypoints
            for i in range(0,len(kp)):
                self.pt.append(kp[i].pt)
                self.angle.append(kp[i].angle)
                self.scale.append(kp[i].size)

        if self.targetMask is not None and imgCenter == False:
            centerW, centerH = np.argwhere(self.targetMask == 255).sum(0)/(self.targetMask == 255).sum()
            centerW = int(round(centerW))
            centerH = int(round(centerH))
        else:
            # compute the midle of the cropped image
            centerH = int(round(self.targetImage.shape[1]/2))
            centerW = int(round(self.targetImage.shape[0]/2))
        # for every ORB point
        for i in range(0, len(self.pt)):
            # store the angle of the line between the ORB point and the middle point
            self.angleCenter.append(np.arctan2(self.pt[i][1], self.pt[i][0])-(-self.angle[i]*np.pi/180))
            # store the length of the line between the ORB point and the middle point devide by the scale of the ORB descriptor
            self.scaleCenter.append(np.sqrt((centerW-self.pt[i][0])**2+(centerH-self.pt[i][1])**2)/self.scale[i])


img = cv2.imread('target.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
mask = cv2.imread('mask.png')
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

model = targetModel(5)
model.createModel(img, mask, True)

print("point")
print(model.pt)
print("scale")
print(model.scale)
print("angle")
print(model.angle)
print("angle to center")
print(model.angleCenter)
print("scale to center")
print(model.scaleCenter)

tri = Delaunay(model.pt)
_ = delaunay_plot_2d(tri)
plt.show()
