# ORBcog is object template matching method for fast object recognition.
# This code is based Li Yang Ku SURF object matching

import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import pyplot

# create base model
class targetModel():
    def __init__(self, nbPointMax):
        super(targetModel, self).__init__()
        # input
        self.targetImage = None # image to train on
        self.targetMask = None # target binary mask
        self.target = None # masked image
        self.targetSize = None # input size
        # image keypoint extractor
        self.orb = cv2.ORB_create(nfeatures=nbPointMax)
        self.kp = None
        self.des = None
        # parameters for the probabilistic model
        self.pt = [] # keypoint
        self.angle = [] # angle of the keypoint
        self.ptSize = [] # size response of the keypoint

    def createModel(self, img, mask):
        # store target image and mask
        self.targetImage = img
        self.targetMask = mask
        self.targetSize = img.shape
        # apply the mask if there is one
        if self.targetMask != None:
            self.target = cv2.bitwise_and(img, img, self.targetMask)
        else:
            self.target = self.targetImage
        # compute ORB keypoint and descriptor
        kp, des = self.orb.detectAndCompute(self.target, None)
        # get point, angle and size from keypoints
        for i in range(0,len(kp)):
            self.pt.append(kp[i].pt)
            self.angle.append(-kp[i].angle*np.pi/180)
            self.ptSize.append(kp[i].size)
        # compute the midle of the cropped image or the centroid of the masked objects
        if self.targetMask != None:
            # compute binary centroid
            centerH, centerW = np.argwhere(self.targetMask==0).sum(0)/(self.targetMask == 0).sum()
            centerH = int(centerX)
            centerW = int(centerY)
        else:
            # compute middle of the image
            centerH = int(self.targetSize[1]/2)
            centerW = int(self.targetSize[0]/2)

        for i in range(0, len(self.pt)):
            # store the angle of the line between the ORB point and the middle point

            # store the length of the line between the ORB point and the middle point devide by the scale of the ORB descriptor

            # store the major orientation of the ORB point descriptor

            # store scale of the ORB point descriptor
