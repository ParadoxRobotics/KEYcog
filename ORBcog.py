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
        # image keypoint extractor
        self.orb = cv2.ORB_create(nfeatures=nbPointMax)
        self.kp = None
        self.des = None
        # parameters for the probabilistic model
        self.pt = [] # keypoint [x,y] -> [W,H]
        self.angle = [] # angle of the keypoint
        self.scale = [] # size response of the keypoint (scale)
        self.angleCenter = [] # angle between the ORB keypoint and the center of the image
        self.scaleCenter = [] # length of the line between the ORB point and the middle point devide by the scale of the ORB descriptor

    def createModel(self, img, mask):
        # store target image and mask
        self.targetImage = img
        self.targetMask = mask
        # compute ORB keypoint and descriptor
        kp, des = self.orb.detectAndCompute(self.targetImage, None)
        kp_img = cv2.drawKeypoints(img, kp, None,  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.imshow(self.targetMask)
        plt.show()
        plt.imshow(kp_img)
        plt.show()
        # get point, angle and size from keypoints
        for i in range(0,len(kp)):
            self.pt.append(kp[i].pt)
            self.angle.append(kp[i].angle)
            self.scale.append(kp[i].size)
        # compute the midle of the cropped image
        centerH = int(self.targetImage.shape[1]/2)
        centerW = int(self.targetImage.shape[0]/2)
        # for every ORB point
        for i in range(0, len(self.pt)):
            # store the angle of the line between the ORB point and the middle point
            self.angleCenter.append(np.arctan2(self.pt[i][1], self.pt[i][0])-(-self.angle[i]*np.pi/180))
            # store the length of the line between the ORB point and the middle point devide by the scale of the ORB descriptor
            self.scaleCenter.append(np.sqrt((centerW-self.pt[i][0])**2+(centerH-self.pt[i][1])**2)/self.scale[i])


