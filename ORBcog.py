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
        self.maxSize = None
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
        self.maxSize = np.max(self.targetImage.shape)
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
                    self.angle.append(-kp[i].angle*np.pi/180)
                    self.scale.append(kp[i].size)
                    self.response.append(kp[i].response)
        else:
            # get point, angle and size from keypoints
            for i in range(0,len(kp)):
                self.pt.append(kp[i].pt)
                self.angle.append(-kp[i].angle*np.pi/180)
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
            self.angleCenter.append(np.arctan2(self.pt[i][1], self.pt[i][0])-self.angle[i])
            # store the length of the line between the ORB point and the middle point devide by the scale of the ORB descriptor
            self.scaleCenter.append(np.sqrt((centerW-self.pt[i][0])**2+(centerH-self.pt[i][1])**2)/self.scale[i])

# match model with current image using KNN matching and Best bin first method 
def matchModel(targetModel, currentImg, nbPointMax, LoweCoeff):
    # create ORB descriptor/detector and BF feature matcher
    orb = cv2.ORB_create(nfeatures=nbPointMax)
    matcher = cv2.DescriptorMatcher_create("BruteForce-L1")
    # compute ORB keypoint and descriptor
    kp, des = orb.detectAndCompute(currentImg, None)
    # draw keypoint
    kp_img = cv2.drawKeypoints(currentImg, kp, None,  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(kp_img)
    plt.show()
    # store current point, angle and scale
    curPt = []
    curAngle = []
    curScale  = []
    curResponse = []
    for i in range(0,len(kp)):
        self.pt.append(kp[i].pt)
        self.angle.append(-kp[i].angle*np.pi/180)
        self.scale.append(kp[i].size)
        self.response.append(kp[i].response)
    # using BF compute match
    matches = matcher.knnMatch(queryDescriptors=des, trainDescriptors=targetModel.des, k=2)
    # filter match using Lowe's method
    goodMatch = []
    goodMatchPlot = [] # only for plotting
    for m,n in matches:
        if m.distance < LoweCoeff*n.distance:
            goodMatch.append(m)
            goodMatchPlot.append([m])
    # set the location bin size to 0.25 the image model max size
    maxLoc = targetModel.maxSize*0.25


img = cv2.imread('target.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
mask = cv2.imread('mask.png')
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

model = targetModel(nbPointMax=10)
model.createModel(img=img, mask=mask, imgCenter=True)



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

matchModel(targetModel=None, currentImg=img, nbPointMax=10, LoweCoeff=0.70)
