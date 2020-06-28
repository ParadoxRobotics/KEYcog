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

# match model with current image using KNN matching and generalized hough transform
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
    for i in range(0,len(kp)):
        curPt.append(kp[i].pt)
        curAngle.append(-kp[i].angle*np.pi/180)
        curScale.append(kp[i].size)
    # using BF compute match
    matches = matcher.knnMatch(queryDescriptors=targetModel.descriptor, trainDescriptors=des, k=2)
    # filter match using Lowe's method
    goodMatch = []
    goodMatchPlot = [] # only for plotting
    for m,n in matches:
        if m.distance < LoweCoeff*n.distance:
            goodMatch.append(m)
            goodMatchPlot.append([m])

    print(len(goodMatch))

    # get ref and cur point, angle and scale
    refPt = np.float32([model.pt[m.queryIdx] for m in goodMatch])
    curPt = np.float32([curPt[m.trainIdx] for m in goodMatch])
    refAngle = np.float32([model.angle[m.queryIdx] for m in goodMatch])
    curAngle = np.float32([curAngle[m.trainIdx] for m in goodMatch])
    refScale = np.float32([model.scale[m.queryIdx] for m in goodMatch])
    curScale = np.float32([curScale[m.trainIdx] for m in goodMatch])
    # get model parameter for the good match
    refAngleCenter = np.float32([model.angleCenter[m.queryIdx] for m in goodMatch])
    refScaleCenter = np.float32([model.scaleCenter[m.queryIdx] for m in goodMatch])

    # location of the search space to 1/4 of the image model max size
    LocationBinSize = targetModel.maxSize*0.25
    # set orientation bin to 12
    nbAngleBin = 12
    # set scale bin to 10
    nbScaleBin = 10
    # compute location in W axis
    nbLocationW = int(np.round(currentImg.shape[1]*2/LocationBinSize))
    # compute location in H axis
    nbLocationH = int(np.round(currentImg.shape[0]*2/LocationBinSize))
    # orientation bin (rad)
    AngleBinSize = 2*np.pi/nbAngleBin

    # init consensus grid (4D)
    vote = np.zeros((nbAngleBin, nbScaleBin, nbLocationW, nbLocationH))
    # init match map (ORB index) -> [nbAngleBin, nbScaleBin, nbLocationW, nbLocationH]
    voteIdx = [[],[],[],[]]

    # for every match pair
    for i in range(0,len(goodMatch)):
        # calculate the proposed target point, angle and scale
        angleVal = curAngle[i] - refAngle[i] + 2*np.pi

        scaleVal = curScale[i] / refScale[i]
        ptWVal = curPt[i][0] + np.cos(curAngle[i]) + refAngleCenter[i]*curScale[i]*refScaleCenter[i]
        ptHVal = curPt[i][1] + np.sin(curAngle[i]) + refAngleCenter[i]*curScale[i]*refScaleCenter[i]
        # find the bin
        angleBin = np.mod(np.round(angleVal/AngleBinSize), nbAngleBin)+1
        scaleBin = np.round(np.log2(scaleVal)+nbScaleBin/2)
        ptWBin = np.ceil(ptWVal/LocationBinSize)
        ptHBin = np.ceil(ptHVal/LocationBinSize)

        # compute hough voting
        for a in range(0,2): # angle
            for s in range(0,2): # scale
                for w in range(0,2): # W
                    for h in range(0,21): # H
                        # vote +1
                        vote[int(np.mod(angleBin+a,nbAngleBin)),
                        int(np.mod(scaleBin+s,nbScaleBin)),
                        int(np.mod(ptWBin+w,nbLocationW)),
                        int(np.mod(ptHBin+h,nbLocationH))] = vote[int(np.mod(angleBin+a,nbAngleBin)),
                                                             int(np.mod(scaleBin+s,nbScaleBin)),
                                                             int(np.mod(ptWBin+w,nbLocationW)),
                                                             int(np.mod(ptHBin+h,nbLocationH))]+1
                        # store the vote vote
                        voteIdx[0].append(((int(np.mod(angleBin+a,nbAngleBin))), i))
                        voteIdx[1].append(((int(np.mod(scaleBin+s,nbScaleBin))), i))
                        voteIdx[2].append(((int(np.mod(ptWBin+w,nbLocationW))), i))
                        voteIdx[3].append(((int(np.mod(ptHBin+h,nbLocationH))), i))

"""
    # sort the maximum vote in the hough vote
    maxVoteId = np.argsort(vote.flatten())
    sortMaxVote = np.sort(vote.flatten())
    # refine result
    nbCluster = len(np.argwhere(sortMaxVote > 2)) # cluster with more than 2 vote
    for i in range(0, nbCluster):
        houghMatch = voteIdx[maxVoteId[i]]
        A = []
        B = []
"""

target = cv2.imread('target.jpg')
target = cv2.resize(target,(640,480))
target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
test = cv2.imread('test.jpg')
test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)

model = targetModel(nbPointMax=100)
model.createModel(img=target, mask=None, imgCenter=True)

matchModel(targetModel=model, currentImg=test, nbPointMax=500, LoweCoeff=0.70)
