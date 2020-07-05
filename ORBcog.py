# ORBcog is object template matching method for fast object recognition.
# This code is based Li Yang Ku SURF object matching

import numpy as np
from scipy.linalg import lstsq, norm, inv
import cv2
from matplotlib import pyplot as plt
from matplotlib import pyplot

# flatten a nD nested list
def flatten(l):
    try:
        return flatten(l[0]) + (flatten(l[1:]) if len(l) > 1 else []) if type(l) is list else [l]
    except IndexError:
        return []

# create base model
class targetModel():
    def __init__(self, nbPointMax):
        super(targetModel, self).__init__()
        # input
        self.targetImage = None # image to train on
        self.targetMask = None # target binary mask
        self.maxSize = None
        # image keypoint extractor
        self.orb = cv2.ORB_create(nfeatures=nbPointMax, scaleFactor=2, nlevels=8, WTA_K=2)
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
                    self.angle.append(-kp[i].angle*(np.pi/180))
                    self.scale.append(kp[i].size)
        else:
            # get point, angle and size from keypoints
            for i in range(0,len(kp)):
                self.pt.append(kp[i].pt)
                self.angle.append(-kp[i].angle*(np.pi/180))
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
    orb = cv2.ORB_create(nfeatures=nbPointMax, scaleFactor=2, nlevels=8, WTA_K=2)
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
        curAngle.append(-kp[i].angle*(np.pi/180))
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

    print("number of match =" , len(goodMatchPlot))

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

    print("reference :")
    print(refPt)
    print("current :")
    print(curPt)

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
    voteIdx = [[[[[] for w in range(nbLocationH)] for v in range(nbLocationW)] for j in range(nbScaleBin)] for i in range(nbAngleBin)]
    # for every match pair
    for i in range(0,len(goodMatch)):
        # calculate the proposed target point, angle and scale
        angleVal = curAngle[i] - refAngle[i] + 2*np.pi
        scaleVal = curScale[i] / refScale[i]
        ptWVal = curPt[i][0] + np.cos(curAngle[i] + refAngleCenter[i])*curScale[i]*refScaleCenter[i]
        ptHVal = curPt[i][1] + np.sin(curAngle[i] + refAngleCenter[i])*curScale[i]*refScaleCenter[i]
        # find the bin
        angleBin = np.mod(np.round(angleVal/AngleBinSize), nbAngleBin)+1
        scaleBin = np.round(np.log2(scaleVal)+nbScaleBin/2)
        ptWBin = np.ceil(ptWVal/LocationBinSize)
        ptHBin = np.ceil(ptHVal/LocationBinSize)

        # compute hough voting
        for a in range(0,2): # angle
            for s in range(0,2): # scale
                for w in range(0,2): # W
                    for h in range(0,2): # H
                        # vote +1
                        vote[int(np.mod(angleBin+a,nbAngleBin)),
                        int(np.mod(scaleBin+s,nbScaleBin)),
                        int(np.mod(ptWBin+w,nbLocationW)),
                        int(np.mod(ptHBin+h,nbLocationH))] = vote[int(np.mod(angleBin+a,nbAngleBin)),
                                                             int(np.mod(scaleBin+s,nbScaleBin)),
                                                             int(np.mod(ptWBin+w,nbLocationW)),
                                                             int(np.mod(ptHBin+h,nbLocationH))]+1
                        # store the vote index
                        voteIdx[int(np.mod(angleBin+a,nbAngleBin))][int(np.mod(scaleBin+s,nbScaleBin))][int(np.mod(ptWBin+w,nbLocationW))][int(np.mod(ptHBin+h,nbLocationH))].append(i)

    # init obj counter
    objFinder = 0
    # candidate object part
    candidateRotation = []
    candidateTranslation = []
    candidateMatch = []
    # sort the maximum vote in the hough vote (axis=0)
    maxVoteId = np.unravel_index(np.argsort(vote, axis=None), vote.shape)
    maxVoteId = maxVoteId[0][::-1]
    sortMaxVote = np.sort(vote.flatten())
    sortMaxVote = sortMaxVote[::-1]

    # refine result
    nbCluster = len(np.argwhere(sortMaxVote > 2)) # cluster with more than 2 votes
    # for every cluster compute the houghMatch
    for i in range(0, nbCluster):
        # good index in the hough space projected in the match space
        houghMatch = flatten(voteIdx[maxVoteId[i]])
        houghMatch = list(dict.fromkeys(houghMatch)) # filter out the same index
        # compute matrix A and B for solving the linear pose estimation (Ax = B)
        for j in range(0, int(sortMaxVote[i])):
            testPt = curPt[houghMatch]
            modelPt = refPt[houghMatch]
            # create A and B matrix given the current point
            A = np.array([[modelPt[0,0],modelPt[0,1],0,0,1,0],[0,0,modelPt[0,0],modelPt[0,1],0,1]])
            B = np.array([[testPt[0,0]],[testPt[0,1]]])
            for mt in range(1, modelPt.shape[0]):
                CA = np.array([[modelPt[mt,0],modelPt[mt,1],0,0,1,0],[0,0,modelPt[mt,0],modelPt[mt,1],0,1]])
                CB = np.array([[testPt[mt,0]],[testPt[mt,1]]])
                A = np.concatenate((A, CA))
                B = np.concatenate((B, CB))
            # compute translation and rotation
        AT = inv(np.dot(A.T, A))
        BT = np.dot(A.T,B)
        UV = np.dot(AT,BT)
        R = np.reshape(UV[0:4,0],(2,2))
        T = np.reshape(UV[4:6,0],(2,1))

        # refine match
        filterMatch = []
        for j in range(0, int(sortMaxVote[i])):
            # position
            testPt = curPt[houghMatch]
            modelPt = refPt[houghMatch]
            # angle
            testAngle = curAngle[houghMatch]
            modelAngle = refAngle[houghMatch]
            # scale
            testScale = curScale[houghMatch]
            modelScale = refScale[houghMatch]
            # check position
            for k in range(0, modelPt.shape[0]):
                # check position
                if (norm(np.reshape(testPt[k],(2,1)) - (np.dot(R, np.reshape(modelPt[k],(2,1)))+T)) > 0.2*model.maxSize):
                    continue
                # check angle
                modelAngleVec = np.dot(R, np.array([[np.cos(modelAngle[k])],[np.sin(modelAngle[k])]]))
                if np.mod(np.abs(testAngle[k] - np.arctan2(modelAngleVec[1,0], modelAngleVec[0,0])), 2*np.pi) > np.pi/12:
                    continue
                # check scale
                modelScaleRatio = np.sqrt(norm(R*np.array([[1],[0]]))*norm(R*np.array([[0],[1]])))
                if (testScale[k]/modelScale[k])/modelScaleRatio > np.sqrt(2) or (testScale[k]/modelScale[k])/modelScaleRatio < -np.sqrt(2):
                    continue
                # append if good match
                filterMatch.append(j)

        # filter out the identical index match
        filterMatch = list(dict.fromkeys(filterMatch))
        # number of match < 3 -> dont compute
        if len(filterMatch) < 3:
            print("not enough point to compute -_- ")
            continue
        # Recompute pose
        testPt = curPt[houghMatch[filterMatch[0]]]
        modelPt = refPt[houghMatch[filterMatch[0]]]
        A = np.array([[modelPt[0],modelPt[1],0,0,1,0],[0,0,modelPt[0],modelPt[1],0,1]])
        B = np.array([[testPt[0]],[testPt[1]]])
        for v in range(1, len(filterMatch)):
            testPt = curPt[houghMatch[filterMatch[v]]]
            modelPt = refPt[houghMatch[filterMatch[v]]]
            CA = np.array([[modelPt[0],modelPt[1],0,0,1,0],[0,0,modelPt[0],modelPt[1],0,1]])
            CB = np.array([[testPt[0]],[testPt[1]]])
            A = np.concatenate((A, CA))
            B = np.concatenate((B, CB))
        # compute translation and rotation
        AT = inv(np.dot(A.T, A))
        BT = np.dot(A.T,B)
        UV = np.dot(AT,BT)
        R = np.reshape(UV[0:4,0],(2,2))
        T = np.reshape(UV[4:6,0],(2,1))
        # update object part found
        objFinder += 1
        # update candidate rotation, translation and match
        candidateRotation.append(R)
        candidateTranslation.append(T)
        candidateMatch.append([houghMatch[w] for w in filterMatch])

    if objFinder == 0:
        print("no object detected /!\ ")
    else:
        print("object part found !")

# TEST
target = cv2.imread('target.jpg')
target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
test = cv2.imread('test.jpg')
test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
# create model
model = targetModel(nbPointMax=1000)
model.createModel(img=target, mask=None, imgCenter=True)
# match the model with the current image state
matchModel(targetModel=model, currentImg=test, nbPointMax=1000, LoweCoeff=0.80)
