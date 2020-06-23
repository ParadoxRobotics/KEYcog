# ORBcog is object template matching method for fast object recognition.
# This code is based Li Yang Ku SURF object matching

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

    def createModel(self, img, mask):
        # store image and mask
        self.targetImage = img
        self.targetMask = mask
        # compute ORB keypoint and descriptor
        kp, des = self.orb.detectAndCompute(self.targetImage, None)
