# ORBcog is object template matching method for fast object recognition.
# This code is based Li Yang Ku SURF object matching

# create base model
class targetModel():
    def __init__(self, nbPointMax):
        super(targetModel, self).__init__()
        # input
        self.targetImage = None # image to train on
        self.targetMask = None # target binary mask
        self.targetSize = None
        # image keypoint extractor
        self.orb = cv2.ORB_create(nfeatures=nbPointMax)
        self.kp = None
        self.des = None
        # parameters for the probabilistic model

    def createModel(self, img, mask):
        # store image, mask, size
        self.targetImage = img
        self.targetMask = mask
        self.targetSize = img.shape
        # compute ORB keypoint and descriptor
        kp, des = self.orb.detectAndCompute(self.targetImage, None)
        # convert kp and get orientation and position in the image

        # compute the midle of the cropped object or the image

        # for each ORB keypoint
            # store the angle of the line between the ORB point and the middle point

            # store the length of the line between the ORB point and the middle point devide by the scale of the ORB descriptor

            # store the major orientation of the ORB point descriptor

            # store scale of the ORB point descriptor
