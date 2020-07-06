# SURF/KNN template matching

# Math lib
import math
import numpy as np
# Computer vision lib
import cv2
import imutils
from imutils.video import WebcamVideoStream

# template size
w_template = 320
h_template = 320
# scene size
w_scene = 640
h_scene = 480

# get template object image
template_img = cv2.imread('template.jpg', cv2.IMREAD_GRAYSCALE)
template_img = cv2.resize(template_img, (w_template, h_template))
# init feature detector ORB
orb_detector = cv2.ORB_create(nfeatures=3000, nlevels=8, scoreType=cv2.ORB_FAST_SCORE)
# init feature matcher KNN
matcher = cv2.DescriptorMatcher_create("BruteForce-L1")
ratio_thresh = 0.80

# find feature in the object template
kp_obj, des_obj = orb_detector.detectAndCompute(template_img, None)

# object corner based on the template image
obj_corners = np.empty((4,1,2), dtype=np.float32)
obj_corners[0,0,0] = 0
obj_corners[0,0,1] = 0
obj_corners[1,0,0] = h_template
obj_corners[1,0,1] = 0
obj_corners[2,0,0] = h_template
obj_corners[2,0,1] = w_template
obj_corners[3,0,0] = 0
obj_corners[3,0,1] = w_template

# start frame acquisition
print("Camera init -> DONE")
cam = WebcamVideoStream(src=0).start()

while True:
    # Capture frame
    frame = cam.read()
    scene_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scene_img = cv2.resize(scene_img,(w_scene, h_scene))
    # get feature in the scene
    kp_scene, des_scene = orb_detector.detectAndCompute(scene_img, None)
    # match feature with the template using KNN matching (norm L2)
    knn_matches = matcher.knnMatch(des_obj, des_scene, 2)
    # filter matches (lowe ratio)
    good_matches = []
    for m,n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    # draw matches
    img_matches = np.empty((w_scene, h_scene),dtype=np.uint8)
    img_matches = scene_img

    # create empty keypoint position vector for all good matches
    obj = np.empty((len(good_matches),2), dtype=np.float32)
    scene = np.empty((len(good_matches),2), dtype=np.float32)
    # update keypoints position
    for i in range(len(good_matches)):
        obj[i,0] = kp_obj[good_matches[i].queryIdx].pt[0]
        obj[i,1] = kp_obj[good_matches[i].queryIdx].pt[1]
        scene[i,0] = kp_scene[good_matches[i].trainIdx].pt[0]
        scene[i,1] = kp_scene[good_matches[i].trainIdx].pt[1]

    # estimate homographical transformation
    TF, _ =  cv2.findHomography(obj, scene, cv2.RANSAC)

    # transform obj corners according the homographical transformation in the scene
    scene_corners = cv2.perspectiveTransform(obj_corners, TF)

    # Draw lines between the corners
    cv2.line(img_matches, (int(scene_corners[0,0,0]), int(scene_corners[0,0,1])),\
        (int(scene_corners[1,0,0]), int(scene_corners[1,0,1])), (255,0,0), 4)
    cv2.line(img_matches, (int(scene_corners[1,0,0]), int(scene_corners[1,0,1])),\
        (int(scene_corners[2,0,0]), int(scene_corners[2,0,1])), (255,0,0), 4)
    cv2.line(img_matches, (int(scene_corners[2,0,0]), int(scene_corners[2,0,1])),\
        (int(scene_corners[3,0,0]), int(scene_corners[3,0,1])), (255,0,0), 4)
    cv2.line(img_matches, (int(scene_corners[3,0,0]), int(scene_corners[3,0,1])),\
        (int(scene_corners[0,0,0]), int(scene_corners[0,0,1])), (255,0,0), 4)

    # draw
    cv2.imshow('Good Matches & Object detection', img_matches)
    if cv2.waitKey(1) == 27:
        break
