# SIFT multi-instance object matching

# Math lib
import math
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
# Computer vision lib
import cv2
import imutils
from imutils.video import WebcamVideoStream

# get template object image
template_img = cv2.imread('template_.png', cv2.IMREAD_GRAYSCALE)
# init feature detector SIFT
SIFT = cv2.SIFT_create()
# init feature matcher KNN
matcher = cv2.DescriptorMatcher_create("BruteForce")
ratio_thresh = 0.80

# find feature in the object template
kp_obj, des_obj = SIFT.detectAndCompute(template_img, None)
des_obj /= (des_obj.sum(axis=1, keepdims=True) + 1e-7)
des_obj = np.sqrt(des_obj)

# start frame acquisition
print("Camera init -> DONE")
cam = WebcamVideoStream(src=0).start()

while True:
    # Capture frame
    frame = cam.read()
    scene_img_RGB = frame
    # image for drawing
    img_matches = scene_img_RGB.copy()
    # image for processing
    scene_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # get feature in the scene
    kp_scene, des_scene = SIFT.detectAndCompute(scene_img, None)
    des_scene /= (des_scene.sum(axis=1, keepdims=True) + 1e-7)
    des_scene = np.sqrt(des_scene)
    # match feature with the template using KNN matching (norm L2)
    knn_matches = matcher.knnMatch(des_obj, des_scene, 2)
    # filter matches (lowe ratio)
    good_matches = []
    for m,n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    # create empty keypoint position vector for all good matches
    obj = np.empty((len(good_matches),2), dtype=np.float32)
    scene = np.empty((len(good_matches),2), dtype=np.float32)
    # update keypoints position
    for i in range(len(good_matches)):
        obj[i,0] = kp_obj[good_matches[i].queryIdx].pt[0]
        obj[i,1] = kp_obj[good_matches[i].queryIdx].pt[1]
        scene[i,0] = kp_scene[good_matches[i].trainIdx].pt[0]
        scene[i,1] = kp_scene[good_matches[i].trainIdx].pt[1]

    if scene.shape[0] > 30:
        # compute bandwith for the clustering
        bandwidth = estimate_bandwidth(scene, quantile=0.2)
        # compute clusters for keypoint
        meanShift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        meanShift.fit(scene)
        labels = meanShift.labels_
        clusterCenters = meanShift.cluster_centers_

        # compute pose using cluster label
        for c in range(len(clusterCenters)):
            currentCluster = labels == c
            objPoint = obj[currentCluster, :]
            scenePoint = scene[currentCluster, :]
            # if cluster point number superior to a threshold e=10
            if scenePoint.shape[0] > 30:
                # estimate homographical transformation
                TF, mask =  cv2.findHomography(objPoint, scenePoint, cv2.RANSAC, 0.99)
                if TF is not None:
                    # transform obj corners according the homographical transformation in the scene
                    h,w = template_img.shape
                    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                    dst = cv2.perspectiveTransform(pts, TF)
                    for i in range(scenePoint.shape[0]):
                        img_matches = cv2.circle(img_matches, (scenePoint[i,0], scenePoint[i,1]), 5, [0,255,255], -1)
                    img_matches = cv2.polylines(img_matches, [np.int32(dst)], True, (0,255,0), 20, cv2.LINE_AA)

    # draw
    cv2.imshow('Good Matches & Object detection', img_matches)
    if cv2.waitKey(33) == 27:
        # kill thread and close window
        cam.stop()
        cv2.destroyAllWindows()
        break
