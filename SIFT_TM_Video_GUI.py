# SIFT template model matching GUI

# Math lib
import math
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
# Computer vision lib
import cv2
import imutils
from imutils.video import WebcamVideoStream
# graphical lib
import PySimpleGUI as sg

# main code for learning object on the fly
def main():
    # generate theme
    sg.theme('DarkAmber')
    # All the stuff inside your window.
    layout = [  [sg.Image(filename='', key='-frame-'), sg.Image(filename='', key='-model-')],
                [sg.Button('Learn Model'), sg.Button('Close')] ]
    # Create the Window
    window = sg.Window('SIFT model Learning GUI', layout, location=(800, 400), finalize=True)
    # init frame acquisition
    cam = cv2.VideoCapture(0)
    print("Camera init -> DONE")
    # init feature detector SIFT
    SIFT = cv2.SIFT_create()
    # init feature matcher KNN
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    ratio_thresh = 0.80
    # init state frame and model
    ret, scene_img_RGB = cam.read()
    model_img = np.zeros((480, 640, 3))
    # init model keypoint and descriptor
    kp_obj = None
    des_obj = None
    # Event Loop to process "events" and get the "values" of the inputs
    while True:
        # read state windows
        event, values = window.read(timeout=0, timeout_key='timeout')
        # get state
        ret, frame = cam.read()
        scene_img_RGB = frame
        # start / stop the application
        if event == 'Close' or event is None:
            # kill thread and close window
            cam.release()
            cv2.destroyAllWindows()
            # stop program
            break
        # Learn model
        if event == 'Learn Model' or event is None:
            # get object ROI
            roi = cv2.selectROI(scene_img_RGB)
            model_img = scene_img_RGB[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
            cv2.destroyAllWindows()
            # find feature in the object ROI
            kp_obj, des_obj = SIFT.detectAndCompute(cv2.cvtColor(model_img, cv2.COLOR_BGR2GRAY), None)
            des_obj /= (des_obj.sum(axis=1, keepdims=True) + 1e-7)
            des_obj = np.sqrt(des_obj)
            # draw detected feature in the ROI
            model_img = cv2.drawKeypoints(model_img, kp_obj, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # perform object detection in the current state
        if model_img is not None and kp_obj is not None and des_obj is not None:
            # convert frame in gray
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
            if scene.shape[0] > 10:
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
                    if scenePoint.shape[0] > 10:
                        # estimate homographical transformation
                        TF, mask =  cv2.findHomography(objPoint, scenePoint, cv2.RANSAC, 0.99)
                        if TF is not None and mask[mask==1].size > 15:
                            # transform obj corners according the homographical transformation in the scene
                            h,w,c = model_img.shape
                            pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
                            dst = cv2.perspectiveTransform(pts, TF)
                            for i in range(scenePoint.shape[0]):
                                scene_img_RGB = cv2.circle(scene_img_RGB, (scenePoint[i,0], scenePoint[i,1]), 5, [0,255,255], -1)
                                scene_img_RGB = cv2.polylines(scene_img_RGB, [np.int32(dst)], True, (0,255,0), 20, cv2.LINE_AA)
        # update image on the GUI
        imgbytes_frame = cv2.imencode('.png', cv2.resize(scene_img_RGB, (640, 480), cv2.INTER_LINEAR))[1].tobytes()
        imgbytes_model = cv2.imencode('.png', model_img)[1].tobytes()
        window['-frame-'].update(data=imgbytes_frame)
        window['-model-'].update(data=imgbytes_model)

    window.close()

# start
main()
