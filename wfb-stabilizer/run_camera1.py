#!/usr/bin/python3
# Modified version to run with camera index 1
# Based on ejo_wfb_stabilizer.py by ejowerks
# Updated: January 17, 2026

import cv2
import numpy as np

#################### USER VARS ######################################
downSample = 1.0
zoomFactor = 0.85  # More zoom to hide edge bouncing
processVar = 0.01  # Lower = more aggressive smoothing
measVar = 4        # Higher = more aggressive smoothing
showFullScreen = 0  # Set to 0 for windowed mode
delay_time = 1

######################## ROI Settings ###############################
roiDiv = 2.5       # Smaller = bigger ROI for more tracking area (was 3.5)
showrectROI = 1
showTrackingPoints = 0
showUnstabilized = 1
maskFrame = 0

######################## Video Source ###############################
CAMERA_INDEX = 1  # Harrier camera

######################################################################

lk_params = dict(winSize=(15, 15), maxLevel=3, 
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
count = 0
a = 0
x = 0
y = 0
Q = np.array([[processVar] * 3])
R = np.array([[measVar] * 3])
K_collect = []
P_collect = []
prevFrame = None

print(f"Opening camera {CAMERA_INDEX}...")
video = cv2.VideoCapture(CAMERA_INDEX)

if not video.isOpened():
    print(f"ERROR: Could not open camera {CAMERA_INDEX}")
    exit(1)

# Set resolution
video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
video.set(cv2.CAP_PROP_FPS, 60)

print(f"Camera opened. Resolution: {int(video.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {video.get(cv2.CAP_PROP_FPS)} FPS")
print("Press 'Q' to quit")

while True:
    grab, frame = video.read()
    if grab is not True:
        print("Failed to grab frame")
        continue
        
    res_w_orig = frame.shape[1]
    res_h_orig = frame.shape[0]
    res_w = int(res_w_orig * downSample)
    res_h = int(res_h_orig * downSample)
    top_left = [int(res_h / roiDiv), int(res_w / roiDiv)]
    bottom_right = [int(res_h - (res_h / roiDiv)), int(res_w - (res_w / roiDiv))]
    frameSize = (res_w, res_h)
    Orig = frame
    
    if downSample != 1:
        frame = cv2.resize(frame, frameSize)
        
    currFrame = frame
    currGray = cv2.cvtColor(currFrame, cv2.COLOR_BGR2GRAY)
    currGray = currGray[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]

    if prevFrame is None:
        prevOrig = frame
        prevFrame = frame
        prevGray = currGray

    if (grab == True) & (prevFrame is not None):
        if showrectROI == 1:
            cv2.rectangle(prevOrig, (top_left[1], top_left[0]), (bottom_right[1], bottom_right[0]), 
                         color=(211, 211, 211), thickness=1)

        prevPts = cv2.goodFeaturesToTrack(prevGray, maxCorners=400, qualityLevel=0.01, 
                                          minDistance=30, blockSize=3)
        if prevPts is not None:
            currPts, status, err = cv2.calcOpticalFlowPyrLK(prevGray, currGray, prevPts, None, **lk_params)
            assert prevPts.shape == currPts.shape
            idx = np.where(status == 1)[0]
            prevPts = prevPts[idx] + np.array([int(res_w_orig / roiDiv), int(res_h_orig / roiDiv)])
            currPts = currPts[idx] + np.array([int(res_w_orig / roiDiv), int(res_h_orig / roiDiv)])
            
            if showTrackingPoints == 1:
                for pT in prevPts:
                    cv2.circle(prevOrig, (int(pT[0][0]), int(pT[0][1])), 5, (211, 211, 211))
                    
            if prevPts.size & currPts.size:
                m, inliers = cv2.estimateAffinePartial2D(prevPts, currPts)
            if m is None:
                m = lastRigidTransform
                
            dx = m[0, 2]
            dy = m[1, 2]
            da = np.arctan2(m[1, 0], m[0, 0])
        else:
            dx = 0
            dy = 0
            da = 0

        x += dx
        y += dy
        a += da
        Z = np.array([[x, y, a]], dtype="float")
        
        if count == 0:
            X_estimate = np.zeros((1, 3), dtype="float")
            P_estimate = np.ones((1, 3), dtype="float")
        else:
            X_predict = X_estimate
            P_predict = P_estimate + Q
            K = P_predict / (P_predict + R)
            X_estimate = X_predict + K * (Z - X_predict)
            P_estimate = (np.ones((1, 3), dtype="float") - K) * P_predict
            K_collect.append(K)
            P_collect.append(P_estimate)
            
        diff_x = X_estimate[0, 0] - x
        diff_y = X_estimate[0, 1] - y
        diff_a = X_estimate[0, 2] - a
        dx += diff_x
        dy += diff_y
        da += diff_a
        
        m = np.zeros((2, 3), dtype="float")
        m[0, 0] = np.cos(da)
        m[0, 1] = -np.sin(da)
        m[1, 0] = np.sin(da)
        m[1, 1] = np.cos(da)
        m[0, 2] = dx
        m[1, 2] = dy

        fS = cv2.warpAffine(prevOrig, m, (res_w_orig, res_h_orig))
        s = fS.shape
        T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, zoomFactor)
        f_stabilized = cv2.warpAffine(fS, T, (s[1], s[0]))
        
        window_name = f'WFB Stabilizer - Camera {CAMERA_INDEX}'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        if maskFrame == 1:
            mask = np.zeros(f_stabilized.shape[:2], dtype="uint8")
            cv2.rectangle(mask, (100, 200), (1180, 620), 255, -1)
            f_stabilized = cv2.bitwise_and(f_stabilized, f_stabilized, mask=mask)
            
        if showFullScreen == 1:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        cv2.imshow(window_name, f_stabilized)

        if showUnstabilized == 1:
            cv2.imshow("Unstabilized ROI", prevGray)
            
        if cv2.waitKey(delay_time) & 0xFF == ord('q'):
            break

        prevOrig = Orig
        prevGray = currGray
        prevFrame = currFrame
        lastRigidTransform = m
        count += 1

video.release()
cv2.destroyAllWindows()
print("Done")
