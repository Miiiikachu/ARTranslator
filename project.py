import cv2 as cv
import numpy as np
import os

cam = cv.VideoCapture(0)

fs = cv.FileStorage("calibration_result.xml", cv.FILE_STORAGE_READ)
cam_shape = fs.getNode("img_shape").mat()
ret = fs.getNode("rms")
cam_int = fs.getNode("cam_int").mat()
cam_dist = fs.getNode("cam_dist").mat()
proj_int = fs.getNode("proj_int").mat()
proj_dist = fs.getNode("proj_dist").mat()
cam_proj_rmat = fs.getNode("rotation").mat()
cam_proj_tvec = fs.getNode("translation").mat()

R1, R2, P1, P2, Q, ROI1, ROI2 = cv.stereoRectify(cam_int, cam_dist, proj_int, proj_dist, (1280, 960), cam_proj_rmat, cam_proj_tvec, flags=cv.CALIB_ZERO_DISPARITY, alpha=0)
w = 1280
h = 960
res = (int(w), int(h))
res2 = (int(w), int(h))
print(Q)
#new_camera_matrix, valid_pix_roi = cv.getOptimalNewCameraMatrix(cam_int, cam_dist, res, 1)
a, b = cv.initUndistortRectifyMap(cam_int, cam_dist, R1, P1, res2, cv.CV_32FC1)
#c, d = cv.initUndistortRectifyMap(proj_int, proj_dist, R2, P2, (1280, 960), cv.CV_32FC1)
print(cam_proj_rmat[0][1])
T = np.array([[cam_proj_rmat[0][0], cam_proj_rmat[0][1], cam_proj_tvec[0][0]],
              [cam_proj_rmat[1][0], cam_proj_rmat[1][1], cam_proj_tvec[1][0]],
              [0, 0, 1]])

from vcam import vcam,meshGen

c1 = vcam(H=h,W=w)
c1.set_tvec(cam_proj_tvec[0][0], cam_proj_tvec[1][0], cam_proj_tvec[2][0])
c1.set_rvec(0, 0, -5)
c1.sx = 0.3
c1.sy = 0.3
plane = meshGen(h, w)
plane.Z = plane.X*0 + 1
#print(cam_dist)
#c1.KpCoeff[0] = cam_dist[0][0]
#c1.KpCoeff[1] = cam_dist[0][1]
#c1.KpCoeff[2] = cam_dist[0][2]
#c1.KpCoeff[3] = cam_dist[0][3]
#c1.KpCoeff[4] = cam_dist[0][4]

a = 0
b = -1.5
g = -8
x = 50
y = -300
z = cam_proj_tvec[2][0]
l = 0.34
while True:
    c1 = vcam(H=h, W=w)
    c1.set_tvec(x, y, z)
    c1.set_rvec(a, b, g)
    c1.sx = l
    c1.sy = l
    plane = meshGen(h, w)
    plane.Z = plane.X * 0 + 1
    pts3d = plane.getPlane()
    pts2d = c1.project(pts3d)

    map_x, map_y = c1.getMaps(pts2d)

    ret, frame = cam.read()
    h, w = frame.shape[:2]
    #test = cv.remap(frame, a, b, cv.INTER_LINEAR)
    test = cv.remap(frame, map_x, map_y, interpolation=cv.INTER_LINEAR)

    #test = cv.warpPerspective(test, T, res)

    #test = cv.remap(test, c, d, cv.INTER_LINEAR)
    #test = cv.undistort(frame, cam_int, cam_dist, None, cam_int)
    cv.imshow("original", frame)
    cv.imshow("test", test)
    k = cv.waitKey(1) & 0xFF
    # press 'q' to exit
    if k == ord('q'):
        break
    if k == ord('a'):
        a += 1
        print(a)
    if k == ord('b'):
        b += 0.1
        print(b)
    if k == ord('p'):
        b -= 0.1
        print(b)
    if k == ord('g'):
        g += 1
        print(g)
    if k == ord('x'):
        x += 1
        print(x)
    if k == ord('y'):
        y -= 1
        print(y)
    if k == ord('z'):
        z += 1
        print(z)
    if k == ord('l'):
        l += 0.01
        print(l)

