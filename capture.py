import cv2 as cv
import os


cam = cv.VideoCapture(0)
test = cv.imread("graycode_pattern/pattern_00.png")

while True:
    cv.imshow("test", test)
    ret, frame = cam.read()
    cv.imshow("OG", frame)
    k = cv.waitKey(1) & 0xFF
    # press 'q' to exit
    if k == ord('q'):
        break

for i in range(100):
    folder = "capture_" + str(i)
    if not os.path.exists(folder):
        os.makedirs(folder)
        break

for i in range(0, 34):
    file = "pattern_" + str(i).zfill(2)+".png"
    newfile = "graycode_" + str(i).zfill(2) + ".png"
    print(file)
    print(newfile)
    test = cv.imread("graycode_pattern/" + file)
    cv.imshow("test", test)
    cv.waitKey(1000)

    ret, frame = cam.read()
    cv.imshow("OG", frame)
    cv.imwrite(folder+"/"+newfile, frame)
    cv.waitKey(100)
