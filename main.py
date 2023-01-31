import cv2
import pytesseract
from pytesseract import Output
cam = cv2.VideoCapture(0)
cv2.namedWindow("OG")

while True:
    ret, frame = cam.read()
    flip = frame #cv2.flip(frame, 1)  #flipping camera to fix alignment

    print("Printing...")
    custom_config = r'--oem 3 --psm 11 -l jpn'
    results = pytesseract.image_to_data(flip, config=custom_config, output_type=Output.DICT)
    print(results)
    for i in range(0, len(results["text"])):
        x = results["left"][i]
        y = results["top"][i]

        w = results["width"][i]
        h = results["height"][i]

        text = results["text"][i]
        conf = int(results["conf"][i])

        if conf > 70:
            text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
            cv2.rectangle(flip, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(flip, text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2)
    cv2.imshow("OG", flip)
    cv2.waitKey(1)

cam.release()
c