import os
import sys
import MeCab
import re
import jaconv
import unicodedata
import numpy as np
import cv2 as cv
from PIL import ImageFont, ImageDraw, Image
from vcam import vcam, meshGen
import pytesseract
from pytesseract import Output
from googletrans import Translator, constants

# PARAMETERS
HEIGHT = 768
WIDTH = 1024

POSITION = [-20, -370, -140]
ROTATION = [-0.3, -0.5, 2.1]
ZOOM = [1, 1]
DISTORTION = [0, 0, 0, 0, 0, 0]

# FUNCTION


def set_camera_configuration():
    """Call only once if not manually calibrating"""
    cam_config = vcam(H=HEIGHT, W=WIDTH)
    cam_config.set_tvec(*POSITION)
    cam_config.set_rvec(*ROTATION)
    cam_config.sx = ZOOM[0]
    cam_config.sy = ZOOM[1]

    plane = meshGen(HEIGHT, WIDTH)
    plane.Z = plane.X * 0 + 1
    pts3d = plane.getPlane()
    pts2d = cam_config.project(pts3d)
    map_x, map_y = cam_config.getMaps(pts2d)

    return map_x, map_y


def set_externals(input_cam, output_window_name):
    cam = cv.VideoCapture(input_cam)
    cam.set(cv.CAP_PROP_FRAME_WIDTH, WIDTH)
    cam.set(cv.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    ret, frame = cam.read()
    output = np.zeros([HEIGHT,WIDTH,3], dtype=np.uint8)
    output.fill(255)

    cv.namedWindow(output_window_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(output_window_name, WIDTH, HEIGHT)

    cv.namedWindow("check", cv.WINDOW_NORMAL)
    cv.resizeWindow("check", WIDTH, HEIGHT)

    return cam, output_window_name, output


def set_text_detector():
    mecab = MeCab.Tagger("-d /opt/homebrew/lib/mecab/dic/ipadic -Ochasen")
    mecab.parse('')

    translator = Translator()

    ocr_config = r'--oem 3 --psm 6 -l jpn'
    return mecab, translator, ocr_config


def manual_calib(k):
    if k == ord('x'):
        POSITION[0] -= 1
    elif k == ord('X'):
        POSITION[0] += 1

    elif k == ord('y'):
        POSITION[1] -= 1
    elif k == ord('Y'):
        POSITION[1] += 1

    elif k == ord('z'):
        POSITION[2] -= 1
    elif k == ord('Z'):
        POSITION[2] += 1

    elif k == ord('a'):
        ROTATION[0] -= 0.1
    elif k == ord('A'):
        ROTATION[0] += 0.1

    elif k == ord('b'):
        ROTATION[1] -= 0.1
    elif k == ord('B'):
        ROTATION[1] += 0.1

    elif k == ord('g'):
        ROTATION[2] -= 0.1
    elif k == ord('G'):
        ROTATION[2] += 0.1

    elif k == ord('k'):
        ZOOM[0] -= 0.01
    elif k == ord('K'):
        ZOOM[0] += 0.01

    elif k == ord('l'):
        ZOOM[1] -= 0.01
    elif k == ord('L'):
        ZOOM[1] += 0.01

    print("Position: " + str(POSITION))
    print("Rotation: " + str(ROTATION))
    print("Zoom: " + str(ZOOM))
    print("Distortion: " + str(DISTORTION))


def ocr(frame, detector, custom_config):
    frame = get_grayscale(frame)
    frame = thresholding(frame)
    frame = remove_noise(frame)
    frame = opening(frame)
    frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
    sentence = pytesseract.image_to_string(frame, config=custom_config, output_type=Output.DICT)["text"]
    print(sentence)
    sentence = sentence.strip().split("\n")
    furigana = []
    for s in sentence:
        furigana = get_furigana(furigana, s)

    results = pytesseract.image_to_data(frame, config=custom_config, output_type=Output.DICT)
    print(results)
    for posif, f in enumerate(furigana):
        for i in range(0, len(results["text"])):
            conf = int(results["conf"][i])
            if conf > 70:
                print("im detected", end=" ")
                print(results["text"][i])
                if any(is_kanji(_) for _ in results["text"][i]):
                    print("im kanji", end = " ")
                    print(results["text"][i])
                    print(furigana)
                    if results["text"][i] in f[0]:
                        print("i have furigana", end=" ")
                        print(results["text"][i])
                        word_x = results["left"][i]
                        word_y = results["top"][i]

                        text = f[1]
                        detector = write_japanese(detector, text, word_x, word_y)
                        frame = write_japanese(frame, text, word_x, word_y)
                        furigana[posif] = ["", ""]
                        break
    return frame, detector


def get_furigana(furigana, sentence):
    node = mecab.parseToNode(sentence)
    while node is not None:
        origin = node.surface
        if not origin:
            node = node.next
            continue

        if origin != "" and any(is_kanji(_) for _ in origin):
            if len(node.feature.split(",")) > 7:
                kana = node.feature.split(",")[7]
            else:
                kana = node.surface
            hiragana = jaconv.kata2hira(kana)
            furigana.append([origin, hiragana])
        node = node.next
    return furigana


def is_kanji(ch):
    return 'CJK UNIFIED IDEOGRAPH' in unicodedata.name(ch)


def write_japanese(image, text, word_x, word_y):
    font = ImageFont.truetype("Koruri-Regular.ttf", 14)

    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    draw.text((word_x + 20, word_y - 20), text, font=font, fill=(0, 0, 255, 0), stroke_width=1)
    image = np.array(img_pil)
    return image


# get grayscale image
def get_grayscale(image):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


# noise removal
def remove_noise(image):
    return cv.medianBlur(image, 3)


# thresholding
def thresholding(image):
    return cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]


# dilation
def dilate(image):
    kernel = np.ones((3, 3), np.uint8)
    return cv.dilate(image, kernel, iterations=1)


# erosion
def erode(image):
    kernel = np.ones((3, 3), np.uint8)
    return cv.erode(image, kernel, iterations=1)


# opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((3, 3), np.uint8)
    return cv.morphologyEx(image, cv.MORPH_OPEN, kernel)


cam, output_window, detector = set_externals(0, "out")
map_x, map_y = set_camera_configuration()
mecab, translator, custom_config = set_text_detector()

while True:
    ret, frame = cam.read()
    frame = cv.flip(frame, 1)

    # just call this if manually positioning camera
    #map_x, map_y = set_camera_configuration()
    detector.fill(255)
    frame = cv.remap(frame, map_x, map_y, interpolation=cv.INTER_LINEAR)
    frame, detector = ocr(frame, detector, custom_config)




    cv.imshow("check", frame)
    cv.imshow("out", detector)

    k = cv.waitKey(1000) & 0xFF
    if k == ord('q'):
        break

    #manual_calib(k)


def main():
    return


if __name__ == '__main__':
    main()