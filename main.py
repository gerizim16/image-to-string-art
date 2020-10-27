import cmath
import math
import cv2
import sys
import random
import numpy as np
import skimage.draw
from enum import Enum
from PIL import ImageEnhance, Image


class Shape(Enum):
    CIRCLE = 1
    SQUARE = 2


def getEdges(img):
    return cv2.Canny(img, 100, 200)
    return cv2.Laplacian(img, cv2.CV_8UC1)


def getCenter(img):
    return (img.shape[0] // 2, img.shape[1] // 2)


def getLines(img):
    radius = img.shape[1] / 2
    return cv2.HoughLinesP(img,
                           1,
                           np.pi / 180,
                           30,
                           minLineLength=radius,
                           maxLineGap=radius * 0.25)


def cleanEdges(img, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img = cv2.dilate(img, kernel, iterations=1)
    return img


def crop(img, shape):
    if shape == Shape.SQUARE:
        h, w = img.shape[:2]
        if h == w:
            return img

        padding = abs(h - w) // 2
        if h > w:
            img = img[padding:-padding, :]
        else:
            img = img[:, padding:-padding]
    elif shape == Shape.CIRCLE:
        img = crop(img, Shape.SQUARE)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.circle(mask,
                   getCenter(mask),
                   img.shape[0] // 2, (255, ),
                   thickness=-1)
        img = cv2.bitwise_and(img, img, mask=mask)
    return img


def getLineMetric(img, rr, cc, color):
    line = img[rr, cc].astype(np.int32)

    # If color is near black, invert colors
    if sum(color) < 255 / 2:
        line = 255 - line

    return np.sum(line)


def getDistance(a, b):
    x = a[0] - b[0]
    y = a[1] - b[1]
    return math.sqrt(x**2 + y**2)


def stringArt(img,
              nails_count=180,
              bgcolor=(255, 255, 255),
              color=(0, 0, 0),
              shape=Shape.CIRCLE):
    random.seed(16)
    # Initialize images
    img_basis = crop(img, shape)
    img_basis = cv2.resize(img_basis, (600, 600))
    img_comp = cv2.cvtColor(img_basis, cv2.COLOR_BGR2GRAY)

    img_basis = Image.fromarray(img_basis)
    img_basis_enhancer = ImageEnhance.Brightness(img_basis)
    img_basis = img_basis_enhancer.enhance(1.5)
    img_basis_enhancer = ImageEnhance.Contrast(img_basis)
    img_basis = img_basis_enhancer.enhance(1)

    img_basis = np.asarray(img_basis)
    img_art = np.full(img_basis.shape, bgcolor, dtype=np.uint8)
    img_art = crop(img_art, shape)


    # Initialize nail locations
    nails = set()
    angle_step = 2 * cmath.pi / nails_count
    radius = img_art.shape[0] / 2 - 3
    for i in range(nails_count):
        complex_rect = cmath.rect(
            radius, angle_step * i) + complex(*getCenter(img_art))
        nails.add((round(complex_rect.real), round(complex_rect.imag)))

    # debug draw nails
    for nail in nails:
        cv2.circle(img_art, nail, 3, (0, 255, 0), -1)

    # debug show img_basis
    cv2.imshow("img_basis", img_basis)

    # Produce lines
    img_comp = getEdges(img_comp)
    img_comp = cleanEdges(img_comp)
    temp = cv2.cvtColor(img_basis, cv2.COLOR_BGR2GRAY)
    temp = np.clip(temp, 50, 255)
    img_comp = cv2.subtract(temp, img_comp)
    cv2.imshow("img_comp", img_comp)

    lines = getLines(img_comp)
    lines = None
    if lines is not None:
        print(lines.shape)
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img_art, (x1, y1), (x2, y2), (0, 0, 255), 1)
    # Produce lines
    lines = set()
    current_nail = nails.pop()
    nails.add(current_nail)
    while True:
        candidate_metric = -math.inf
        candidate = None
        c_rr, c_cc = None, None
        for nail in nails:
            if getDistance(nail, current_nail) < radius / 3:
                continue
            if nail == current_nail or (current_nail, nail) in lines:
                continue
            rr, cc = skimage.draw.line(*nail, *current_nail)
            line_metric = getLineMetric(img_comp, rr, cc, color)
            if line_metric > candidate_metric:
                candidate_metric = line_metric
                candidate = nail
                c_rr, c_cc = rr, cc
        if candidate is None:
            break

        # Add line to lines made
        lines.add((current_nail, candidate))
        lines.add((candidate, current_nail))

        # Shade img_art
        img_art[c_rr, c_cc] = cv2.subtract(img_art[c_rr, c_cc], 20)

        # Debug
        cv2.imshow("img_art", img_art)
        cv2.waitKey(1)

        # Move to new nail
        current_nail = candidate

    # debug
    cv2.imshow("img_basis", img_basis)
    cv2.imshow("img_art", img_art)

    return img_art


if __name__ == "__main__":
    filename = None
    if len(sys.argv) == 2:
        filename = sys.argv[1]
    else:
        filename = input('Enter filename: ')

    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    img_art = stringArt(img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()