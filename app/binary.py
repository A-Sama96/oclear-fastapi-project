#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 40)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector


def preprocess(img):
    thresh = cv2.threshold(img, 100, 500, 1)[1]
    thresh = np.pad(thresh, 100, pad_with, padder=0)
    ret, thresh = cv2.threshold(
        thresh, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cv2.imwrite('c.png', thresh)
    img = cv2.imread('c.png')
    return thresh


def fix_dimension(img):
    img = cv2.resize(img, (100, 50))
    new_img = np.zeros((50, 100, 3))
    for i in range(3):
        new_img[:, :, i] = img
    return new_img


def bilateral_norm(img):
    img = cv2.bilateralFilter(img, 9, 15, 30)
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)


def histogram_norm(img):
    img = bilateral_norm(img)
    # Filtrage en image dont les pixels de premier plan sont blancs
    add_img = 255 - cv2.threshold(img, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    img = 255 - img  # filtrage en noir
    img = (img - np.min(img)) / (np.max(img) -
                                 np.min(img)) * 255  # normalisation
    # img.ravel pour transformer l'image en un vecteur
    hist, bins = np.histogram(img.ravel(), 256, [0, 256])
    #
    img = img.astype(np.uint8)

    ret, thresh4 = cv2.threshold(
        img, np.argmax(hist)+10, 255, cv2.THRESH_TOZERO)
    return add_img
    return cv2.add(add_img, thresh4, dtype=cv2.CV_8UC1)


def cropp(img):
    h, w = img.shape
    top = 0
    down = 0
    left = 0
    right = 0

    halt = False
    for i in range(h):
        if halt:
            break
        for j in range(w):
            if img[i, j] == 0:
                halt = True
                top = i-1
                break

    halt = False
    for i in reversed(range(h)):
        if halt:
            break
        for j in range(w):
            if img[i, j] == 0:
                halt = True
                down = i+1
                break

    halt = False
    for i in range(w):
        if halt:
            break
        for j in range(h):
            if img[j, i] == 0:
                halt = True
                left = i-1
                break

    halt = False
    for i in reversed(range(w)):
        if halt:
            break
        for j in range(h):
            if img[j, i] == 0:
                halt = True
                right = i+1
                break

    if (top < 0):
        top = 0
    if (down < 0):
        down = 0
    if (left < 0):
        left = 0
    if (right < 0):
        right = 0
    return img[top:down, left:right]


def preprocessing(img):
    img_copy = img.copy()
    img = cv2.cvtColor(img_copy, cv2.COLOR_RGB2GRAY)
    thresh = 255-histogram_norm(img)
    thresh = preprocess(thresh)
    thresh = cropp(thresh)
    return thresh
