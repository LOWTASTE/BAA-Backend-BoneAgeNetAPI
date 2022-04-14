import cv2
import numpy as np
import os


def crop(img, mask):
    index = np.where(mask > 0)
    top = np.min(index[0])
    bottom = np.max(index[0])
    left = np.min(index[1])
    right = np.max((index[1]))
    # extract hand region
    # if top > 200:
    #     top =top -200
    # elif top > 100:
    #     top = top -100

    # extract region1
    # if left>100:
    #     left=left-70
    croped_img = img[top:bottom, left:right]
    return croped_img


def maskout(img, mask):
    index = np.where(mask > 0)
    top = np.min(index[0])
    bottom = np.max(index[0])
    left = np.min(index[1])
    right = np.max((index[1]))
    img[top:bottom, left:right] = np.random.randint(255)
    return img


def find_max_component(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    area = []
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))
    max_ind = np.argmax(area)
    print(area)
    for ind in range(len(contours)):
        if ind != max_ind:
            cv2.fillConvexPoly(mask, contours[ind], 0)
    return mask
