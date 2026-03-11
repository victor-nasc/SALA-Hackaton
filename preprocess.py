import cv2
import numpy as np


def get_fish_mask(frame, bg_sub, clahe):
    w, h = frame.shape[1], frame.shape[0]
    cv2.rectangle(frame, (int(w*0.05), int(h*0.05)), (int(w*0.95), int(h*0.95)), (255, 255, 255), -1)

    frame = cv2.medianBlur(frame, 11)
    
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl,a,b))
    frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    mask = bg_sub.apply(frame) 
    mask[mask == 127] = 255

    mask = cv2.medianBlur(mask, 25)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9,9), np.uint8))

    return mask


def clean_VR_image(stack):
    stack = cv2.medianBlur(stack, 3)
    stack = cv2.dilate(stack, np.ones((5, 5), np.uint8), iterations=1)
    return stack


def get_clusters(vector):
    vector = vector.flatten()
    
    # indices that change from 0 <--> 255
    changes = np.where(np.diff(vector) != 0)[0] + 255
    
    if vector[-1] == 255:
        changes = np.append(changes, len(vector))
    if vector[0] == 255:
        changes = np.append(0, changes)

    indices = changes.reshape(-1, 2)
    indices[:, 1] -= 1  

    return indices


def remove_entering_fish(column, next_column):
    clusters = get_clusters(column)
    for start, end in clusters:
        if next_column[start:end].any():
            column[start:end] = 0

    return column

