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


def remove_entering_fish(column, column_entering, next_column):
    # remove fish that are entering the current frame if on the previous 
    # frame we already detected that they are entering (column_entering == 255)
    column = (((column == 255) & (column_entering == 0)).astype(np.uint8) * 255)

    # detect fish entering the frame
    # current collumn is 255 (fish) and next column is 0 (background)
    positions_to_delete = (column == 255) & (next_column == 0)

    # delete fish entering and save its position for the next frame
    column[positions_to_delete] = 0
    column_entering[positions_to_delete] = 0

    return column, column_entering

