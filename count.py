import cv2
import numpy as np

from preprocess import get_fish_mask, clean_VR_image, remove_entering_fish


video_path = '10s.mp4'
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Error: Could not open video."

bg_sub = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400.0, detectShadows=False)
clahe = cv2.createCLAHE(clipLimit=7.0, tileGridSize=(8,8))

stack = None
delta_lin = 120

while True:
    ret, frame = cap.read()
    if not ret:
        break

    mask = get_fish_mask(frame, bg_sub, clahe)

    # build VR image
    column = mask[delta_lin:-delta_lin, 0].reshape(1, -1)

    next_column = mask[delta_lin:-delta_lin, 1].reshape(1, -1)
    column = remove_entering_fish(column, next_column)

    stack = column if stack is None else np.vstack((stack, column))

    if not ret or cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.line(frame, (0, delta_lin), (0, 1080 - delta_lin), (0, 0, 255), 10)
    cv2.imshow('Mask', mask)
    cv2.imshow('SALAditos', frame)

    print(stack.shape[0], end='\r')

stack = clean_VR_image(stack)

# add bounding boxes to the stack
contours, _ = cv2.findContours(stack, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
stack_with_boxes = cv2.cvtColor(stack, cv2.COLOR_GRAY2BGR)
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if w > 7 and h > 7:  # Filter small contours
        cv2.rectangle(stack_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 1)

# frames = detect_frames()

cv2.imwrite('stack_with_boxes.png', stack_with_boxes)

cap.release()
cv2.destroyAllWindows()



