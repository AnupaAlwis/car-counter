import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap = cv2.VideoCapture("../Resources/cars.mp4")

# cap = cv2.VideoCapture(1)
# cap.set(3, 1080)
# cap.set(4, 720)

model = YOLO("../YOLO-Weights/yolov8n.pt")

# This data set is based onthe COCO dataset this has the values for each of the class number output from the model
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask = cv2.imread("../Resources/mask.png")  # Import the created mask

# tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
limits = [400, 297, 673, 297]  # selected points to draw the line on the image
totalCount = []

while True:
    success, img = cap.read()
    mask_rezied = cv2.resize(mask, (img.shape[1], img.shape[0]))  # resize the mask to match the img size
    imgRegion = cv2.bitwise_and(img, mask_rezied)  # Region of the image that we will be counting
    results = model(img, stream=True)  # Give results only from the mask region
    detections = np.empty((0, 5))  # empty array to store the detections

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2 - x1, y2 - y1

            # confidence
            conf = math.ceil((box.conf[0] * 100)) / 100

            # class
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if (
                    currentClass == "motorbike" or currentClass == "car" or currentClass == "bus" or currentClass == "truck") and conf > 0.4:
                # cvzone.putTextRect(img,f"{currentClass} {conf}",(max(0,x1),max(35,y1)), scale = 2,thickness = 3, offset = 3)
                # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt= 2, colorR=(255, 0, 255))
                currentArray = np.array([x1, y1, x2, y2, conf])  # store the detections
                detections = np.vstack((detections, currentArray))  # stack the detections

    resultsTracker = tracker.update(detections)  # Gives the resuts in the format [[x1, y1, x2, y2, ID],...]
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)  # draw the line based on the points

    for result in resultsTracker:
        x1, y1, x2, y2, id = result  # result consist of coordinates of the cox and id
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2,
                          colorR=(255, 0, 0))  # draw the rectangle of the sorted detector
        cvzone.putTextRect(img, f"{id}", (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=10)  # show id number
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        cv2.circle(img, (int(cx), int(cy)), 5, (255, 0, 255), cv2.FILLED)
        if limits[0] < int(cx) < limits[2] and limits[1] - 15 < int(cy) < limits[3] + 15:
            if totalCount.count(id) == 0:
                totalCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)
    cvzone.putTextRect(img, f"Count: {len(totalCount)}", (50, 50), scale=3, thickness=5, offset=20)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
