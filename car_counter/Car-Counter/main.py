from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture("../Resources/cars.mp4")

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

while True:
    success,img = cap.read()
    results = model(img,stream = True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w,h = x2-x1, y2-y1


            #confidence
            conf = math.ceil((box.conf[0] *100)) / 100

            #class
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            if (currentClass == "motorbike" or currentClass == "car" or currentClass == "bus" or currentClass == "truck") and conf>0.4:
                cvzone.putTextRect(img,f"{currentClass} {conf}",(max(0,x1),max(35,y1)), scale = 1,thickness = 2, offset = 3)
                cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt= 2, colorR=(255, 0, 255))

    cv2.imshow("Image",img)
    cv2.waitKey(1)