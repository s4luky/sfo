#Deteksi SFO dengan pelacakan menggunakan YOLO-NAS dan integrasi FPN pada SORT
import cv2
import torch
from super_gradients.training import models
import math
from sort import *


cap = cv2.VideoCapture("taod1.mp4")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

model = models.get('yolo_nas_s', pretrained_weights="coco").to(device)
def draw_boxes(img, bbox, identities=None,categories=None, names=None, offset=(0,0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[0]
        y2 += offset[0]
        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if categories is not None else 0
        label = str(id) + ":" + classNames[cat]
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (76, 153, 0), 2)
        cv2.rectangle(img, (x1, y1-20), (x1+w, y1), (76,153, 0), 1)
        cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [76,153,0], 1)
    return img
count = 0
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
out = cv2.VideoWriter('Out5c.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

tracker = Sort(max_age = 20, min_hits=3, iou_threshold=0.3)

while True:
    ret, frame = cap.read()
    count += 1
    if ret:
        detections = np.empty((0,6))
        result = list(model.predict(frame, conf=0.35))[0]
        bbox_xyxys = result.prediction.bboxes_xyxy.tolist()
        confidences = result.prediction.confidence
        labels = result.prediction.labels.tolist()
        for (bbox_xyxy, confidence, cls) in zip(bbox_xyxys, confidences, labels):
            bbox = np.array(bbox_xyxy)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            classname = int(cls)
            class_name = classNames[classname]
            conf = math.ceil((confidence*100))/100
            #label = f'{class_name}{conf}'
            #print("Frame N", count, "", x1, y1,x2, y2)
            #t_size = cv2.getTextSize(label, 0, fontScale = 1, thickness=2)[0]
            #c2 = x1 + t_size[0], y1 - t_size[1] -3
            #cv2.rectangle(frame, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)
            #cv2.putText(frame, label, (x1, y1-2), 0, 1, [255, 255, 255], thickness=1, lineType = cv2.LINE_AA)
            #cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            currentArray = np.array([x1, y1, x2, y2, conf, cls])
            detections = np.vstack((detections, currentArray))
            print("Frame N", count, "", x1, y1, x2, y2)
        tracker_dets = tracker.update(detections)
        if len(tracker_dets) >0:
            bbox_xyxy = tracker_dets[:,:4]
            identities = tracker_dets[:, 8]
            categories = tracker_dets[:, 4]
            draw_boxes(frame, bbox_xyxy, identities, categories)
        number = str(count)
        resize_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
      #  cv2.putText(frame, number, (13, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [205, 153, 255], 1)

        out.write(frame)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('1'):
            break
    else:
        break

out.release()
cap.release()
cv2.destroyAllWindows()