import cv2
import torch
from super_gradients.training import models
img = cv2.imread("images/image.jpg");
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
model = models.get("yolo_nas_s", pretrained_weights="coco").to(device)

out=model.predict("videos/aboda/video1.avi",conf=0.4).save("OutputVideo/aboda1.avi")