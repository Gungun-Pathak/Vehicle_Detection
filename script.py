from collections import Counter
from ultralytics import YOLO
import cv2

model_path = 'best.pt'  
model = YOLO(model_path)


def predict_single_image_counts(image_path, model):
    
    results = model.predict(image_path, save=False, imgsz=320, conf=0.5)


    label_counts = Counter()


    for result in results:
        for box in result.boxes:
            class_label = int(box.cls[0])
            if class_label == 0:
                label = "Coming"
            elif class_label == 1:
                label = "Going"
            elif class_label == 2:
                label = "Empty"
            else:
                label = "Unknown"

            label_counts[label] += 1


    for label, count in label_counts.items():
        print(f"Total {label}: {count}")

 
    if not label_counts:
        print("No vehicles detected (Empty).")


image_path = 'car_vehicle.jpg'  
predict_single_image_counts(image_path, model)
