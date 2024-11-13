from collections import Counter
from ultralytics import YOLO
import cv2

# Load the pre-trained YOLO model
model_path = r'C:\Users\Suraj\Desktop\CodeWhy\codebase\vehicle_detection\Vehicle_Detection\best.pt'
model = YOLO(model_path)

# Print model info to ensure it's loaded correctly
print("Loaded model:", model)

# Initialize the webcam (or use a video file if needed)
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam or specify a video file path

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Check the class names in the model
print("Class names:", model.names)

def predict_frame_counts(frame, model):
    # Perform the inference on the current frame
    results = model(frame, save=False, imgsz=320, conf=0.5)

    # Check the results of the inference
    print("Inference results:", results)  # Debugging the inference output

    label_counts = Counter()

    # Process the results from the model
    for result in results:
        # Inspect the result box structure for debugging
        print("Result boxes:", result.boxes)  # Check what the model returns

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

    return label_counts

# Loop to continuously capture frames from the webcam
while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Get label counts for the current frame
    label_counts = predict_frame_counts(frame, model)

    # Display the counts on the frame
    for label, count in label_counts.items():
        cv2.putText(frame, f"{label}: {count}", (10, 30 + 30 * list(label_counts.keys()).index(label)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # If no vehicles are detected, display a message
    if not label_counts:
        cv2.putText(frame, "No vehicles detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame with detection results
    cv2.imshow('Vehicle Detection', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
