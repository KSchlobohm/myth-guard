# does not use a confidence threshold to filter out low-confidence detections
# this model is trained on the COCO dataset, which contains 80 classes of objects

import torch
from PIL import Image
import cv2
from ultralytics import YOLO

# Load the custom model trained from grounded SAM
model = YOLO("./runs/detect/train/weights/best.pt")

for i in range(5):
    
    # Load an image
    img_path = f'./data/input/image{i+1}.jpg'
    img = Image.open(img_path)

    # Inference
    results = model(img)

    # Results
    #results.print()  # Print results to console
    #results.save()   # Save detection results to runs/detect/exp
    #results.show()   # Display detection results

    # Draw bounding boxes on the image (using OpenCV for visualization)
    image_cv = cv2.imread(img_path)
    for box  in results[0].boxes:
        # Get the bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Draw the bounding box
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue box with thickness 2

        # Get the class label and confidence score
        label = f"{results[0].names[int(box.cls)]}: {box.conf.item():.2f}"
        cv2.putText(image_cv, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    output_path = f'./data/output_yolov8_custom/images{i+1}.jpg'
    cv2.imwrite(output_path, image_cv)

    # cv2.imshow('Detected Units', image_cv)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()