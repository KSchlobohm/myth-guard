# does not use a confidence threshold to filter out low-confidence detections
# this model is trained on the COCO dataset, which contains 80 classes of objects

import torch
from PIL import Image
import cv2

# pip install torch torchvision opencv-python-headless numpy pandas requests ultralytics

# Load the custom trained model
#model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')  # path to your trained model
model = torch.hub.load('ultralytics/yolov5', 'yolov5x')  # Load the extra large model from torch.hub

for i in range(5):
    # Load an image
    img_path = f'./data/input/image{i+1}.png'
    img = Image.open(img_path)

    # Inference
    results = model(img)

    # Results
    results.print()  # Print results to console
    #results.save()   # Save detection results to runs/detect/exp
    #results.show()   # Display detection results

    # Draw bounding boxes on the image (using OpenCV for visualization)
    image_cv = cv2.imread(img_path)
    for detection in results.xyxy[0].numpy():
        x1, y1, x2, y2, conf, cls = detection
        cv2.rectangle(image_cv, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        # Get the class label and confidence score
        label = f"{results.names[cls]}: {conf:.2f}"
        cv2.putText(image_cv, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    output_path = f'./data/output_yolov5/image{i+1}.png'
    cv2.imwrite(output_path, image_cv)
