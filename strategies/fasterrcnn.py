# uses a 50% confidence threshold to filter out low-confidence detections
# this model is trained on the COCO dataset, which contains 80 classes of objects

import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np

# pip install numpy<2 opencv-python matplotlib torch torchvision

# object detection using a pre-trained Faster R-CNN model with a ResNet-50 backbone and a Feature Pyramid Network (FPN)
#  ResNet-50 acts as the backbone in Faster R-CNN, meaning it processes the input image and extracts a rich set of features.
#  These features are then passed to other parts of the Faster R-CNN model (such as the Region Proposal Network and the
#  detection head) to identify and classify objects.
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.ToTensor()
])

for i in range(5):
    # Load the image
    image_path = f'./data/input/image{i+1}.png'
        
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    # Perform the detection
    with torch.no_grad():
        predictions = model(image_tensor)

    # all 80 object labels from the COCO dataset
    LABELS = [
        'N/A', 'Person', 'Bicycle', 'Car', 'Motorcycle', 'Airplane', 'Bus', 'Train', 'Truck',
        'Boat', 'Traffic light', 'Fire hydrant', 'N/A', 'Stop sign', 'Parking meter', 'Bench',
        'Bird', 'Cat', 'Dog', 'Horse', 'Sheep', 'Cow', 'Elephant', 'Bear', 'Zebra', 'Giraffe',
        'N/A', 'Backpack', 'Umbrella', 'N/A', 'N/A', 'Handbag', 'Tie', 'Suitcase', 'Frisbee',
        'Skis', 'Snowboard', 'Sports ball', 'Kite', 'Baseball bat', 'Baseball glove',
        'Skateboard', 'Surfboard', 'Tennis racket', 'Bottle', 'N/A', 'Wine glass', 'Cup',
        'Fork', 'Knife', 'Spoon', 'Bowl', 'Banana', 'Apple', 'Sandwich', 'Orange', 'Broccoli',
        'Carrot', 'Hot dog', 'Pizza', 'Donut', 'Cake', 'Chair', 'Couch', 'Potted plant', 'Bed',
        'N/A', 'Dining table', 'N/A', 'N/A', 'Toilet', 'N/A', 'TV', 'Laptop', 'Mouse', 'Remote',
        'Keyboard', 'Cell phone', 'Microwave', 'Oven', 'Toaster', 'Sink', 'Refrigerator',
        'N/A', 'Book', 'Clock', 'Vase', 'Scissors', 'Teddy bear', 'Hair drier', 'Toothbrush'
    ]

    # Extract detection results
    threshold = 0.5
    boxes = predictions[0]['boxes'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()

    # Filter out low-confidence detections
    filtered_boxes = boxes[scores >= threshold].astype(np.int32)
    filtered_labels = labels[scores >= threshold]
    filtered_scores = scores[scores >= threshold]

    # Load the image with OpenCV for visualization
    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Visualize the results
    for box, label, score in zip(filtered_boxes, filtered_labels, filtered_scores):
        if label < len(LABELS):
            cv2.rectangle(image_cv2, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            label_text = f"{LABELS[label]}: {score:.2f}"
            cv2.putText(image_cv2, label_text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the output
    # plt.figure(figsize=(12, 8))
    # plt.imshow(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.show()

    output_path = f'./data/output_fasterrcnn/image{i+1}.png'
    cv2.imwrite(output_path, image_cv2)
