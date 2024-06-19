# uses a 10% confidence threshold to filter out low-confidence detections
# this model is trained on the COCO dataset, which contains 80 classes of objects

import torch
from transformers import DetrForObjectDetection, DetrImageProcessor
from PIL import Image
import requests
import cv2
import numpy as np

# DEtection TRansformer (DETR) model trained end-to-end on COCO 2017 object detection (118k annotated images)
# https://huggingface.co/facebook/detr-resnet-50

# pip install transformers timm

# Load pre-trained model and processor
model_name = "facebook/detr-resnet-50"
model = DetrForObjectDetection.from_pretrained(model_name)
processor = DetrImageProcessor.from_pretrained(model_name)

# Define the threshold for detection
threshold = 0.1

# Load the image
for i in range(5):
    
    # Load an image
    image_path = f'./data/input/image{i+1}.png'
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    # Prepare the image for the model
    inputs = processor(images=image, return_tensors="pt")

    # Perform object detection
    outputs = model(**inputs)

    # Extract boxes and scores
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

    # Draw boxes on the image
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if score > threshold:
            box = [round(i) for i in box.tolist()]
            x, y, w, h = box
            cv2.rectangle(image_np, (x, y), (w, h), (0, 255, 0), 2)
            cv2.putText(image_np, f'{model.config.id2label[label.item()]}: {score:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the result image
    #output_path = '/mnt/data/detected_units.png'
    output_path = f'./data/output_detr_resnet/image{i+1}.png'
    cv2.imwrite(output_path, image_np)

    # Optionally, display the image
    # cv2.imshow('Detected Units', image_np)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

print('detr-resnet-50: object detection completed.')