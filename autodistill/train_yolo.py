# step 3 of 3 train the yolov8 model
# runs 50 epochs with a dataset of 100 training images
# took about 10 minutes to run on CPU

from autodistill_yolov8 import YOLOv8
from IPython.display import Image
import torch

# pip install ipython

device = 'cpu' 
# hit a runtime mutli-threading error that led me back to using CPU
# if torch.cuda.is_available():
#     device = torch.cuda.current_device()
    
print(f"Using device: {device}")

YOLO_DATASET_DIR_PATH = "./data/yolo_dataset"
DATA_YAML_PATH = f"{YOLO_DATASET_DIR_PATH}/data.yaml"

target_model = YOLOv8("yolov8n.pt")
target_model.train(DATA_YAML_PATH, epochs=50, device=device)

Image(filename=f'./runs/detect/train/confusion_matrix.png', width=600)