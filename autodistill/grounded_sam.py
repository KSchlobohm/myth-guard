# step 2 of 3: use GroundedSAM to label images from previous step
# code from https://blog.roboflow.com/autodistill/
# code from https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-auto-train-yolov8-model-with-autodistill.ipynb
# cuda getting started from - https://pytorch.org/get-started/locally/
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# tested with CUDA 12.1 - nvidia-smi

from autodistill.detection import CaptionOntology
from autodistill_grounded_sam import GroundedSAM
import supervision as sv
import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available. Please check your CUDA installation.")
    exit()

IMAGE_DIR_PATH = "./data/input_from_video"
YOLO_DATASET_DIR_PATH = "./data/yolo_dataset"

ontology = CaptionOntology({
    "blue square": "wow_raid_marker_square",
    "orange circle": "wow_raid_marker_circle",
    "pruple diamond": "wow_raid_marker_diamond",
})

base_model = GroundedSAM(ontology=ontology)
dataset = base_model.label(
    input_folder=IMAGE_DIR_PATH,
    extension=".png",
    output_folder=YOLO_DATASET_DIR_PATH)


ANNOTATIONS_DIRECTORY_PATH = f"{YOLO_DATASET_DIR_PATH}/train/labels"
IMAGES_DIRECTORY_PATH = f"{YOLO_DATASET_DIR_PATH}/train/images"
DATA_YAML_PATH = f"{YOLO_DATASET_DIR_PATH}/data.yaml"


dataset = sv.DetectionDataset.from_yolo(
    images_directory_path=IMAGES_DIRECTORY_PATH,
    annotations_directory_path=ANNOTATIONS_DIRECTORY_PATH,
    data_yaml_path=DATA_YAML_PATH)

len(dataset)

# let's take a look at the first 16 images

SAMPLE_SIZE = 16
SAMPLE_GRID_SIZE = (4, 4)
SAMPLE_PLOT_SIZE = (16, 16)

image_names = list(dataset.images.keys())[:SAMPLE_SIZE]

mask_annotator = sv.MaskAnnotator()
box_annotator = sv.BoxAnnotator()

images = []
for image_name in image_names:
    image = dataset.images[image_name]
    annotations = dataset.annotations[image_name]
    labels = [
        dataset.classes[class_id]
        for class_id
        in annotations.class_id]
    annotates_image = mask_annotator.annotate(
        scene=image.copy(),
        detections=annotations)
    annotates_image = box_annotator.annotate(
        scene=annotates_image,
        detections=annotations,
        labels=labels)
    images.append(annotates_image)

sv.plot_images_grid(
    images=images,
    titles=image_names,
    grid_size=SAMPLE_GRID_SIZE,
    size=SAMPLE_PLOT_SIZE)