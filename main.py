import os

#make folder for the data
os.makedirs('./data/input', exist_ok=True)

os.makedirs('./data/output_detr_resnet', exist_ok=True)
os.makedirs('./data/output_fasterrcnn', exist_ok=True)
os.makedirs('./data/output_gpt4o', exist_ok=True)
os.makedirs('./data/output_yolov5', exist_ok=True)
os.makedirs('./data/output_yolov8', exist_ok=True)

print('finished making output folders')