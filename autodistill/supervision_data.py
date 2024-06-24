# step 1 of 3: extract frames from videos
# code from https://blog.roboflow.com/autodistill/
# replaces tqdm with os

import os
import supervision as sv

VIDEO_DIR_PATH = "D:/videos"
IMAGE_DIR_PATH = "./data/input_from_video"
FRAME_STRIDE = 10  # one image for every 10 frames
MAX_IMAGES_PER_VIDEO = 1000  # Maximum number of images to save per video

video_paths = sv.list_files_with_extensions(
    directory=VIDEO_DIR_PATH, 
    extensions=["mov", "mp4"]
)

VIDEO_COUNT = int(len(video_paths) / 2)
# TEST_VIDEO_PATHS, TRAIN_VIDEO_PATHS = video_paths[:VIDEO_COUNT], video_paths[VIDEO_COUNT:]
total_videos = len(video_paths)

print(f"Processing {total_videos} videos...")

for index, video_path in enumerate(video_paths):
    print(f"Processing video {index + 1}/{total_videos}: {video_path}")
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    image_name_pattern = video_name + "-{:05d}.png"
    image_count = 0  # Initialize image count for each video
    
    with sv.ImageSink(target_dir_path=IMAGE_DIR_PATH, image_name_pattern=image_name_pattern) as sink:
        for image in sv.get_video_frames_generator(source_path=str(video_path), stride=FRAME_STRIDE):
            if image_count < MAX_IMAGES_PER_VIDEO:  # Check if maximum limit is reached
                sink.save_image(image=image)
                image_count += 1
            else:
                break  # Stop processing if maximum images are saved

print("Processing complete.")
