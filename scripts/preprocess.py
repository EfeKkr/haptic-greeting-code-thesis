# Extracts frames from raw .mp4 videos, resizes and normalizes them,
# and saves them as .npy arrays for downstream model processing.
# Supports adjustable frame sampling percentage and target output size.

import os
import cv2
import numpy as np

def extract_frames(video_path, output_dir, percentage=1.0, output_size=(224, 224), save_name=None):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        print(f"Skipping {video_path} (no frames)")
        return

    num_frames_to_extract = int(total_frames * percentage)
    frame_indices = np.linspace(0, total_frames - 1, num_frames_to_extract, dtype=np.int32)

    extracted_frames = []

    for idx in range(total_frames):
        success, frame = cap.read()
        if not success:
            break
        if idx in frame_indices:
            resized = cv2.resize(frame, output_size)
            normalized = resized.astype(np.float32) / 255.0
            extracted_frames.append(normalized)

    cap.release()

    frames_array = np.stack(extracted_frames)

    if save_name is None:
        save_name = os.path.splitext(os.path.basename(video_path))[0] + f"_{int(percentage*100)}.npy"
    
    save_path = os.path.join(output_dir, save_name)
    np.save(save_path, frames_array)
    print(f"Saved: {save_path}")

def process_all_videos(input_dir, output_dir, percentage=0.6):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith(".mp4"):
            video_path = os.path.join(input_dir, filename)
            extract_frames(video_path, output_dir, percentage=percentage)

if __name__ == "__main__":
    # Make sure to place the original .mp4 video files inside the 'data/' directory before running this script.
    input_dir = "data" # 
    output_dir = "processed_data"
    process_all_videos(input_dir, output_dir, percentage=1.0)
