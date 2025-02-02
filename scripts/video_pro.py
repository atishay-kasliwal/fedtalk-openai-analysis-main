# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 19:07:29 2024
@author: HP
"""

###Video auto process




###000  Video cut 5 min
from moviepy.video.io.VideoFileClip import VideoFileClip

def split_video(input_file, output_folder, interval=300):
    # Load the video file
    video = VideoFileClip(input_file)
    
    # Get the duration of the video in seconds
    duration = int(video.duration)
    
    # Split the video into smaller clips
    for start_time in range(0, duration, interval):
        end_time = min(start_time + interval, duration)
        clip = video.subclip(start_time, end_time)
        output_file = f"{output_folder}/clip_{start_time // 60}m_{end_time // 60}m.mp4"
        clip.write_videofile(output_file, codec="libx264")
    
    video.close()

# usage
input_file = "D:/Illinois2023/Projects/项目/项目/video/2024Feb.mp4"
output_folder = "D:/Illinois2023/Projects/项目/项目/vision&facial/cut"
split_video(input_file, output_folder)







###001  Video cut into images
import cv2
import os





def extract_frames_at_intervals(video_path, interval_sec, output_folder):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError("Error opening video file")
    
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    frames = []
    timestamps = []
    success, frame = video.read()
    count = 0

    while success:
        if count % int(fps * interval_sec) == 0:  # Capture frame at every interval_sec
            frames.append(frame)
            timestamps.append(count / fps)
        success, frame = video.read()
        count += 1

    video.release()
    return frames, timestamps, duration







def process_videos_in_folder(input_folder, interval_sec, output_base_folder):
    # Ensure the output base folder exists
    os.makedirs(output_base_folder, exist_ok=True)

    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".mp4"):
            video_path = os.path.join(input_folder, filename)
            video_name = os.path.splitext(filename)[0]
            output_folder = os.path.join(output_base_folder, video_name)

            # Ensure the output folder for this video exists
            os.makedirs(output_folder, exist_ok=True)

            frames, timestamps, duration = extract_frames_at_intervals(video_path, interval_sec, output_folder)

            # Save the frames as images
            for i, frame in enumerate(frames):
                output_path = os.path.join(output_folder, f'frame_{i}.png')
                cv2.imwrite(output_path, frame)

            print(f'Extracted {len(frames)} frames from {filename} at {interval_sec}-second intervals and saved to {output_folder}.')

# Path to the folder containing video files
input_folder = "D:\\Illinois2023\\Projects\\Fedtalk-Facial\\vision&facial\\2023Feb_cut\\videos"
output_base_folder = "D:\\Illinois2023\\Projects\\Fedtalk-Facial\\vision&facial\\2023Feb_cut\\output_frames"

interval_sec = 10
process_videos_in_folder(input_folder, interval_sec, output_base_folder)
