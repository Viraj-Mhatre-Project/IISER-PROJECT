import cv2
import os

# Process all videos in dataset/numPlate
video_dir = 'dataset/numPlate'
output_dir = 'dataset/images/Frames'
os.makedirs(output_dir, exist_ok=True)

    for video_file in os.listdir(video_dir):
        if not video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            continue
        video_path = os.path.join(video_dir, video_file)
        video_name = os.path.splitext(video_file)[0]

        cap = cv2.VideoCapture(video_path)
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
        frame_filename = f"{output_dir}/{video_name}_frame_{i}.jpg"
        cv2.imwrite(frame_filename, frame)
            i += 1
        cap.release()
    print(f"Extracted {i} frames from {video_file} and saved to {output_dir}.")
