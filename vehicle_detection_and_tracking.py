import argparse
import os
import cv2
from ultralytics import YOLO

# Vehicle class names in COCO dataset
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck (COCO IDs)
CLASS_NAME_MAP = {
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck',
}

def main(input_path):
    print(f"Processing video: {input_path}")
    
    # Check if input video exists
    if not os.path.exists(input_path):
        print(f"Error: Video file not found at {input_path}")
        return
    
    # Output path
    os.makedirs('dataset/Output_Data/Object_Detection', exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = f'dataset/Output_Data/Object_Detection/{base_name}_detected.mp4'

    # Frame and label output directories
    frames_dir = 'dataset/Output_Data/Frames_Detected'
    labels_dir = 'dataset/labels/Frames'
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Load YOLOv8 model
    model = YOLO('yolov8n.pt')

    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Tracking state
    id_map = {}  # track_id: serial_no
    next_serial = 1

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Save frame as image
        frame_filename = f"{base_name}_frame_{frame_idx}.jpg"
        frame_path = os.path.join(frames_dir, frame_filename)
        cv2.imwrite(frame_path, frame)

        # Run YOLO detection for label generation (all classes)
        results = model(frame)
        boxes = results[0].boxes
        yolo_lines = []
        if boxes is not None and boxes.xyxy is not None:
            for box, cls in zip(boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy().astype(int)):
                x1, y1, x2, y2 = box
                x_center = ((x1 + x2) / 2) / width
                y_center = ((y1 + y2) / 2) / height
                w = (x2 - x1) / width
                h = (y2 - y1) / height
                yolo_lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
        label_filename = f"{base_name}_frame_{frame_idx}.txt"
        label_path = os.path.join(labels_dir, label_filename)
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_lines))

        # Run YOLOv8 tracking for vehicle annotation
        results_track = model.track(frame, persist=True, classes=VEHICLE_CLASSES, verbose=False)
        if results_track[0].boxes.id is not None:
            boxes = results_track[0].boxes.xyxy.cpu().numpy()
            ids = results_track[0].boxes.id.cpu().numpy().astype(int)
            clss = results_track[0].boxes.cls.cpu().numpy().astype(int)
            for box, track_id, cls in zip(boxes, ids, clss):
                if track_id not in id_map:
                    id_map[track_id] = next_serial
                    next_serial += 1
                serial_no = id_map[track_id]
                class_name = CLASS_NAME_MAP.get(cls, str(cls))
                # NOTE: If you want to detect rickshaws, you need a model trained on that class.
                x1, y1, x2, y2 = map(int, box)
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)  # thinner box
                label = f"{class_name} #{serial_no}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)  # smaller font
        out.write(frame)
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"Processed {frame_idx} frames...")

    cap.release()
    out.release()
    print(f"Output saved to {output_path}")

if __name__ == "__main__":
    # Set your input video path here
    input_path = "dataset/input_video/Video1.mp4"  # Change this to your desired video
    main(input_path) 