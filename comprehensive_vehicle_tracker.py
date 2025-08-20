import os
import cv2
import numpy as np
from ultralytics import YOLO

# SORT Tracker Implementation (simplified version)
class Sort:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        
    def update(self, detections):
        self.frame_count += 1
        
        # If no detections, update all trackers
        if len(detections) == 0:
            return np.empty((0, 5))
        
        # If no trackers exist, create new ones
        if len(self.trackers) == 0:
            for det in detections:
                self.trackers.append(self._create_tracker(det))
            return np.array([np.append(det, [i]) for i, det in enumerate(detections)])
        
        # Update existing trackers
        tracked_objects = []
        for i, tracker in enumerate(self.trackers):
            if tracker is not None:
                # Simple tracking: use detection as new position
                if i < len(detections):
                    tracker = detections[i]
                    tracked_objects.append(np.append(tracker, [i]))
        
        return np.array(tracked_objects) if tracked_objects else np.empty((0, 5))
    
    def _create_tracker(self, detection):
        return detection

# Vehicle class names in COCO dataset
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck (COCO IDs)
CLASS_NAME_MAP = {
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck',
}

def calculate_speed(prev_pos, curr_pos, fps, pixel_to_meter_ratio=0.05):
    """Calculate speed in pixels/frame and km/h with better accuracy"""
    if prev_pos is None:
        return 0, 0
    
    dx = curr_pos[0] - prev_pos[0]
    dy = curr_pos[1] - prev_pos[1]
    distance_pixels = np.sqrt(dx*dx + dy*dy)
    
    # More realistic pixel-to-meter conversion (adjust this based on your video)
    # For typical traffic videos, 1 pixel ≈ 0.05 meters
    distance_meters = distance_pixels * pixel_to_meter_ratio
    
    # Calculate speed in m/s
    speed_mps = distance_meters * fps
    
    # Convert to km/h
    speed_kmh = speed_mps * 3.6
    
    # Apply realistic speed limits (typical urban traffic: 0-80 km/h)
    if speed_kmh > 80:
        speed_kmh = 80  # Cap at reasonable urban speed
    
    return distance_pixels, speed_kmh

def smooth_speed(speeds, window_size=5):
    """Apply moving average smoothing to speed values"""
    if len(speeds) < window_size:
        return speeds
    
    smoothed = []
    for i in range(len(speeds)):
        start = max(0, i - window_size + 1)
        end = i + 1
        window_speeds = speeds[start:end]
        smoothed.append(sum(window_speeds) / len(window_speeds))
    
    return smoothed

def calculate_journey_speed(entry_frame, exit_frame, entry_pos, exit_pos, fps, pixel_to_meter_ratio=0.05):
    """Calculate speed based on total journey through the video"""
    if entry_frame is None or exit_frame is None:
        return 0
    
    # Calculate total time in video (in seconds)
    total_frames = exit_frame - entry_frame
    total_time_seconds = total_frames / fps
    
    # Calculate total distance traveled (in pixels)
    dx = exit_pos[0] - entry_pos[0]
    dy = exit_pos[1] - entry_pos[1]
    total_distance_pixels = np.sqrt(dx*dx + dy*dy)
    
    # Convert to meters
    total_distance_meters = total_distance_pixels * pixel_to_meter_ratio
    
    # Calculate average speed (m/s)
    if total_time_seconds > 0:
        avg_speed_mps = total_distance_meters / total_time_seconds
        avg_speed_kmh = avg_speed_mps * 3.6
        
        # Apply realistic speed limits
        if avg_speed_kmh > 80:
            avg_speed_kmh = 80
    else:
        avg_speed_kmh = 0
    
    return avg_speed_kmh, total_distance_meters, total_time_seconds

def main(input_path):
    print(f"Processing video: {input_path}")
    
    # Check if input video exists
    if not os.path.exists(input_path):
        print(f"Error: Video file not found at {input_path}")
        return
    
    # Create output directories
    os.makedirs('dataset/Output_Data/Object_Detection', exist_ok=True)
    os.makedirs('dataset/Output_Data/Frames_Detected', exist_ok=True)
    os.makedirs('dataset/labels/Frames', exist_ok=True)
    os.makedirs('dataset/Speed', exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = f'dataset/Output_Data/Object_Detection/{base_name}_detected.mp4'
    
    # Load YOLOv8 model
    model = YOLO('yolov8n.pt')
    
    # Initialize SORT tracker
    tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
    
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
    track_history = {}  # track_id: [positions]
    vehicle_classes = {}  # track_id: vehicle_class_name
    vehicle_journey = {}  # track_id: {entry_frame, exit_frame, entry_pos, exit_pos}
    frame_idx = 0
    
    print("Starting processing...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1. EXTRACT AND SAVE FRAME
        frame_filename = f"{base_name}_frame_{frame_idx}.jpg"
        frame_path = os.path.join('dataset/Output_Data/Frames_Detected', frame_filename)
        cv2.imwrite(frame_path, frame)
        
        # 2. OBJECT DETECTION (YOLOv8)
        results = model(frame, classes=VEHICLE_CLASSES)
        detections = []
        detection_classes = []  # Store class information
        
        if results[0].boxes is not None and results[0].boxes.xyxy is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for box, cls in zip(boxes, classes):
                detections.append(box)
                detection_classes.append(cls)
        
        # 3. OBJECT TRACKING (SORT)
        tracked_objects = tracker.update(np.array(detections))
        
        # 4. GENERATE LABELS AND DRAW ANNOTATIONS
        yolo_lines = []
        
        for i, track in enumerate(tracked_objects):
            x1, y1, x2, y2, track_id = track
            
            # Convert to integers for drawing
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            track_id = int(track_id)
            
            # Get vehicle class name
            vehicle_class = "unknown"
            if i < len(detection_classes):
                cls_id = detection_classes[i]
                vehicle_class = CLASS_NAME_MAP.get(cls_id, "unknown")
            
            # Calculate centroid for tracking
            centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            # Update track history and journey tracking
            if track_id not in track_history:
                track_history[track_id] = []
                vehicle_classes[track_id] = vehicle_class
                vehicle_journey[track_id] = {
                    'entry_frame': frame_idx,
                    'entry_pos': centroid,
                    'exit_frame': None,
                    'exit_pos': None
                }
            
            track_history[track_id].append(centroid)
            
            # Update exit position (will be updated each frame until vehicle disappears)
            vehicle_journey[track_id]['exit_frame'] = frame_idx
            vehicle_journey[track_id]['exit_pos'] = centroid
            
            # Draw bounding box
            color = (0, 255, 0)  # Green
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw unique serial number and vehicle type above the bounding box
            serial_text = f"#{track_id} {vehicle_class}"
            # Position text above the box
            text_x = x1
            text_y = y1 - 10
            
            # Draw serial number and vehicle type text
            cv2.putText(frame, serial_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Generate YOLO format label (for all detected objects)
            x_center = ((x1 + x2) / 2) / width
            y_center = ((y1 + y2) / 2) / height
            w = (x2 - x1) / width
            h = (y2 - y1) / height
            yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
        
        # 5. SAVE LABEL FILE
        label_filename = f"{base_name}_frame_{frame_idx}.txt"
        label_path = os.path.join('dataset/labels/Frames', label_filename)
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_lines))
        
        # 6. WRITE FRAME TO OUTPUT VIDEO
        out.write(frame)
        
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"Processed {frame_idx} frames...")
    
    # Calculate journey-based speeds and save to file
    speed_file_path = f'dataset/Speed/{base_name}_vehicle_speeds.txt'
    with open(speed_file_path, 'w') as f:
        f.write("Vehicle Serial No | Vehicle Type | Avg Speed (km/h) | Distance (m) | Time (s) | Frames Tracked\n")
        f.write("-" * 95 + "\n")
        
        for track_id, journey in vehicle_journey.items():
            if journey['entry_frame'] is not None and journey['exit_frame'] is not None:
                avg_speed, distance, time = calculate_journey_speed(
                    journey['entry_frame'], 
                    journey['exit_frame'], 
                    journey['entry_pos'], 
                    journey['exit_pos'], 
                    fps
                )
                vehicle_type = vehicle_classes.get(track_id, "unknown")
                total_frames = journey['exit_frame'] - journey['entry_frame']
                f.write(f"{track_id:^18} | {vehicle_type:^12} | {avg_speed:^15.2f} | {distance:^12.2f} | {time:^8.2f} | {total_frames:^15}\n")
    
    # Create final frame with journey-based speeds displayed
    final_frame = np.zeros((height, width, 3), dtype=np.uint8)
    final_frame[:] = (0, 0, 0)  # Black background
    
    # Add title
    cv2.putText(final_frame, f"Vehicle Journey Summary - {base_name}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    y_offset = 100
    for track_id, journey in vehicle_journey.items():
        if journey['entry_frame'] is not None and journey['exit_frame'] is not None:
            avg_speed, distance, time = calculate_journey_speed(
                journey['entry_frame'], 
                journey['exit_frame'], 
                journey['entry_pos'], 
                journey['exit_pos'], 
                fps
            )
            vehicle_type = vehicle_classes.get(track_id, "unknown")
            total_frames = journey['exit_frame'] - journey['entry_frame']
            text = f"Vehicle {track_id} ({vehicle_type}): {avg_speed:.2f} km/h, {distance:.1f}m in {time:.1f}s"
            cv2.putText(final_frame, text, (50, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 35
    
    # Save final frame
    final_frame_path = f'dataset/Speed/{base_name}_journey_summary.jpg'
    cv2.imwrite(final_frame_path, final_frame)
    
    # Cleanup
    cap.release()
    out.release()
    
    print(f"Processing complete!")
    print(f"Frames saved to: dataset/Output_Data/Frames_Detected/")
    print(f"Labels saved to: dataset/labels/Frames/")
    print(f"Output video saved to: {output_path}")
    print(f"Journey-based speed data saved to: {speed_file_path}")
    print(f"Journey summary image saved to: {final_frame_path}")
    print(f"Total frames processed: {frame_idx}")

if __name__ == "__main__":
    # Set your input video path here
    input_path = "dataset/input_video/Video1.mp4"
    main(input_path)
