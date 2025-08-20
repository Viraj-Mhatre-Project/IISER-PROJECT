<<<<<<< HEAD
# Vehicle Detection and Tracking with SORT Algorithm

This project implements a complete vehicle detection and tracking system using **SORT (Simple Online and Realtime Tracking)** algorithm and **YOLOv8** for object detection.

## 🚗 Features

- **Vehicle Detection**: Uses YOLOv8 to detect cars, motorcycles, buses, and trucks
- **SORT Tracking**: Implements the SORT algorithm for robust multi-object tracking
- **Frame Extraction**: Saves all processed frames with detections
- **Video Output**: Creates a tracked video with bounding boxes and IDs
- **Tracking Data**: Exports tracking results to CSV format
- **Vehicle Counting**: Counts vehicles crossing designated lines
- **Trajectory Visualization**: Shows vehicle movement paths

## 📁 Project Structure

```
IISER/
├── sort.py                           # SORT algorithm implementation
├── vehicle_detection_tracking.py     # Main detection and tracking script
├── requirements.txt                  # Python dependencies
├── yolov8n.pt                       # YOLO model weights
└── dataset/
    ├── Input_Video/                 # Your input video
    └── Output_Data/
        ├── Frames_Detected/         # Individual frames with detections
        ├── Object_Detection/        # Output tracked video
        └── Tracking_Results/        # Tracking data CSV files
```

## 🛠️ Installation

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify installation**:
   ```bash
   python -c "import cv2, numpy, ultralytics, sort; print('✅ All dependencies installed!')"
   ```

## 🚀 Usage

### Quick Start

Run the main vehicle detection and tracking script:

```bash
python vehicle_detection_tracking.py
```

This will:
1. Load your video from `dataset/Input_Video/Video1.mp4`
2. Detect vehicles using YOLOv8
3. Track vehicles using SORT algorithm
4. Save individual frames with detections
5. Create output video with tracking
6. Export tracking results to CSV

## 📊 Output Files

### 1. Individual Frames
- **Location**: `dataset/Output_Data/Frames_Detected/`
- **Format**: `Video1_frame_XXXX_detected.jpg`
- **Content**: Each frame with bounding boxes, track IDs, and trajectories

### 2. Tracked Video
- **Location**: `dataset/Output_Data/Object_Detection/`
- **Format**: `Video1_SORT_tracked.mp4`
- **Content**: Complete video with vehicle tracking visualization

### 3. Tracking Results
- **Location**: `dataset/Output_Data/Tracking_Results/`
- **Format**: `Video1_tracking_results.csv`
- **Content**: Frame-by-frame tracking data (Frame, Vehicle_ID, Class, Confidence, X1, Y1, X2, Y2)

## 🎯 SORT Algorithm Details

### How It Works
1. **Detection**: YOLOv8 detects vehicles in each frame
2. **Prediction**: Kalman filter predicts vehicle positions
3. **Association**: Hungarian algorithm matches detections to tracks
4. **Update**: Updates track states with new detections
5. **Creation**: Creates new tracks for unmatched detections
6. **Deletion**: Removes tracks that haven't been updated recently

### Parameters
- `max_age=30`: Maximum frames to keep track without detection
- `min_hits=3`: Minimum detections before track is confirmed
- `iou_threshold=0.3`: IOU threshold for track association

## 🚙 Vehicle Classes Detected

The system detects these vehicle types (COCO dataset IDs):
- **Car** (ID: 2)
- **Motorcycle** (ID: 3)
- **Bus** (ID: 5)
- **Truck** (ID: 7)

## 🎨 Visualization Features

- **Bounding Boxes**: Color-coded rectangles around tracked vehicles
- **Track IDs**: Unique identifier for each vehicle
- **Trajectories**: Vehicle movement paths (last 20 points)
- **Counting Line**: Yellow horizontal line for vehicle counting
- **Statistics Panel**: Real-time frame count, track count, and total vehicles

## ⚙️ Configuration

### Model Selection
- **Speed**: Use `yolov8n.pt` (current)
- **Accuracy**: Use `yolov8s.pt` or `yolov8m.pt`

### Confidence Threshold
- **Current**: 0.5 (50%)
- **Adjust**: Modify `conf > 0.5` in the code

### SORT Parameters
- **max_age**: How long to keep tracks without detection
- **min_hits**: Minimum detections to confirm a track
- **iou_threshold**: Matching threshold for track association

## 📈 Performance

### Expected Performance
- **FPS**: 15-30 FPS depending on hardware
- **Accuracy**: High tracking accuracy with minimal ID switches
- **Memory**: Efficient memory usage

### Optimization Tips
1. **Lower Resolution**: Reduce video resolution for faster processing
2. **Model Selection**: Use smaller YOLO models for speed
3. **Confidence**: Adjust confidence threshold based on your needs
4. **SORT Parameters**: Tune parameters for your specific use case

## 🔧 Troubleshooting

### Common Issues

1. **Import Error for SORT**:
   ```bash
   pip install filterpy scikit-image lap
   ```

2. **CUDA/GPU Issues**:
   - Install PyTorch with CUDA support
   - Or use CPU-only version

3. **Video Codec Issues**:
   - Ensure OpenCV is properly installed
   - Try different video codecs (mp4v, avc1, etc.)

4. **Memory Issues**:
   - Reduce video resolution
   - Process shorter video segments

### Debug Mode
Enable debug output by modifying the tracking scripts to print more information about detections and tracks.

## 📝 Customization

### Adding New Vehicle Classes
1. Update `vehicle_classes` dictionary in the script
2. Add corresponding COCO dataset class IDs
3. Update visualization colors if needed

### Modifying Tracking Parameters
1. Adjust SORT parameters for your specific use case
2. Change confidence thresholds based on detection quality
3. Modify counting line position for different scenarios

### Output Customization
1. Modify visualization colors and styles
2. Add additional statistics or metrics
3. Change output video format or quality

## 📄 License

This project uses the SORT algorithm which is licensed under the GNU General Public License v3.0.

## 🙏 Acknowledgments

- **SORT algorithm** by Alex Bewley
- **YOLOv8** by Ultralytics
- **OpenCV** for computer vision operations

## 🆘 Support

If you encounter any issues:
1. Check the troubleshooting section above
2. Verify all dependencies are installed
3. Ensure your video file is in the correct format and location
4. Check the console output for error messages

---

**Happy Vehicle Tracking! 🚗📹**
=======
# IISER-PROJECT
>>>>>>> 6469a7085d54c2defd2c387d7d9bdc1d48e23d30
