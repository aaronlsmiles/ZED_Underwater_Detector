# ZED SDK + YOLO Object Detection with Depth Integration

A comprehensive object detection system that combines YOLO models with ZED camera depth sensing capabilities. This project evolved from basic object detection to a full-featured system with 3D object analysis, dimension measurement, and data export capabilities.

## 🎯 Project Overview

This repository contains multiple versions of an object detection system, progressing from simple confidence-based detection to advanced 3D analysis with the following key features:

- **Real-time object detection** using YOLO models (YOLOv5/YOLOv8)
- **3D depth integration** with ZED stereo cameras
- **Object dimension measurement** (width, height, depth)
- **Multi-point distance sampling** for accurate measurements
- **Real-time data export** to Excel with timestamps
- **Screenshot capture** with visual distance markers
- **Pause/resume functionality** for detailed analysis
- **SVO recording support** for offline processing
- **3D visualization** with OpenGL rendering

## 🛠️ Prerequisites & Setup

### Hardware Requirements
- ZED stereo camera (ZED 2, ZED 2i, ZED X, etc.) - *ZED v1 not supported*
- CUDA-compatible GPU (recommended for real-time performance)

### Software Dependencies

1. **Install ZED SDK**
   ```bash
   # Download and install ZED SDK from https://www.stereolabs.com/developers/release/
   # Follow the installation guide for your platform
   ```

2. **Python Environment Setup**
   ```bash
   # Create virtual environment (recommended)
   python -m venv zed_detection_env
   source zed_detection_env/bin/activate  # Linux/Mac
   # or
   zed_detection_env\Scripts\activate  # Windows
   
   # Install core dependencies
   pip install torch torchvision ultralytics opencv-python pandas numpy
   pip install pyzed  # ZED Python API
   ```

3. **YOLO Model Setup**
   ```bash
   # Download pre-trained models (will be downloaded automatically on first use)
   # Or specify custom trained models using --weights parameter
   ```

## 📁 Project Structure

```
├── detector -conf.py                           # Basic confidence-based detection
├── detector -confDepth -xl.py                 # Added depth integration
├── detector -confDepth -xl2.py                # Enhanced depth processing
├── detector -confDepth -xlBlack.py            # Background handling improvements
├── detector -confDepth -xlBlackss.py          # Screenshot functionality
├── detector -confDepth -xlBlackss2 -bboxDist.py   # Bounding box distance analysis
├── detector -confDepth -xlBlackss2 -objDims.py    # Object dimension measurement
├── detector/pytorch_yolov8/detector.py        # Organized YOLO8 implementation
├── custom detector/                            # Custom trained model support
│   ├── pytorch_yolov5/                       # YOLOv5 integration
│   └── pytorch_yolov8/                       # YOLOv8 integration (most advanced)
└── AI Models/                                 # Pre-trained model storage
```

## 🚀 Usage Guide

### Version 1: Basic Detection (`detector -conf.py`)
*Foundation version with core object detection*

```bash
python "detector -conf.py" --weights yolov8m.pt --conf_thres 0.25
```

**Features:**
- Real-time YOLO object detection
- Confidence filtering
- Basic ZED SDK integration
- Console output of detections

### Version 2: Depth Integration (`detector -confDepth -xl.py`)
*Added 3D depth measurements*

```bash
python "detector -confDepth -xl.py" --weights yolov8m.pt --conf_thres 0.25
```

**New Features:**
- Depth measurement at object center points
- 3D object tracking with ZED SDK
- Distance calculations in real-time

### Version 3: Enhanced Analysis (`detector -confDepth -xlBlackss2 -objDims.py`)
*Most advanced version with comprehensive 3D analysis*

```bash
python "detector -confDepth -xlBlackss2 -objDims.py" --weights yolov8m.pt --conf_thres 0.25
```

**Advanced Features:**
- **Multi-point distance sampling** (upper, center, lower points)
- **Object dimension measurement** (width × height × depth)
- **Screenshot capture** with visual markers (Space bar)
- **Excel data export** with timestamps
- **Pause/resume functionality** (Space bar toggles)
- **Visual distance markers** on screenshots
- **Real-time data logging**

**Controls:**
- `ESC`: Exit and save data
- `SPACE`: Pause/resume detection + capture screenshot
- `Mouse`: Interact with 3D visualization

### SVO Recording Support
*Process pre-recorded ZED files*

```bash
python "detector -confDepth -xlBlackss2 -objDims.py" --weights yolov8m.pt --svo path/to/recording.svo2
```

## 🎛️ Command Line Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--weights` | `yolov8m.pt` | Path to YOLO model file |
| `--img_size` | `640` | Input image size for YOLO |
| `--conf_thres` | `0.25` | Confidence threshold (0.0-1.0) |
| `--iou_thres` | `0.45` | IoU threshold for NMS |
| `--svo` | `None` | Path to SVO file for offline processing |

## 📊 Data Output

The advanced versions automatically generate:

1. **Excel Files** (`detection_data_YYYYMMDD_HHMMSS.xlsx`)
   - Timestamp, Class, Confidence
   - Upper/Center/Lower distances
   - Object dimensions (W×H×D)
   - Screenshot references

2. **Screenshot Files** (`detections/screenshot_DDMMYYYY_HHMMSS.png`)
   - Visual markers showing measurement points
   - Distance annotations
   - Captured at pause moments

3. **Console Logs**
   - Real-time detection information
   - Distance measurements
   - System status updates

## 🔧 Adding New Enhancements

### Pattern for New Features
This project follows a consistent enhancement pattern. To add new features:

1. **Create a new detector version** following the naming convention:
   ```
   detector -confDepth -xlBlackss2 -[YOUR_FEATURE].py
   ```

2. **Extend the base functionality** by adding to existing patterns:
   - **Data export**: Add columns to `detection_data` DataFrame
   - **Visual features**: Extend screenshot functionality
   - **Measurements**: Add new measurement types to object processing
   - **Controls**: Add keyboard shortcuts following existing pattern

### Example: Adding New Measurement Type
```python
# In torch_thread function, add your measurement logic:
def torch_thread(weights, img_size, conf_thres=0.2, iou_thres=0.45):
    # ... existing code ...
    
    # Your new measurement
    custom_measurement = calculate_custom_metric(obj, point_cloud)
    
    # Add to data export
    new_row = pd.DataFrame({
        'Class': [class_id],
        'Name': [class_names[class_id]],
        'Confidence': [f"{confidence:.2%}"],
        # ... existing columns ...
        'Custom_Metric': [str(custom_measurement)],  # New column
        'Timestamp': [current_time]
    })
```

### Example: Adding New Control
```python
# In main loop, add new key handler:
key = cv2.waitKey(10)
if key == 27:  # ESC - Exit
    exit_signal = True
elif key == 32:  # SPACE - Pause/Screenshot
    # ... existing pause logic ...
elif key == ord('s'):  # S - Your new feature
    execute_your_feature()
    print("Custom feature executed!")
```

## 🎯 Best Practices

### Model Selection
- **yolov8n.pt**: Fastest, good for real-time on limited hardware
- **yolov8m.pt**: Balanced speed/accuracy (recommended)
- **yolov8l.pt**: Higher accuracy, requires more GPU memory
- **yolov8x.pt**: Maximum accuracy, slowest

### Performance Optimization
```bash
# For real-time performance
python detector.py --weights yolov8n.pt --img_size 416 --conf_thres 0.3

# For maximum accuracy
python detector.py --weights yolov8x.pt --img_size 832 --conf_thres 0.15
```

### Data Collection Tips
- Use SPACE bar strategically to capture key moments
- Lower confidence thresholds capture more objects (with more false positives)
- Ensure good lighting for optimal depth measurements
- Process SVO files offline for consistent results

## 🐛 Troubleshooting

### Common Issues

**"ZED Camera not detected"**
```bash
# Check camera connection and run ZED diagnostics
ZED_Diagnostic
```

**"CUDA out of memory"**
```bash
# Reduce image size or use smaller model
python detector.py --weights yolov8n.pt --img_size 416
```

**"No module named 'pyzed'"**
```bash
# Ensure ZED SDK and Python API are properly installed
pip install pyzed
```

**"Depth measurements showing 0.0"**
- Check camera calibration
- Ensure objects are within depth range (0.3m - 40m for most ZED cameras)
- Verify proper lighting conditions

## 📈 Future Enhancements

The architecture supports easy addition of:
- **Multi-object tracking** across frames
- **Custom object classes** with specialized measurements
- **Real-time alerts** based on object properties
- **Network streaming** of detection results
- **Database integration** for long-term storage
- **Machine learning** on collected measurement data

## 📝 Contributing

When adding new features, please follow the established patterns:
1. Extend existing versions rather than modifying them
2. Maintain backward compatibility
3. Add comprehensive logging
4. Update this README with new usage patterns
5. Test with both live camera and SVO files

## 📄 License

This project builds upon the ZED SDK samples and YOLO implementations. Please respect the licensing terms of:
- [ZED SDK](https://www.stereolabs.com/developers/release/)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)

---

**Note**: This project is continuously evolving. The most advanced features are currently in the `detector -confDepth -xlBlackss2 -objDims.py` version. Check the `custom detector/pytorch_yolov8/` directory for the latest organized implementations and enhancements.
