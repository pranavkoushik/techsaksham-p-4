# Human Pose Detection

A real-time human pose estimation application that uses MediaPipe for detecting and tracking body landmarks. This application provides both webcam-based real-time detection and static image analysis through a user-friendly Streamlit interface.

## Project Structure
```
human_pose_detection/
├── Dataset/                    # Sample video datasets
├── streamlit_pose_detection.py # Main Streamlit application
├── webcam_pose_detection.py    # Standalone webcam detection script
├── static_pose_detection.py    # Standalone image detection script
└── requirements.txt           # Project dependencies
```

## Setup and Installation

1. Clone the repository:
```bash
git clone [[Your Repository URL]](https://github.com/pranavkoushik/techsaksham-p-4).git
cd human_pose_detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit application:
```bash
streamlit run streamlit_pose_detection.py
```

## Features
- Real-time pose detection using webcam
- Static image pose detection with file upload
- User-friendly Streamlit interface
- Support for both image and video processing

## Usage
1. Start the Streamlit app
2. Choose between:
   - Upload Image: Process static images
   - Use Webcam: Real-time pose detection
3. For webcam mode, press 'Q' to exit

## Technical Details

### How it works
MediaPipe uses TensorFlow lite in the backend. The process involves:
1. Person detection to locate the Region of Interest (ROI)
2. Landmark prediction within the ROI
3. 3D coordinate estimation for each landmark

The MediaPipe pose estimator detects 33 key points and provides:
- X, Y coordinates for 2D positioning
- Z coordinate for depth information
- Background segmentation mask

### Key Features of MediaPipe Pose
- High-fidelity body pose tracking
- 33 3D landmarks detection
- Background segmentation
- Real-time performance on various devices
- Cross-platform compatibility (mobile, desktop, web)

## Examples

### Graphical Skeleton Example
![image](https://user-images.githubusercontent.com/48796009/228968225-7509e39c-9d41-42f5-aed9-3387ad9eaa17.png)

### Pose Landmarks
![image](https://user-images.githubusercontent.com/48796009/228968898-73de4945-1957-4656-a17a-c4180c49dbe7.png)

### Quality Evaluation
![image](https://user-images.githubusercontent.com/48796009/228968792-c3da1cd4-7b18-4d57-ab2c-482825deccd6.png)

## Requirements
- Python 3.10 or later
- Webcam (for real-time detection)
- Required Python packages are listed in requirements.txt
