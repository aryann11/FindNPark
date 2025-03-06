# 🚗 Smart Parking Space Detector

## 📌 Overview

**You Only Look Onc**This project is an **AI-powered Smart Parking Space Detector** that utilizes **YOLOv8 (e)** for real-time object detection to identify empty parking spots. The system processes video footage of a parking lot, detects parked cars, and highlights available spaces dynamically.

## 🏆 Features

- **Real-time Car Detection**: Uses YOLOv8 to accurately identify cars in a parking lot.
- **Empty Parking Space Identification**: Calculates available parking spots based on detected vehicles.
- **Polygonal ROI Definition**: Ensures detection happens only within the designated parking area.
- **Intersection over Union (IoU) Filtering**: Ensures detected spaces are genuinely unoccupied.
- **Frame Skipping for Optimization**: Processes every third frame for better efficiency.
- **User-Friendly Visualization**: Highlights cars in **red** and available spaces in **green**.

## 🔧 Installation

### 1️⃣ Prerequisites

Ensure you have **Python 3.8+** installed along with the following dependencies:

```sh
pip install ultralytics opencv-python numpy
```

### 2️⃣ Clone the Repository

```sh
git clone https://github.com/your-repo/smart-parking-detector.git
cd smart-parking-detector
```

### 3️⃣ Download YOLO Weights

Ensure you have the **YOLOv8 model**:

```sh
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
```

### 4️⃣ Prepare Class List

Make sure your `coco.txt` file contains the **COCO dataset class labels** (car, person, etc.).

### 5️⃣ Run the Detector

```sh
python parking_detector.py
```

## 🏗 How It Works

1. **Loads YOLOv8 Model**: The pre-trained model detects objects in the video.
2. **Defines ROI (Region of Interest)**: Only detects cars within the parking lot boundary.
3. **Processes Each Frame**:
   - Detects cars and highlights them in red.
   - Estimates possible parking spaces to the **left and right** of detected cars.
   - Uses **IoU** to check if these spaces are occupied.
   - Highlights empty spaces in green.
4. **Displays Video Feed**: Shows the parking lot with real-time updates.

## 📂 Project Structure

```
📂 smart-parking-detector
 ├── parking_detector.py      # Main script for detection
 ├── coco.txt                 # COCO dataset class labels
 ├── yolov8s.pt               # Pre-trained YOLO model
 ├── sample_video.mp4         # Sample parking lot video
 ├── README.md                # Project documentation
```

## ⚙️ Configuration

### Modify ROI (Region of Interest)

Adjust the **PARKING\_LOT\_ROI** array in `parking_detector.py` to fit different parking lot layouts:

```python
PARKING_LOT_ROI = np.array([
    [100, 250],
    [900, 250],
    [950, 450],
    [50, 450]
], dtype=np.int32)
```

### Adjust Parking Space Size

Change the parking space dimensions as needed:

```python
PARKING_SPACE_WIDTH = 120
PARKING_SPACE_HEIGHT = 250
```

## 🎯 Future Improvements

- 🏁 **Enhance Accuracy**: Fine-tune YOLOv8 for better detection.
- 📡 **Cloud Integration**: Connect with cloud APIs for live monitoring.
- 📱 **Mobile App**: Develop an app to show real-time parking availability.
- 🎯 **Multi-Camera Support**: Extend the system to manage multiple parking areas.

## 📜 License

This project is open-source under the **MIT License**.

## 🤝 Contributing

Want to enhance this project? Feel free to **fork, improve, and submit pull requests!**

---

**🔗 Connect with me:** [Your GitHub](https://github.com/your-profile) | [Your LinkedIn](https://linkedin.com/in/your-profile) 🚀

