# VisionGuard: Edge AI Pedestrian Safety System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![YOLO26](https://img.shields.io/badge/YOLO-v26-orange?style=for-the-badge)
![Flask](https://img.shields.io/badge/Backend-Flask-lightgrey?style=for-the-badge&logo=flask)
![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%20%7C%20Jetson%20%7C%20Windows-green?style=for-the-badge)

**VisionGuard** is a state-of-the-art, real-time smart surveillance system designed to enhance pedestrian safety at crosswalks and industrial zones. Leveraging the cutting-edge **YOLO26** object detection architecture, it delivers NMS-free, low-latency performance optimized for edge computing devices.

This project was engineered to solve the "blind spot" problem in urban environments by autonomously detecting pedestrians and traffic signals, triggering visual and audio alerts to prevent accidents.

## 🚀 Key Features

*   **⚡ Next-Gen AI Performance**: Powered by **YOLO26 (Nano)**, featuring an NMS-free architecture for faster inference and lower latency on edge devices.
*   **👁️ Smart Perception**: Simultaneous detection of multiple classes:
    *   Pedestrians (with ROI filtering)
    *   Traffic Lights (State recognition: Red/Green)
*   **🎨 Professional Dashboard**: A completely responsive, modern web interface featuring:
    *   **Glassmorphism UI** with dark mode default.
    *   Real-time multi-camera streaming.
    *   Interactive ROI (Region of Interest) drawing tools.
    *   Live environmental sensor data (Temperature, Humidity, Voltage).
*   **🔊 Autonomous Response**:
    *   **Audio Alerts**: Automated voice warnings ("Caution", "Stop") via hardware relay.
    *   **Visual Signaling**: GPIO-controlled LED Warning Systems.
*   **💾 Intelligent Recording**: Automated loop recording with localized storage and easy playback retrieval.
*   **💻 Cross-Platform**: Runs seamlessly on **Raspberry Pi 4/5**, **NVIDIA Jetson Nano**, and **Windows** (Simulation Mode).

## 🛠️ Technical Stack

- **Core AI**: `Ultralytics YOLO26`, `PyTorch`, `OpenCV`
- **Backend**: `Flask`, `Threading (Concurrent Streams)`
- **Frontend**: `HTML5`, `CSS3 (Variables/Grid)`, `Bootstrap 5`, `JavaScript (ES6)`
- **Hardware Integration**: `RPi.GPIO`, `gpiod`, `Adafruit_DHT`, `I2C (ADS1115)`

## 🔧 Installation

### Prerequisites
- Python 3.9+
- Webcam or USB Camera

### Quick Start

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/AimanRosman/Pedestrian-AI-Guard.git
    cd Pedestrian-AI-Guard
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the System**
    ```bash
    python main.py
    ```
    *The system will automatically download the `yolo26n.pt` model on the first run.*

4.  **Access Dashboard**
    Open your browser and navigate to: `http://localhost:8080`

## 🖥️ Usage

### Drawing Detection Zones
1.  Navigate to the **Settings** or **Dashboard** tab.
2.  Click **"Configure ROI Zones"**.
3.  Select a zone type (Red Light, Green Light, or Pedestrian).
4.  Draw a polygon on the camera feed to define the active detection area.

### System Configuration
*   **AI Confidence**: Adjust the detection sensitivity sliding bar.
*   **Low Power Mode**: Toggle to reduce resolution and FPS for energy saving.
*   **Simulation Mode**: On Windows, the system automatically mocks GPIO calls, allowing full UI testing without hardware.

## 📈 Performance

| Platform | Model | Resolution | FPS (Approx) |
|----------|-------|------------|--------------|
| PC (CPU) | YOLO26n | 640x480 | 30+ |
| RPi 5 | YOLO26n | 640x480 | 15-20 |
| Jetson Nano | YOLO26n | 640x480 | 25+ |

## 🛡️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Built with ❤️ by Aiman Rosman*
