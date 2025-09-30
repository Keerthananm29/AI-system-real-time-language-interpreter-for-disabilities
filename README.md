# Sign Language Recognition System

A real-time sign language recognition system that supports multiple sign language systems including ASL (American Sign Language), BSL (British Sign Language), ISL (Indian Sign Language), and Word Gestures. The application uses computer vision and deep learning to detect and interpret hand gestures in real-time.

## Features

- **Multi-Language Support**:
  - American Sign Language (ASL)
  - British Sign Language (BSL) - One Hand and Two Hand variants
  - Indian Sign Language (ISL)
  - Word Gestures

- **Real-time Recognition**:
  - Live webcam feed processing
  - Hand landmark detection and tracking
  - Gesture recognition with confidence thresholds
  - Auto-capture and manual capture options

- **User-Friendly Interface**:
  - Modern GUI with live video feed
  - Text output area with character count
  - Common character shortcuts
  - Text-to-speech capability
  - Save recognized text to file

## Requirements

- Python 3.10 or higher
- OpenCV (cv2)
- TensorFlow
- MediaPipe
- tkinter
- pyttsx3
- Pillow (PIL)
- numpy
- joblib

## Installation

1. Clone the repository:
```bash
git https://github.com/Keerthananm29/AI-system-real-time-language-interpreter-for-disabilities.git
cd AI-system-real-time-language-interpreter-for-disabilities
```

2. Install the required packages:
```bash
pip install opencv-python tensorflow mediapipe pyttsx3 Pillow numpy joblib
```

3. Ensure you have the following directory structure:
```
AI-system-real-time-language-interpreter-for-disabilities/
├── main.py
├── model/
│   ├── asl_cnn_model.h5
│   ├── isl_cnn_model.h5
│   ├── bsl_one_hand_model.pkl
│   ├── bsl_two_hand_model.pkl
│   └── word_gesture_model.h5
├── asl/
│   └── asl_alphabet_train/
├── isl/
├── hand_gestures/
└── README.md
```

## Usage

1. Run the application:
```bash
python main.py
```

2. Select your desired sign language from the dropdown menu.

3. Click "Start Camera" to begin capturing.

4. Position your hand in front of the camera:
   - Keep your hand within frame
   - Ensure good lighting
   - Hold gestures steady for better recognition

5. Use the interface options:
   - Adjust confidence threshold
   - Add spaces and punctuation
   - Save or speak recognized text

## Controls

- **Start/Stop Camera**: Toggle webcam feed
- **Clear All**: Clear the text output
- **Save to File**: Save recognized text to a file
- **Speak**: Convert text to speech
- **Space**: Add space between words
- **Auto-append**: Automatically add recognized signs to text
- **Auto-capture**: Automatically capture gestures

## Confidence Threshold

- Adjust the confidence threshold slider to control recognition sensitivity
- Higher values (closer to 1.0) require more confident predictions
- Lower values (closer to 0.1) are more lenient but may increase false positives

## Troubleshooting

1. **Camera Issues**:
   - Ensure your webcam is properly connected
   - Check if other applications are using the camera
   - Try restarting the application

2. **Recognition Issues**:
   - Ensure good lighting conditions
   - Keep hand gestures within frame
   - Adjust confidence threshold
   - Hold gestures steady for better recognition

3. **Model Loading Errors**:
   - Verify all model files are present in the model directory
   - Check file permissions
   - Ensure correct Python and package versions


## Acknowledgments

- MediaPipe for hand landmark detection
- TensorFlow for deep learning capabilities
- The sign language community for dataset contributions