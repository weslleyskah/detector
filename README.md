# Ultralytics YOLOv8 Object Detection with OpenCV and ONNX

This example demonstrates how to implement [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/) object detection using [OpenCV](https://opencv.org/) in [Python](https://www.python.org/), leveraging the [ONNX (Open Neural Network Exchange)](https://onnx.ai/) model format for efficient inference.

# Setup

## Clone
```bash
git clone https://github.com/weslleyskah/detector.git
cd detector
```

## Requirements
```bash
uv pip install -r requirements.txt
```

## Model

```bash
uv run yolo export model=yolov8n.pt imgsz=640 format=onnx opset=12
```

## Image
Download a sample image or use a local one.
Save it as ``image.jpg`` inside the``detector`` directory.

## Run
The script will perform object detection on the image using the `yolov8n.onnx` model and display the results.
```bash
uv run main.py --model yolov8n.onnx --img image.png
```