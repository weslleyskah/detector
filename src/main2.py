from __future__ import annotations
import argparse
from typing import Any
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# Paths for models and images
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models"
IMAGE_PATH = PROJECT_ROOT / "data" / "img"

# Create required directories if they don't exist
MODEL_PATH.mkdir(parents=True, exist_ok=True)
(IMAGE_PATH / "input").mkdir(parents=True, exist_ok=True)
(IMAGE_PATH / "output").mkdir(parents=True, exist_ok=True)


def draw_bounding_box(img, label, confidence, x1, y1, x2, y2, color):
    text = f"{label} ({confidence:.2f})"
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, text, (x1 - 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Inference function that processes images and returns detected objects with boxes and scores
def main(model_path: str, input_image: str) -> list[dict[str, Any]]:
    # Load YOLO26 model — auto-downloads yolo26n.pt if not present
    model = YOLO(model_path)

    colors = np.random.uniform(0, 255, size=(len(model.names), 3))

    # Run inference
    results = model(input_image, verbose=True)

    original_image = cv2.imread(input_image)
    detections = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detection = {
                "class_id": class_id,
                "class_name": model.names[class_id],
                "confidence": confidence,
                "box": [x1, y1, x2 - x1, y2 - y1],  # x, y, w, h
            }
            detections.append(detection)
            draw_bounding_box(
                original_image,
                model.names[class_id],
                confidence,
                x1, y1, x2, y2,
                colors[class_id],
            )

    output_filename = Path(input_image).stem + "_output" + Path(input_image).suffix
    cv2.imwrite(str(IMAGE_PATH / "output" / output_filename), original_image)

    return detections

# Train the model on a custom dataset (coco8.yaml) for 100 epochs with 640x640 images
def train(model_path: str, data: str = "coco8.yaml", epochs: int = 100, imgsz: int = 640):
    model = YOLO(model_path)
    return model.train(data=data, epochs=epochs, imgsz=imgsz)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo26n.pt")  # Run with yolo26n.pt by default, auto-downloads if not present
    parser.add_argument("--img", default=None)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--data", default="coco8.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    model_arg = MODEL_PATH / args.model

    if args.train:
        train(model_arg, data=args.data, epochs=args.epochs)
    elif args.img:
        main(model_arg, args.img)
    else:
        # Run inference on all images in data/img/input
        for img in (IMAGE_PATH / "input").glob("*.*"): 
            main(model_arg, str(img))