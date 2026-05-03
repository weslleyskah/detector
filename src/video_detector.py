import cv2
import argparse
import csv
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
INPUT_DIR = PROJECT_ROOT / "data" / "video" / "input"
OUTPUT_DIR = PROJECT_ROOT / "data" / "video" / "output"
MODEL_DIR = PROJECT_ROOT / "models"
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
CURRENT_RUN_DIR = OUTPUT_DIR / f"run_{RUN_ID}"

def get_time_string(milliseconds):
    ms = int(milliseconds % 1000)
    seconds = int((milliseconds // 1000) % 60)
    minutes = int((milliseconds // (1000 * 60)) % 60)
    hours = int((milliseconds // (1000 * 60 * 60)) % 24)
    return f"{hours:02d}h-{minutes:02d}m-{seconds:02d}s-{ms:03d}ms"

def process_video(model, video_path: Path, save_data: bool, csv_writer: any):
    print(f"\n--- Processing: {video_path.name} ---")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30
    cap.release()
    
    label_dir = CURRENT_RUN_DIR / "labels"
    image_dir = CURRENT_RUN_DIR / "images"
    
    if save_data:
        label_dir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)

    results = model.track(source=str(video_path), stream=True, persist=True)
    
    FRAME_SKIP = 120
    last_saved_frame = -FRAME_SKIP 
    total_saved = 0

    for frame_idx, result in enumerate(results):
        # Save if we have detections AND it's been at least 15 frames since the last save
        if save_data and len(result.boxes) > 0:
            if (frame_idx - last_saved_frame) >= FRAME_SKIP:
                
                timestamp_ms = (frame_idx / fps) * 1000
                time_str = get_time_string(timestamp_ms)
                base_name = f"{video_path.stem}_{time_str}_f{frame_idx}"
                
                try:
                    # Save labels
                    result.save_txt(str(label_dir / f"{base_name}.txt"))
                    
                    # Save Image
                    annotated_frame = result.plot()
                    success = cv2.imwrite(str(image_dir / f"{base_name}.jpg"), annotated_frame)
                    
                    if success:
                        print(f"  [SUCCESS] Saved frame {frame_idx} to {image_dir.name}")
                        csv_writer.writerow([video_path.name, time_str, base_name, RUN_ID])
                        last_saved_frame = frame_idx
                        total_saved += 1
                    else:
                        print(f" OpenCV failed to write image at frame {frame_idx}")
                except Exception as e:
                    print(f" Save failed: {e}")

    print(f"--- Finished {video_path.name}. Total images saved: {total_saved} ---")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    model_path = MODEL_DIR / args.model
    model = YOLO(str(model_path))

    if args.optimize:
        print("Optimizing for Intel CPU...")
        ov_path = model.export(format="openvino")
        model = YOLO(ov_path)

    videos = [p for p in INPUT_DIR.glob("*") if p.suffix.lower() in [".mp4", ".avi", ".mov"]]
    
    # History log
    master_log = OUTPUT_DIR / "master_detection_log.csv"
    file_exists = master_log.exists()

    with open(master_log, mode='a', newline='', encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["Video", "Timestamp", "Filename_Prefix", "Run_ID"])
        
        for v in videos:
            process_video(model, v, args.save, writer)

    print(f"\nFiles saved in: {CURRENT_RUN_DIR}")
    print(f"Master log updated at: {master_log}")

if __name__ == "__main__":
    main()