from ultralytics import YOLO
import os

# Paths
current_dir = os.getcwd()

# Load a pretrained pytorch model
model = YOLO("yolo26n.pt")

# Train the model on a dataset
train_results = model.train(
    data="coco8.yaml",  # Path to dataset configuration file
    epochs=100,  # Number of training epochs
    imgsz=640,  # Image size for training
    device="cpu",  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
    project=current_dir, # This saves 'runs' inside your current folder
    name="example_result" # This names the specific subfolder, the results folder
)

# Evaluate the model's performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("image.png")  # Predict on an image
results[0].show()  # Display results

# Export the model to ONNX format for deployment
path = model.export(format="onnx")  # Returns the path to the exported model

print(path)
print(f"Your trained model is at: {train_results.save_dir}/weights/best.pt")
