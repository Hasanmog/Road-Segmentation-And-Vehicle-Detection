from pathlib import Path
import torch

# Make sure you have yolov5 repo cloned or installed with requirements
# This loads a PyTorch model directly, not through hub
model = torch.load('yolov5s.pt', map_location='cpu')['model'].float().fuse()

# Now you can access the Detect head
detection_head = model.model[-1]  # This is the Detect() module

detection_head
