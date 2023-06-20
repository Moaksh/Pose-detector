from pathlib import Path
import torch
from calc import shoulderPress, curl_calc

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

MODEL_PATH = "yolov7-w6-pose.pt"

weights = torch.load(MODEL_PATH, map_location=device)
pose_model = weights['model'].float().eval()

if torch.cuda.is_available():
        pose_model.half().to(device)


print(f"Loaded model {MODEL_PATH}")

# shoulderPress(pose_model, angle_max=135, angle_min=70, threshold=85)
curl_calc(pose_model)
