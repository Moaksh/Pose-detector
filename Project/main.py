from pathlib import Path
import torch
from calc import curl_calc

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

MODEL_PATH = "yolov7-w6-pose.pt"

weights = torch.load(MODEL_PATH, map_location=device)
pose_model = weights['model'].float().eval()

if torch.cuda.is_available():
        pose_model.half().to(device)


print(f"Loaded model {MODEL_PATH}")

curl_calc(pose_model, angle_max=150, angle_min=30, threshold=35)