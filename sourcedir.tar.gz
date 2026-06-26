import base64
import io
import json
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn

def model_fn(model_dir):
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Linear(num_ftrs, 1),
        nn.Sigmoid()
    )
    model_path = os.path.join(model_dir, "model.pth")
    if os.path.exists(model_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        request_data = json.loads(request_body)
        image_b64 = request_data["image"]
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return preprocess(image).unsqueeze(0)
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_data = input_data.to(device)
    with torch.no_grad():
        output = model(input_data)
    return output.item()

def output_fn(prediction, content_type):
    if content_type == "application/json":
        return json.dumps({"probability": prediction}), content_type
    raise ValueError(f"Unsupported content type: {content_type}")
