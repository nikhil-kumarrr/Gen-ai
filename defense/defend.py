import torch
import torch.nn.functional as F
from model.train import DEVICE

def gaussian_denoise(image, sigma=0.1):
    image = image.squeeze()
    while image.dim() < 4:
        image = image.unsqueeze(0)
    image = image.to(DEVICE)
    noise = torch.randn_like(image) * sigma
    return torch.clamp(image + noise, -1, 1)

def median_filter_defense(image):
    image = image.squeeze()
    while image.dim() < 4:
        image = image.unsqueeze(0)
    image = image.to(DEVICE)
    return F.avg_pool2d(image, kernel_size=3, stride=1, padding=1)

def detect_adversarial(model, image, threshold=0.3):
    image = image.squeeze()
    while image.dim() < 4:
        image = image.unsqueeze(0)
    image = image.to(DEVICE)

    with torch.no_grad():
        original_score  = model(image).item()
        smoothed_image  = median_filter_defense(image)
        smoothed_score  = model(smoothed_image).item()
        confidence_drop = abs(original_score - smoothed_score)

    return {
        "is_adversarial":  confidence_drop > threshold,
        "original_score":  round(original_score,  4),
        "smoothed_score":  round(smoothed_score,  4),
        "confidence_drop": round(confidence_drop, 4),
        "threshold":       threshold
    }