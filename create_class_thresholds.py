import torch

class_thresholds = {
    "AMD": 0.088,
    "CNV": 0.140,
    "CSR": 0.0702,
    "DME": 0.119,
    "DR": 0.069,
    "DRUSEN": 0.114,
    "MH": 0.084,
    "NORMAL": 0.104
}

torch.save(class_thresholds, "class_thresholds.pt")
print("✅ class_thresholds.pt saved correctly")
