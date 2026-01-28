import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from collections import defaultdict

# ---------------- SETTINGS ----------------
TRAIN_DIR = "train"
MODEL_PATH = "best_model_resnet_cbam.pth"
SAVE_PATH = "centroids.pt"

class_names = ['AMD', 'CNV', 'CSR', 'DME', 'DR', 'DRUSEN', 'MH', 'NORMAL']
NUM_CLASSES = len(class_names)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- CBAM (EXACT MATCH) ----------------
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(
            self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x))
        )

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x

# ---------------- LOAD MODEL ----------------
def load_model():
    model = models.resnet50(weights=None)

    model.layer1 = nn.Sequential(model.layer1, CBAM(256))
    model.layer2 = nn.Sequential(model.layer2, CBAM(512))
    model.layer3 = nn.Sequential(model.layer3, CBAM(1024))
    model.layer4 = nn.Sequential(model.layer4, CBAM(2048))

    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device)
    )

    model.to(device).eval()
    return model

model = load_model()

# ---------------- FEATURE HOOK ----------------
features = defaultdict(list)
current_label = None

def hook_fn(module, input, output):
    features[current_label].append(input[0].detach().cpu())

model.fc.register_forward_hook(hook_fn)

# ---------------- DATA ----------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=8, shuffle=False)

# ---------------- EXTRACT FEATURES ----------------
with torch.no_grad():
    for images, labels in loader:
        images = images.to(device)
        for lbl in labels:
            current_label = class_names[lbl.item()]
        _ = model(images)

# ---------------- COMPUTE CENTROIDS ----------------
centroids = {}
for cls in class_names:
    centroids[cls] = torch.mean(
        torch.cat(features[cls], dim=0), dim=0
    )

torch.save(centroids, SAVE_PATH)
print("✅ centroids.pt CREATED SUCCESSFULLY")
