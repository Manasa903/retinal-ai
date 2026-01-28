from flask import Flask, render_template, request
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import os
from prompt import get_recommendations

# ---------------- APP SETUP ----------------
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- CLASSES ----------------
class_names = ['AMD', 'CNV', 'CSR', 'DME', 'DR', 'DRUSEN', 'MH', 'NORMAL']
NUM_CLASSES = len(class_names)

# ---------------- CBAM MODULES ----------------
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
        torch.load("best_model_resnet_cbam.pth", map_location=device)
    )

    model.to(device)
    model.eval()
    return model

model = load_model()

# ---------------- LOAD OPEN-SET DATA ----------------
class_centroids = torch.load("centroids.pt", map_location=device)
class_thresholds = torch.load("class_thresholds.pt")

# ---------------- FEATURE HOOK ----------------
extracted_features = []

def feature_hook(module, input, output):
    extracted_features.append(input[0].detach())

model.fc.register_forward_hook(feature_hook)

# ---------------- IMAGE TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ---------------- OPEN-SET PREDICTION ----------------
def open_set_predict(image):
    extracted_features.clear()
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)

    feature = extracted_features[-1][0].cpu()

    distances = {}
    for cls in class_centroids:
        d = 1 - F.cosine_similarity(
            feature.unsqueeze(0),
            class_centroids[cls].unsqueeze(0)
        )
        distances[cls] = d.item()

    best_class = min(distances, key=distances.get)
    min_dist = distances[best_class]
    confidence = probs.max().item() * 100

    # -------- OPEN-SET DECISION --------
    if min_dist > class_thresholds[best_class]:
        return "UNKNOWN", confidence, True
    else:
        return best_class, confidence, False

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/detect", methods=["GET", "POST"])
def detect():
    if request.method == "POST":
        file = request.files["image"]
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        image = Image.open(path).convert("RGB")
        pred, conf, is_unknown = open_set_predict(image)

        if not is_unknown:
            result = get_recommendations(pred, conf)
        else:
            result = None

        return render_template(
            "result.html",
            image=path,
            pred=pred,
            conf=conf,
            result=result,
            is_unknown=is_unknown
        )

    return render_template("detect.html")

if __name__ == "__main__":
    app.run()
