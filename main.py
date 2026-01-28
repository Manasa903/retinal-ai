import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from prompt import get_recommendations

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Class Names
# -----------------------------
class_names = ['AMD', 'CNV', 'CSR', 'DME', 'DR', 'DRUSEN', 'MH', 'NORMAL']
NUM_CLASSES = len(class_names)

# -----------------------------
# CBAM DEFINITIONS
# -----------------------------
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

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
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

# -----------------------------
# Load Open-Set Data
# -----------------------------
class_centroids = torch.load("centroids.pt", map_location=device)
class_thresholds = torch.load("class_thresholds.pt")

# -----------------------------
# Feature Hook
# -----------------------------
extracted_features = []

def feature_hook(module, input, output):
    # input[0] = feature vector before FC
    extracted_features.append(input[0].detach())

model.fc.register_forward_hook(feature_hook)

# -----------------------------
# Image Transform
# -----------------------------
infer_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Open-Set Prediction
# -----------------------------
def open_set_predict(image):
    extracted_features.clear()
    image = infer_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)

    # feature vector (2048-D)
    feat = extracted_features[-1][0].cpu()

    # cosine distance to class centroids
    distances = {}
    for cls in class_centroids:
        d = 1 - F.cosine_similarity(
            feat.unsqueeze(0),
            class_centroids[cls].unsqueeze(0)
        )
        distances[cls] = d.item()

    best_cls = min(distances, key=distances.get)
    min_dist = distances[best_cls]
    confidence = probs.max().item() * 100

    # open-set decision
    if min_dist > class_thresholds[best_cls]:
        return "UNKNOWN", confidence, True
    else:
        return best_cls, confidence, False

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title(" OCT Retinal Disease Classification")
st.markdown("**ResNet50 + CBAM | Open-Set Medical OCT AI System**")

uploaded_file = st.file_uploader(
    "Upload OCT Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded OCT Image", use_column_width=True)

    if st.button(" Predict"):
        with st.spinner("Analyzing OCT image..."):
            pred_class, confidence, is_unknown = open_set_predict(image)

        st.markdown("---")

        if not is_unknown:
            st.success(f"🧠 **Predicted Class:** {pred_class}")
            st.info(f"📊 **Confidence:** {confidence:.2f}%")
            result = get_recommendations(pred_class, confidence)
            render_recommendation_ui(result)
        else:
            st.error("❓ **Unknown Retinal Condition**")
            st.warning(
                "The image does not sufficiently match any trained retinal disease class."
            )
            st.info(f"Model confidence: {confidence:.2f}%")

        st.markdown("---")
        st.caption("⚠️ For research & clinical decision support only")
