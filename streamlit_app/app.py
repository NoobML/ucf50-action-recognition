"""
Streamlit Web Application for UCF-50 Action Recognition
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# Page config
st.set_page_config(
    page_title="UCF-50 Action Recognition",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL DEFINITIONS
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerModel(nn.Module):
    def __init__(self, input_dim=2048, d_model=512, nhead=8, num_layers=4,
                 num_classes=50, dropout=0.3):
        super(TransformerModel, self).__init__()

        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = torch.mean(x, dim=1)
        x = self.dropout(x)
        out = self.fc(x)
        return out


# ============================================================
# UCF-50 CLASSES
# ============================================================

CLASS_NAMES = [
    'BaseballPitch', 'Basketball', 'BenchPress', 'Biking', 'Billiards',
    'BreastStroke', 'CleanAndJerk', 'Diving', 'Drumming', 'Fencing',
    'GolfSwing', 'HighJump', 'HorseRace', 'HorseRiding', 'HulaHoop',
    'JavelinThrow', 'JugglingBalls', 'JumpRope', 'JumpingJack', 'Kayaking',
    'Lunges', 'MilitaryParade', 'Mixing', 'Nunchucks', 'PizzaTossing',
    'PlayingGuitar', 'PlayingPiano', 'PlayingTabla', 'PlayingViolin', 'PoleVault',
    'PommelHorse', 'PullUps', 'Punch', 'PushUps', 'RockClimbingIndoor',
    'RopeClimbing', 'Rowing', 'SalsaSpin', 'SkateBoarding', 'Skiing',
    'Skijet', 'SoccerJuggling', 'Swing', 'TaiChi', 'TennisSwing',
    'ThrowDiscus', 'TrampolineJumping', 'VolleyballSpiking', 'WalkingWithDog', 'YoYo'
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

@st.cache_resource
def load_models():
    """Load feature extractor and action recognition model"""
    # Feature extractor
    resnet = models.resnet50(pretrained=True)
    feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
    feature_extractor.eval()

    # Action recognition model
    model = TransformerModel(
        input_dim=2048,
        d_model=512,
        nhead=8,
        num_layers=4,
        num_classes=50,
        dropout=0.3
    )

    # Load checkpoint if available
    if os.path.exists('best_model.pth'):
        try:
            checkpoint = torch.load('best_model.pth', map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            st.success("✓ Trained model loaded successfully!")
        except Exception as e:
            st.warning(f"⚠️ Could not load trained weights: {str(e)}. Using random initialization.")
    else:
        st.warning("⚠️ No trained model found. Using random initialization. Upload best_model.pth for predictions.")

    model.eval()

    return feature_extractor, model


def extract_frames(video_path, num_frames=32):
    """Extract uniformly sampled frames from video"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < num_frames:
        frame_indices = list(range(total_frames)) + [total_frames - 1] * (num_frames - total_frames)
    else:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))

    cap.release()

    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else Image.new('RGB', (224, 224)))

    return frames[:num_frames]


def preprocess_frames(frames):
    """Preprocess frames for model input"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return torch.stack([transform(frame) for frame in frames])


def predict_action(video_path, feature_extractor, model):
    """Complete prediction pipeline"""
    # Extract frames
    frames = extract_frames(video_path, num_frames=32)

    # Preprocess
    frames_tensor = preprocess_frames(frames)

    # Extract features
    with torch.no_grad():
        features = feature_extractor(frames_tensor)
        features = features.view(features.size(0), -1)
        features = features.unsqueeze(0)  # Add batch dimension

        # Predict
        outputs = model(features)
        probs = F.softmax(outputs, dim=1)
        top5_probs, top5_indices = torch.topk(probs, 5)

    return frames, top5_probs[0].numpy(), top5_indices[0].numpy()


# ============================================================
# STREAMLIT UI
# ============================================================

st.markdown('<p class="main-header">🎬 Video Action Recognition</p>', unsafe_allow_html=True)
st.markdown("### Powered by Transformer + ResNet50 | Trained on UCF-50 Dataset")

# Sidebar
with st.sidebar:
    st.header("📊 Model Info")
    st.info("""
    **Architecture:**
    - Feature Extractor: ResNet50 (pretrained)
    - Sequence Model: Transformer (4 layers, 8 heads)
    - Input: 32 frames per video
    - Classes: 50 human actions

    **Training:**
    - Dataset: UCF-50
    - Videos: ~6,500
    - Framework: PyTorch
    """)

    st.header("🎯 How it works")
    st.write("""
    1. **Extract 32 frames** uniformly from video
    2. **Extract features** using ResNet50
    3. **Model temporal patterns** with Transformer
    4. **Predict action** with confidence scores
    """)

    st.header("📚 Supported Actions")
    with st.expander("View all 50 actions"):
        for i in range(0, 50, 2):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"• {CLASS_NAMES[i]}")
            with col2:
                if i + 1 < 50:
                    st.write(f"• {CLASS_NAMES[i + 1]}")

    st.markdown("---")
    st.markdown("**Made with ❤️ by Mushtaq Saeed**")
    st.markdown("📧 mushtaqsaeed577@gmail.com")

# Main content
st.markdown("---")

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("📤 Upload Your Video")

    uploaded_file = st.file_uploader(
        "Choose a video file (MP4, AVI, MOV)",
        type=['mp4', 'avi', 'mov'],
        help="Upload a video showing a human action"
    )

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name

        st.video(video_path)

        if st.button("🎯 Recognize Action", type="primary", use_container_width=True):
            with st.spinner("🔄 Processing video..."):
                try:
                    feature_extractor, model = load_models()
                    frames, top5_probs, top5_indices = predict_action(video_path, feature_extractor, model)

                    st.session_state['frames'] = frames
                    st.session_state['top5_probs'] = top5_probs
                    st.session_state['top5_indices'] = top5_indices
                    st.session_state['predicted'] = True

                    st.success("✅ Prediction complete!")

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                finally:
                    if os.path.exists(video_path):
                        os.unlink(video_path)

with col2:
    st.subheader("🎯 Prediction Results")

    if 'predicted' in st.session_state and st.session_state['predicted']:
        frames = st.session_state['frames']
        top5_probs = st.session_state['top5_probs']
        top5_indices = st.session_state['top5_indices']

        st.success(f"### 🏆 {CLASS_NAMES[top5_indices[0]]}")
        st.metric("Confidence", f"{top5_probs[0] * 100:.1f}%")

        st.markdown("#### Top 5 Predictions:")
        for i, (prob, idx) in enumerate(zip(top5_probs, top5_indices)):
            with st.container():
                col_rank, col_name, col_prob = st.columns([0.5, 2, 1.5])
                with col_rank:
                    st.markdown(f"**{i + 1}.**")
                with col_name:
                    st.markdown(f"**{CLASS_NAMES[idx]}**")
                with col_prob:
                    st.progress(float(prob), text=f"{prob * 100:.1f}%")

        st.markdown("#### 📸 Sample Frames:")
        cols = st.columns(4)
        for i, col in enumerate(cols):
            idx = i * 8
            if idx < len(frames):
                col.image(frames[idx], use_column_width=True, caption=f"Frame {idx + 1}")

    else:
        st.info("👆 Upload a video and click **'Recognize Action'** to see predictions!")

        st.markdown("### 📹 Need a test video?")
        st.markdown("""
        Try recording a short video of:
        - 🏀 Playing basketball
        - 💪 Doing push-ups
        - 🎸 Playing guitar
        - 🚴 Riding a bike
        - Or any of the 50 supported actions!
        """)

# Footer
st.markdown("---")
st.markdown("**Note:** Upload `best_model.pth` to this directory for trained model predictions!")