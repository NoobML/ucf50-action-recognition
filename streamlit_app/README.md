# 🎬 Video Action Recognition - Streamlit App

Interactive web application for recognizing human actions in videos using deep learning.

## 🚀 Features

- Upload videos (MP4, AVI, MOV)
- Real-time action prediction
- Top-5 predictions with confidence scores
- Visual frame extraction display
- 50 supported action classes

## 📦 Installation
```bash
pip install -r requirements.txt
```

## 🎯 Running Locally
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## 📋 Requirements

- Python 3.8+
- PyTorch
- Streamlit
- OpenCV
- Trained model checkpoint: `best_model.pth`

## 🔧 Model Setup

Place your trained model checkpoint as `best_model.pth` in this directory.

The model should be a PyTorch state dict with the key `'model_state_dict'`.

## 🤗 Deployment on HuggingFace

This app is designed to be deployed on HuggingFace Spaces:

1. Create a new Space with **Streamlit** SDK
2. Upload all files from this directory
3. Add your `best_model.pth` (Git LFS for large files)
4. HuggingFace will automatically deploy!

## 📊 Model Architecture

- **Feature Extractor:** ResNet50 (pretrained on ImageNet)
- **Sequence Model:** Transformer (4 layers, 8 attention heads)
- **Input:** 32 uniformly sampled frames
- **Output:** 50 action classes

## 🎓 Supported Actions

The model recognizes 50 different human actions including:
- Sports (Basketball, Baseball, Tennis, etc.)
- Music (Guitar, Piano, Violin, etc.)
- Fitness (Push-ups, Pull-ups, Lunges, etc.)
- Daily activities (Walking, Mixing, etc.)

See full list in the sidebar of the app!

## 📧 Contact

**Mushtaq Saeed**
- Email: mushtaqsaeed577@gmail.com
- GitHub: [@NoobML](https://github.com/NoobML)

## 📜 License

MIT License - See main repository for details.
```

