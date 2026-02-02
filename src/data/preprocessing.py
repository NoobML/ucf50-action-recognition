"""
Video preprocessing and frame extraction utilities
"""

import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


def extract_frames_uniform(video_path, num_frames=32):
    """
    Extract uniformly sampled frames from video

    Args:
        video_path (str): Path to video file
        num_frames (int): Number of frames to extract

    Returns:
        list: List of PIL Images (RGB format)
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Warning: Cannot open video {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Handle videos with fewer frames than requested
    if total_frames < num_frames:
        frame_indices = list(range(total_frames)) + [total_frames - 1] * (num_frames - total_frames)
    else:
        # Uniformly sample frames across the video
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if ret:
            # Convert BGR (OpenCV) to RGB (PIL/PyTorch)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)
        else:
            # If frame read fails, use last successful frame
            if frames:
                frames.append(frames[-1])

    cap.release()

    # Ensure we have exactly num_frames
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else Image.new('RGB', (224, 224)))

    return frames[:num_frames]


def get_transforms(train=True, img_size=224):
    """
    Get data transforms for training or validation

    Args:
        train (bool): If True, return training transforms with augmentation
        img_size (int): Target image size

    Returns:
        torchvision.transforms.Compose: Transform pipeline
    """
    if train:
        # Training transforms with augmentation
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        # Validation/test transforms (no augmentation)
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    return transform