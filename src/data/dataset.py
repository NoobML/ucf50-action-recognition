"""
PyTorch Dataset classes for UCF-50
"""

import os
import torch
from torch.utils.data import Dataset
from .preprocessing import extract_frames_uniform


class UCF50VideoDataset(Dataset):
    """
    UCF-50 Video Dataset

    Loads videos and extracts frames on-the-fly

    Args:
        root_dir (str): Root directory containing action class folders
        action_classes (list): List of action class names
        transform: Torchvision transforms to apply to each frame
        num_frames (int): Number of frames to extract per video
    """

    def __init__(self, root_dir, action_classes, transform=None, num_frames=32):
        self.root_dir = root_dir
        self.transform = transform
        self.num_frames = num_frames

        # Create class to index mapping
        self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(action_classes))}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

        # Build dataset: list of (video_path, label) tuples
        self.samples = []
        for action_class in action_classes:
            class_path = os.path.join(root_dir, action_class)
            if not os.path.isdir(class_path):
                continue

            label = self.class_to_idx[action_class]
            video_files = [f for f in os.listdir(class_path)
                           if f.endswith(('.avi', '.mp4'))]

            for video_file in video_files:
                video_path = os.path.join(class_path, video_file)
                self.samples.append((video_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]

        # Extract frames
        frames = extract_frames_uniform(video_path, self.num_frames)

        if frames is None:
            # Fallback: return zero tensor if video can't be read
            frames_tensor = torch.zeros(self.num_frames, 3, 224, 224)
            return frames_tensor, label

        # Apply transforms to each frame
        if self.transform:
            frames_tensor = torch.stack([self.transform(frame) for frame in frames])
        else:
            from torchvision import transforms
            to_tensor = transforms.ToTensor()
            frames_tensor = torch.stack([to_tensor(frame) for frame in frames])

        return frames_tensor, label

    def get_class_name(self, idx):
        """Get class name from index"""
        return self.idx_to_class[idx]


class FeatureDataset(Dataset):
    """
    Dataset that loads pre-extracted features from disk

    Much faster than extracting features on-the-fly during training

    Args:
        features (torch.Tensor): Pre-extracted features of shape (num_videos, num_frames, feature_dim)
        labels (torch.Tensor): Labels of shape (num_videos,)
    """

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]