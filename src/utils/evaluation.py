"""
Evaluation utilities and metrics
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm.notebook import tqdm
from sklearn.metrics import top_k_accuracy_score


def calculate_top_k_accuracy(model, dataloader, device, k=5):
    """
    Calculate Top-K accuracy

    Args:
        model: PyTorch model
        dataloader: Data loader
        device: torch device
        k (int): K for top-k accuracy

    Returns:
        float: Top-K accuracy percentage
    """
    model.eval()

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc=f'Top-{k} Accuracy', leave=False):
            features = features.to(device)

            outputs = model(features)
            probs = F.softmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Calculate top-k accuracy
    top_k_acc = top_k_accuracy_score(
        all_labels,
        all_probs,
        k=k,
        labels=range(all_probs.shape[1])
    )

    return top_k_acc * 100


def evaluate_model(model, dataloader, device):
    """
    Comprehensive model evaluation

    Args:
        model: PyTorch model
        dataloader: Data loader
        device: torch device

    Returns:
        dict: Dictionary with evaluation metrics
    """
    model.eval()

    all_predictions = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc='Evaluating', leave=False):
            features = features.to(device)

            outputs = model(features)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Calculate metrics
    top1_acc = 100 * np.mean(all_predictions == all_labels)
    top5_acc = top_k_accuracy_score(
        all_labels,
        all_probs,
        k=5,
        labels=range(all_probs.shape[1])
    ) * 100

    return {
        'top1_accuracy': top1_acc,
        'top5_accuracy': top5_acc,
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probs
    }