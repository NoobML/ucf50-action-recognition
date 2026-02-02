"""
Training utilities and functions
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train model for one epoch

    Args:
        model: PyTorch model
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: torch device

    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc='Training', leave=False)

    for features, labels in progress_bar:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(features)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })

    avg_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """
    Validate model

    Args:
        model: PyTorch model
        dataloader: Validation data loader
        criterion: Loss function
        device: torch device

    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc='Validation', leave=False):
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def train_model(model, model_name, train_loader, val_loader, device,
                num_epochs=20, learning_rate=0.001, weight_decay=1e-5,
                scheduler_step_size=5, scheduler_gamma=0.5, save_checkpoints=True):
    """
    Complete training loop for a model

    Args:
        model: PyTorch model
        model_name (str): Name of the model (for saving)
        train_loader: Training data loader
        val_loader: Validation data loader
        device: torch device
        num_epochs (int): Number of training epochs
        learning_rate (float): Initial learning rate
        weight_decay (float): L2 regularization weight
        scheduler_step_size (int): Step size for learning rate scheduler
        scheduler_gamma (float): Multiplicative factor for learning rate decay
        save_checkpoints (bool): Whether to save model checkpoints

    Returns:
        tuple: (history dict, trained model)
    """
    print("\n" + "=" * 70)
    print(f"TRAINING: {model_name}")
    print("=" * 70)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    history = {
        'train_losses': [],
        'val_losses': [],
        'train_accs': [],
        'val_accs': [],
        'learning_rates': []
    }

    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 70)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        history['train_losses'].append(train_loss)
        history['val_losses'].append(val_loss)
        history['train_accs'].append(train_acc)
        history['val_accs'].append(val_acc)
        history['learning_rates'].append(current_lr)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {current_lr:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if save_checkpoints:
                import os
                checkpoint_dir = 'models'
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}_best.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                }, checkpoint_path)
            print(f"✓ New best validation accuracy: {val_acc:.2f}%")

    total_time = time.time() - start_time
    history['training_time'] = total_time
    history['best_val_acc'] = best_val_acc

    print("\n" + "=" * 70)
    print(f"TRAINING COMPLETE: {model_name}")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Total Training Time: {total_time:.1f} seconds ({total_time / 60:.1f} minutes)")
    print("=" * 70)

    return history, model