"""
Visualization utilities for training results
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_training_curves(model_name, history, save_path=None):
    """
    Plot training and validation curves for a single model

    Args:
        model_name (str): Name of the model
        history (dict): Training history dictionary
        save_path (str): Path to save the plot (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['train_losses']) + 1)

    # Loss curves
    axes[0].plot(epochs, history['train_losses'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_losses'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0].set_title(f'{model_name} - Loss Curves', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)

    # Accuracy curves
    axes[1].plot(epochs, history['train_accs'], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, history['val_accs'], 'r-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[1].set_title(f'{model_name} - Accuracy Curves', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")

    plt.show()


def plot_all_models_grid(all_results, save_path=None):
    """
    Plot training curves for all models in a grid

    Args:
        all_results (dict): Dictionary with all model results
        save_path (str): Path to save the plot (optional)
    """
    num_models = len(all_results)
    fig, axes = plt.subplots(num_models, 2, figsize=(16, num_models * 3))

    if num_models == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle('Training Curves - All Models Comparison',
                 fontsize=16, fontweight='bold', y=0.995)

    for idx, (model_name, history) in enumerate(all_results.items()):
        epochs = range(1, len(history['train_losses']) + 1)

        # Loss curves
        axes[idx, 0].plot(epochs, history['train_losses'], 'b-',
                         label='Train', linewidth=2, alpha=0.7)
        axes[idx, 0].plot(epochs, history['val_losses'], 'r-',
                         label='Val', linewidth=2, alpha=0.7)
        axes[idx, 0].set_ylabel('Loss', fontsize=10, fontweight='bold')
        axes[idx, 0].set_title(f'{model_name} - Loss', fontsize=11, fontweight='bold')
        axes[idx, 0].legend(fontsize=8)
        axes[idx, 0].grid(alpha=0.3)

        # Accuracy curves
        axes[idx, 1].plot(epochs, history['train_accs'], 'b-',
                         label='Train', linewidth=2, alpha=0.7)
        axes[idx, 1].plot(epochs, history['val_accs'], 'r-',
                         label='Val', linewidth=2, alpha=0.7)
        axes[idx, 1].set_ylabel('Accuracy (%)', fontsize=10, fontweight='bold')
        axes[idx, 1].set_title(f'{model_name} - Accuracy', fontsize=11, fontweight='bold')
        axes[idx, 1].legend(fontsize=8)
        axes[idx, 1].grid(alpha=0.3)

        if idx == num_models - 1:
            axes[idx, 0].set_xlabel('Epoch', fontsize=10, fontweight='bold')
            axes[idx, 1].set_xlabel('Epoch', fontsize=10, fontweight='bold')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")

    plt.show()


def plot_model_comparison(summary_df, save_path=None):
    """
    Create comprehensive comparison bar charts

    Args:
        summary_df (pd.DataFrame): Summary dataframe with results
        save_path (str): Path to save the plot (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    df_sorted = summary_df.sort_values('Best Val Acc (%)', ascending=True)

    models = df_sorted['Model']
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))

    # 1. Top-1 Accuracy
    axes[0, 0].barh(models, df_sorted['Best Val Acc (%)'],
                   color=colors, edgecolor='black', linewidth=1.5)
    axes[0, 0].set_xlabel('Top-1 Accuracy (%)', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Top-1 Validation Accuracy Comparison',
                        fontsize=14, fontweight='bold')
    axes[0, 0].grid(axis='x', alpha=0.3)
    for i, v in enumerate(df_sorted['Best Val Acc (%)']):
        axes[0, 0].text(v + 0.5, i, f'{v:.2f}%', va='center', fontweight='bold')

    # 2. Top-5 Accuracy
    axes[0, 1].barh(models, df_sorted['Top-5 Acc (%)'],
                   color=colors, edgecolor='black', linewidth=1.5)
    axes[0, 1].set_xlabel('Top-5 Accuracy (%)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Top-5 Validation Accuracy Comparison',
                        fontsize=14, fontweight='bold')
    axes[0, 1].grid(axis='x', alpha=0.3)
    for i, v in enumerate(df_sorted['Top-5 Acc (%)']):
        axes[0, 1].text(v + 0.5, i, f'{v:.2f}%', va='center', fontweight='bold')

    # 3. Training Time
    axes[1, 0].barh(models, df_sorted['Training Time (s)'],
                   color=colors, edgecolor='black', linewidth=1.5)
    axes[1, 0].set_xlabel('Training Time (seconds)', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].grid(axis='x', alpha=0.3)
    for i, v in enumerate(df_sorted['Training Time (s)']):
        axes[1, 0].text(v + 5, i, f'{v:.1f}s', va='center', fontweight='bold')

    # 4. Parameters
    axes[1, 1].barh(models, df_sorted['Parameters (M)'],
                   color=colors, edgecolor='black', linewidth=1.5)
    axes[1, 1].set_xlabel('Parameters (Millions)', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Model Size Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].grid(axis='x', alpha=0.3)
    for i, v in enumerate(df_sorted['Parameters (M)']):
        axes[1, 1].text(v + 0.2, i, f'{v:.2f}M', va='center', fontweight='bold')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")

    plt.show()