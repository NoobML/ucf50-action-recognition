"""
Configuration file for UCF-50 Action Recognition project
Centralized hyperparameters and settings
"""

import os
import torch


class Config:
    """Global configuration class"""

    # ========== Paths ==========
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'UCF50')
    FEATURES_PATH = os.path.join(PROJECT_ROOT, 'features')
    MODELS_PATH = os.path.join(PROJECT_ROOT, 'models')
    RESULTS_PATH = os.path.join(PROJECT_ROOT, 'results')

    # ========== Video Processing ==========
    NUM_FRAMES = 32  # Frames to extract per video
    IMG_SIZE = 224  # Image size (ResNet input)

    # ========== Feature Extraction ==========
    FEATURE_EXTRACTOR = 'resnet50'
    FEATURE_DIM = 2048  # ResNet50 output dimension
    FREEZE_EXTRACTOR = True

    # ========== Training Hyperparameters ==========
    BATCH_SIZE = 16
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5

    # Scheduler
    SCHEDULER_STEP_SIZE = 5
    SCHEDULER_GAMMA = 0.5

    # ========== Model Architecture ==========
    HIDDEN_DIM = 512
    NUM_LAYERS = 2  # For Stacked LSTM
    NUM_HEADS = 8  # For Transformer
    DROPOUT = 0.3

    # Transformer specific
    TRANSFORMER_D_MODEL = 512
    TRANSFORMER_NUM_LAYERS = 4
    TRANSFORMER_DIM_FEEDFORWARD = 2048

    # ========== Dataset ==========
    NUM_CLASSES = 50
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.2

    # ========== Device ==========
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ========== Reproducibility ==========
    RANDOM_SEED = 42

    # ========== Class Names ==========
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

    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        os.makedirs(cls.FEATURES_PATH, exist_ok=True)
        os.makedirs(cls.MODELS_PATH, exist_ok=True)
        os.makedirs(cls.RESULTS_PATH, exist_ok=True)
        os.makedirs(os.path.join(cls.RESULTS_PATH, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(cls.RESULTS_PATH, 'metrics'), exist_ok=True)

    @classmethod
    def display_config(cls):
        """Display current configuration"""
        print("=" * 70)
        print("CONFIGURATION")
        print("=" * 70)
        print(f"Device: {cls.DEVICE}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Number of Epochs: {cls.NUM_EPOCHS}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Hidden Dimension: {cls.HIDDEN_DIM}")
        print(f"Number of Classes: {cls.NUM_CLASSES}")
        print("=" * 70)