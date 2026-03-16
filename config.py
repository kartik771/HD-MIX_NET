# config.py
import os
import torch


class Config:
    CPU_COUNT = os.cpu_count() or 4
    IS_COLAB = 'COLAB_GPU' in os.environ
    GPU_NAME = torch.cuda.get_device_name(0).lower() if torch.cuda.is_available() else ''
    GPU_VRAM_GB = (
        torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if torch.cuda.is_available() else 0.0
    )
    HIGH_VRAM_PROFILE = torch.cuda.is_available() and GPU_VRAM_GB >= 14.0
    MID_VRAM_PROFILE = torch.cuda.is_available() and 8.0 <= GPU_VRAM_GB < 14.0

    # System
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEED = 42
    NUM_WORKERS = min(2, CPU_COUNT)
    PIN_MEMORY = DEVICE.type == 'cuda'
    NON_BLOCKING = DEVICE.type == 'cuda'
    USE_AMP = DEVICE.type == 'cuda'
    USE_TF32 = DEVICE.type == 'cuda'
    USE_CHANNELS_LAST = DEVICE.type == 'cuda'
    USE_GRAD_CHECKPOINTING = DEVICE.type == 'cuda'
    PREFETCH_FACTOR = 1

    # Data Paths
    DATA_ROOT = os.environ.get('KVASIR_DATA_ROOT', './Data/Kvasir')
    TRAIN_IMG_DIR = os.path.join(DATA_ROOT, 'images')
    TRAIN_MASK_DIR = os.path.join(DATA_ROOT, 'masks')

    # Training Hyperparams
    VAL_SPLIT = 0.2
    IMG_SIZE = 384 if HIGH_VRAM_PROFILE else (320 if MID_VRAM_PROFILE else 256)
    BATCH_SIZE = 2 if DEVICE.type == 'cuda' else 1
    ACCUMULATION_STEPS = 2 if DEVICE.type == 'cuda' else 1
    LEARNING_RATE = 3e-4
    MIN_LEARNING_RATE = 1e-6
    NUM_EPOCHS = 120
    WARMUP_EPOCHS = 8
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP_NORM = 1.0

    # Model Hyperparams
    NUM_CLASSES = 1
    RES2NET_SCALE = 4
    SWIN_WINDOW_SIZE = 7
    CNN_BASE_CHANNELS = 48 if HIGH_VRAM_PROFILE else 40
    SWIN_EMBED_DIM = 96 if HIGH_VRAM_PROFILE else 72
    SWIN_HEADS_STAGE1 = 4 if HIGH_VRAM_PROFILE else 3
    SWIN_HEADS_STAGE2 = 8 if HIGH_VRAM_PROFILE else 6
    SWIN_STAGE_DEPTHS = (2, 2)
    SWIN_MLP_RATIO = 4.0
    SWIN_DROP_PATH = 0.10

    # Loss Weights
    LAMBDA_STRUCT = 1.0
    LAMBDA_DICE = 0.4
    LAMBDA_BCE = 0.2
    LAMBDA_BOUNDARY = 0.25
    LAMBDA_HD = 0.0
    LAMBDA_EDGE = 0.15
    LAMBDA_AUX = 0.35
    STRUCTURE_POOL_KERNEL = 31
    BOUNDARY_LOSS_KERNEL = 5
    MIN_BCE_POS_WEIGHT = 1.0
    MAX_BCE_POS_WEIGHT = 6.0

    # Validation / Inference
    DEFAULT_THRESHOLD = 0.45
    THRESHOLD_CANDIDATES = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65)
    USE_TTA = True
    VAL_USE_TTA = False
    USE_POST_PROCESSING = True
    POST_PROCESS_KERNEL = 5
    KEEP_LARGEST_COMPONENT = True
    MIN_COMPONENT_AREA_RATIO = 0.001
    VALIDATE_EVERY = 1
    HD95_EVERY = 4
