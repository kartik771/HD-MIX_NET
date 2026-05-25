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

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEED = 42
    NUM_WORKERS = min(4, CPU_COUNT)
    PIN_MEMORY = DEVICE.type == 'cuda'
    NON_BLOCKING = DEVICE.type == 'cuda'
    USE_AMP = DEVICE.type == 'cuda'
    USE_TF32 = DEVICE.type == 'cuda'
    USE_CHANNELS_LAST = DEVICE.type == 'cuda'
    USE_GRAD_CHECKPOINTING = DEVICE.type == 'cuda'
    PREFETCH_FACTOR = 2

    DATA_ROOT = os.environ.get('KVASIR_DATA_ROOT', './Data/Kvasir')
    TRAIN_IMG_DIR = os.path.join(DATA_ROOT, 'images')
    TRAIN_MASK_DIR = os.path.join(DATA_ROOT, 'masks')

    VAL_SPLIT = 0.2
    IMG_SIZE = 384 if HIGH_VRAM_PROFILE else (384 if MID_VRAM_PROFILE else 384)
    BATCH_SIZE = 4 if DEVICE.type == 'cuda' else 1
    ACCUMULATION_STEPS = 4 if DEVICE.type == 'cuda' else 2
    LEARNING_RATE = 5e-4
    MIN_LEARNING_RATE = 5e-7
    NUM_EPOCHS = 200
    WARMUP_EPOCHS = 12
    WEIGHT_DECAY = 2e-4
    GRAD_CLIP_NORM = 0.5

    NUM_CLASSES = 1
    RES2NET_SCALE = 4
    SWIN_WINDOW_SIZE = 7
    CNN_BASE_CHANNELS = 64 if HIGH_VRAM_PROFILE else 56
    SWIN_EMBED_DIM = 128 if HIGH_VRAM_PROFILE else 96
    SWIN_HEADS_STAGE1 = 4 if HIGH_VRAM_PROFILE else 4
    SWIN_HEADS_STAGE2 = 8 if HIGH_VRAM_PROFILE else 8
    SWIN_STAGE_DEPTHS = (3, 3)
    SWIN_MLP_RATIO = 4.0
    SWIN_DROP_PATH = 0.15

    LAMBDA_STRUCT = 1.5
    LAMBDA_DICE = 0.6
    LAMBDA_BCE = 0.3
    LAMBDA_BOUNDARY = 0.75
    LAMBDA_HD = 0.20
    LAMBDA_EDGE = 0.30
    LAMBDA_AUX = 0.50
    STRUCTURE_POOL_KERNEL = 31
    BOUNDARY_LOSS_KERNEL = 3
    MIN_BCE_POS_WEIGHT = 1.0
    MAX_BCE_POS_WEIGHT = 10.0

    DEFAULT_THRESHOLD = 0.48
    THRESHOLD_CANDIDATES = tuple(round(x * 0.01, 2) for x in range(20, 81, 2))
    USE_TTA = True
    VAL_USE_TTA = True
    USE_POST_PROCESSING = True
    POST_PROCESS_KERNEL = 7
    KEEP_LARGEST_COMPONENT = True
    MIN_COMPONENT_AREA_RATIO = 0.0005
    VALIDATE_EVERY = 1
    HD95_EVERY = 1

    INFERENCE_IMG_SIZE = 384
    INFERENCE_BATCH_SIZE = 1
    USE_INFERENCE_TTA = True

    STORE_LAYER_OUTPUTS = False

    USE_ADVANCED_AUGMENTATION = True
    AUGMENTATION_STRENGTH = 0.8
    ENABLE_CUTMIX = True
    ENABLE_MIXUP = True
    MIXUP_ALPHA = 0.3

    USE_EMA = True
    EMA_DECAY = 0.999

    USE_CYCLIC_LR = True
    CYCLE_LENGTH = 10

    LABEL_SMOOTHING = 0.1

    BACKBONE_DROPOUT = 0.3

    USE_DEEPSPEED_PRECISION = False

    ATTENTION_TEMPERATURE = 0.5

    NUM_MODELS_ENSEMBLE = 3

    EARLY_STOPPING_PATIENCE = 30
    EARLY_STOPPING_METRIC = 'iou'

    DICE_SMOOTH = 1e-4

    USE_MULTI_SCALE_LOSS = True
    SCALE_FACTORS = (1, 0.5, 0.25)

