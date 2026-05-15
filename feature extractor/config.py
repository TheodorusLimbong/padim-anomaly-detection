import sys
import os

# Import shared config from src.config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import (
    DATASET_PATH,
    IMAGE_SIZE,
    BATCH_SIZE,
    SELECTED_LAYERS,
    DEVICE,
    FEATURE_OUTPUT_PATH,
)