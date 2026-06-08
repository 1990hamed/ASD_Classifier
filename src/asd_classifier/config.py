"""Central configuration constants for the asd_classifier package.

Defines training hyperparameters, filesystem paths, and GA search-space
ranges used across all modules. Import individual names rather than using
``from config import *`` so it stays easy to trace where each constant is used.
"""

from pathlib import Path

# Training hyperparameters
BATCH_SIZE = 64
BASE_LEARNING_RATE = 3e-5
FC_BLOCKS_LEARNING_RATE = 8e-4
LABEL_SMOOTHING = 0.0
WEIGHT_DECAY = 1e-4
NUM_CLASSES = 2
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1
IMAGE_SIZE = 224

# Data paths (relative to project root)
DATA_DIR = Path("data")
IMAGES_DIR = DATA_DIR / "Images"
AUGMENTED_DIR = DATA_DIR / "Augmented_Images"
MERGED_DIR = DATA_DIR / "Merged_Images"
PREPROCESSED_DIR = DATA_DIR / "Preprocessed_Images"
METADATA_DIR = DATA_DIR / "Metadata"
AUGMENTED_METADATA_DIR = METADATA_DIR / "Augmented_Metadata"
MERGED_METADATA_DIR = METADATA_DIR / "Merged_Metadata"

PARTICIPANTS_CSV = METADATA_DIR / "Metadata_Participants.csv"
TC_CSV = METADATA_DIR / "Metadata_TC.csv"
TS_CSV = METADATA_DIR / "Metadata_TS.csv"

# Output paths
BEST_MODEL_DIR = Path("Best_Model")
CHECKPOINT_PATH = Path("checkpoint") / "checkpoint.pkl"
GENETIC_RESULTS_DIR = Path("Genetic_Result")

# GA parameters
GA_POPULATION_SIZE = 30
GA_NGEN = 15
GA_CXPB = 0.7
GA_MUPB = 0.3

control_ranges: dict = {
    "fc_layers": (1, 2),
    "epochs": (10, 100, 10),
}

fc_ranges: dict = {
    "layer_type": (1, 2),
    "num_neurons": lambda: [(1024, 512), (512, 256), (256, 128)],
    "dropout": (10, 50, 10),
}
