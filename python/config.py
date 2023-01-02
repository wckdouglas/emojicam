import logging
from pathlib import Path

logger = logging.getLogger("Emoji")

IMG_SIZE = (50, 50, 3)
MUTATION_PER_IMAGE = 32
EPOCH = 800
LEARNING_RATE = 0.01
MINIBATCH_SIZE = 30
TRAINING_STATE_OUTPUT_DIR = Path(__name__).absolute().parent / "train_state"
