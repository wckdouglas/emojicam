import logging
from pathlib import Path

logger = logging.getLogger("Emoji")

IMG_SIZE = (36, 36, 3)
MUTATION_PER_IMAGE = 32
EPOCH = 800
LEARNING_RATE = 0.01
MINIBATCH_SIZE = 30
BASE_DIR = Path(__name__).absolute().parents[1]
TRAINING_STATE_OUTPUT_DIR = BASE_DIR / "python" / "train_state"
EMOJI_IMG_SHAPE = (50, 72)
