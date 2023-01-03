import json
import pickle

import matplotlib
import matplotlib.pyplot as plt
from config import EMOJI_IMG_SHAPE, IMG_SIZE, TRAINING_STATE_OUTPUT_DIR
from model import EmojiClassifier
from skimage import io
from skimage.transform import resize
from skimage.util.shape import view_as_blocks

matplotlib.pyplot.switch_backend("Agg")


with (TRAINING_STATE_OUTPUT_DIR / "params.pkl").open("rb") as pkl:
    loaded_params = pickle.load(pkl)


with (TRAINING_STATE_OUTPUT_DIR / "encoded.json").open("r") as f:
    encoded_emoji = {val["index"]: val["unicode"] for val in json.load(f).values()}

model = EmojiClassifier(n_target=len(encoded_emoji) - 1)


def apply_filter(frame, debug=True):
    resized_y = EMOJI_IMG_SHAPE[0] * IMG_SIZE[0]
    resized_x = EMOJI_IMG_SHAPE[1] * IMG_SIZE[1]

    resized_frame = resize(frame, (resized_y, resized_x, IMG_SIZE[2]))
    if debug:
        plt.imshow(resized_frame)
        plt.savefig("test.png")

    blocks = view_as_blocks(resized_frame, block_shape=IMG_SIZE)
    rows = []
    for row in range(EMOJI_IMG_SHAPE[0]):
        rowwise_prediction = model.apply(loaded_params, blocks[row, :, 0, :]).argmax(axis=1)
        rows.append("".join(map(lambda x: encoded_emoji[x], rowwise_prediction.tolist())))
    return rows


if __name__ == "__main__":
    img = "/Users/wckdouglas/Desktop/test.png"
    frame = io.imread(img)
    processed_string = apply_filter(frame, debug=True)
    print("\n".join(processed_string))
