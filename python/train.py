import json
import logging
import pickle

logging.basicConfig(level=logging.INFO)
import dataclasses

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import optax
import tqdm
from config import (
    EPOCH,
    LEARNING_RATE,
    MINIBATCH_SIZE,
    TRAINING_STATE_OUTPUT_DIR,
    logger,
)
from jax import jit, random, value_and_grad
from jax.nn import one_hot
from make_dataset import make_dataset
from model import EmojiClassifier
from more_itertools import sample


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


def plot_loss(figname: str, losses: npt.ArrayLike):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(losses)
    fig.savefig(figname)


def main():
    emoji_dict, dataset = make_dataset(n_cpus=2)

    encoded_json = TRAINING_STATE_OUTPUT_DIR / "encoded.json"
    with encoded_json.open("w") as j:
        json.dump(emoji_dict, fp=j, cls=EnhancedJSONEncoder, indent=2, ensure_ascii=False)
    logger.info("Written encoder: %s", encoded_json)

    n_classes = max(emoji.index for emoji in emoji_dict.values())

    model = EmojiClassifier(n_target=n_classes)
    optimizer = optax.adam(learning_rate=LEARNING_RATE)
    rng_key = random.PRNGKey(0)
    _, init_X = zip(*sample(dataset, k=MINIBATCH_SIZE))
    params = model.init(rng_key, jnp.array(jnp.array(init_X)))  # Initialization call
    opt_state = optimizer.init(params)
    losses = np.zeros(EPOCH)

    @jit
    def forward_loss(params, x, Y):
        pred_y = model.apply(params, x)
        return sum(optax.softmax_cross_entropy(y1, y2) for y1, y2 in zip(pred_y, Y))

    for i in tqdm.tqdm(range(EPOCH), desc="Training NN"):
        labels, X = zip(*sample(dataset, k=MINIBATCH_SIZE))
        y = [label.index for label in labels]
        Y = one_hot(y, n_classes)
        X = jnp.array(X)
        loss, grads = value_and_grad(forward_loss)(params, X, Y)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        losses[i] = loss
        if (i + 1) % (EPOCH // 5) == 0:
            logger.info("%i iteration: RMSE = %.2f" % (i + 1, loss))

    plot_loss(TRAINING_STATE_OUTPUT_DIR / "losses.png", losses)

    # save trained params
    param_file = TRAINING_STATE_OUTPUT_DIR / "params.pkl"
    with param_file.open("wb") as pkl:
        pickle.dump(params, pkl)
    logger.info("Saved params to: %s", param_file)


if __name__ == "__main__":
    main()
