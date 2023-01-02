import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import numpy.typing as npt
from config import MUTATION_PER_IMAGE, logger
from image_modifier import mutate_image
from mpire import WorkerPool
from skimage import io

BASE_DIR = Path("../")


@dataclass
class Emoji:
    index: int
    unicode: str


def index_emoji(emoji_set: Set[str]) -> Dict[str, Emoji]:
    emoji_dict = {}
    for i, emoji in enumerate(emoji_set):
        unicode = f"0x${''.join(emoji.split('-'))}"
        emoji_dict[emoji] = Emoji(index=i, unicode=unicode)
    return emoji_dict


def make_mutate_image(
    emoji_dict: Dict[str, Emoji], emoji_label: str, orig_image: npt.NDArray[np.int64]
) -> List[Tuple[Emoji, npt.NDArray[np.int64]]]:
    """
    Given a original image, make variations by introducing mutations in pixels

    Args:
        emoji_dict (Dict[str, Emoji]): emoji label to emoji data
        emoji_label (str): emoji label of the image
        orig_image (npt.NDArray[np.int64]): native emoji image

    Returns:
        List[Tuple[Emoji, npt.NDArray[np.int64]]]: a list of tuples of emoji labels and modified images
    """
    dataset = []
    for _ in range(MUTATION_PER_IMAGE):
        mod_img = mutate_image(orig_image)
        dataset.append((emoji_dict[emoji_label], mod_img))
    return dataset


def make_dataset(n_cpus=4):
    logger.info("Using %i cpus", n_cpus)
    unicode_list = []  # training label
    img_list = []  # training input
    emoji_set = set()  # storing the unique set of emoji
    for img_file in BASE_DIR.glob("emojis/*png"):
        img = io.imread(img_file)
        if len(img.shape) == 3:  # ignore flat images
            img_name_components = img_file.with_suffix("").name.split("_")
            if len(img_name_components) == 2:
                unicode = img_name_components[1]
            else:
                unicode = img_name_components[2]
            emoji_set.add(unicode)
            unicode_list.append(unicode)
            img_list.append(img)
    logger.info("Collected %i unmodified emoji", len(img_list))

    emoji_dict = index_emoji(emoji_set=emoji_set)
    params = []
    for emoji, image in zip(unicode_list, img_list):
        params.append(
            dict(
                emoji_dict=emoji_dict,
                emoji_label=emoji,
                orig_image=image,
            )
        )

    dataset = []
    with WorkerPool(n_jobs=n_cpus) as p:
        for res in p.imap(make_mutate_image, params, progress_bar=True):
            dataset.extend(res)

    return emoji_dict, dataset
