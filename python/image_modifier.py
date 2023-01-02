import numpy as np
import numpy.typing as npt
from config import IMG_SIZE
from skimage.filters import gaussian
from skimage.transform import rescale, resize, rotate


def random_mutate_color(img: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Randomly mutate the pixels of the image

    Args:
        img (npt.NDArray[np.float64]): rgb image (0 < pixel < 1)

    Returns:
        npt.NDArray[np.float64]: mutated image (0 < pixel <255)
    """
    if img.max() > 1 and img.min() < 0:
        raise ValueError("Pixel should be between 0 and 1")
    rgb_masked = 1 + (np.random.random(img.shape) - 0.5) / 50
    random_scale_mutation = 1 + (np.random.random() - 0.5) / 5
    masked_img = img * rgb_masked
    masked_img = masked_img * random_scale_mutation * 255
    return np.array(np.clip(masked_img, 0, 255), dtype=np.int64) / 255


def random_crop(img: npt.NDArray[np.float64]) -> npt.ArrayLike:
    """
    Randomly croping the image

    Args:
        img (npt.NDArray[np.float64]): input image

    Returns:
        npt.NDArray[np.float64]: cropped image
    """
    # max 20% shift
    if img.max() > 1:
        img = img / 255
    scale = 1 + np.random.random() / 5
    offset_x = int(np.floor(np.random.random() * img.shape[1] * (scale - 1)))
    offset_y = int(np.floor(np.random.random() * img.shape[0] * (scale - 1)))
    return rescale(img, scale, channel_axis=2)[offset_y : img.shape[0], offset_x : img.shape[1]]


def random_rotate(img: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Randomly rotate the image

    Args:
        img (npt.NDArray[np.float64]): input image

    Returns:
        npt.NDArray[np.float64]: rotated image
    """
    angle = (np.random.random() - 0.5) * 20
    return rotate(img, angle=angle)


def random_blur(img: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Gausian filter blurring with random magnitude

    Args:
        img (npt.NDArray[np.float64]): input image

    Returns:
        npt.NDArray[np.float64]: blurred image
    """
    sigma = np.random.random() / 2
    return gaussian(img, sigma=sigma)


def unify_image(img: npt.ArrayLike) -> npt.ArrayLike:
    """
    Resize image to a given size

    Args:
        img (npt.NDArray[np.float64]): input image

    Returns:
        npt.NDArray[np.float64]: resized image
    """
    return resize(img, output_shape=IMG_SIZE, anti_aliasing=True)


def mutate_image(img: npt.NDArray[np.float64]) -> npt.ArrayLike:
    """
    Image mutation pipeline

    Args:
        img (npt.NDArray[np.float64]): input image

    Returns:
        npt.NDArray[np.float64]: modified image
    """
    mod_img = random_rotate(img)
    mod_img = random_mutate_color(mod_img)
    if np.random.binomial(1, p=0.5) == 1:
        mod_img = random_blur(mod_img)
    mod_img = random_crop(mod_img)
    mod_img = unify_image(mod_img)
    return mod_img
