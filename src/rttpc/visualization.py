import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from PIL import Image
import gc


def show_mask(mask: np.ndarray, ax: Axes, random_color: bool = False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    del mask
    gc.collect()

def show_score(point: list[int], score: float, ax: Axes):
    x,y, = point
    ax.text(x,y,f"{score:.2f}")

    
    

def visualize_segment_scores(segments: list[dict], raw_image: Image.Image) -> None:
    plt.imshow(np.array(raw_image))
    ax = plt.gca()
    ax.set_autoscale_on(False)
    for segment in segments:
        show_mask(segment['mask'], ax, random_color=True)
        show_score(segment['point'], segment['trav_score'], ax)
    plt.axis("off")
    plt.show()
    #del mask
    #gc.collect()