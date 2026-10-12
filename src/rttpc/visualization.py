import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import gc

def visualize_processing_times(processing_times: dict[str, list[float]]):
    # Show 4 histograms with processing times as X axis and frequency as Y axis
    da: list = processing_times["DA"]
    sam: list = processing_times["SAM"]
    clip: list = processing_times["CLIP"]
    total: list = processing_times["total"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, title, times in zip(
        axes.flat, ("DA", "SAM", "CLIP", "total"), (da, sam, clip, total)
    ):
        ax.hist(times, bins=30)
        ax.set(title=title, xlabel="Processing time", ylabel="Frequency")
    fig.tight_layout()
    plt.show()


def show_mask(mask: np.ndarray, ax: Axes, random_color: bool = False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.5])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.4])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    del mask
    gc.collect()


def show_score(point: list[int], score: float, ax: Axes):
    x,y, = point
    ax.text(x,y,f"{score:.2f}")


def visualize_traversable_segments(segments: list[dict], raw_image: Image.Image, threshold: float = 0.01, ax: Axes | None = None) -> None:
    standalone = ax is None
    if standalone:
        _, ax = plt.subplots()
    ax.imshow(np.array(raw_image))
    ax.set_autoscale_on(False)
    for segment in segments:
        if segment['trav_score'] is not None and segment['trav_score'] > threshold:
            show_mask(segment['mask'], ax)
            show_score(segment['point'], segment['trav_score'], ax)
    ax.axis("off")
    if standalone:
        plt.show()
    #del mask
    #gc.collect()


def visualize_segment_scores(segments: list[dict], raw_image: Image.Image, ax: Axes | None = None) -> None:
    standalone = ax is None
    if standalone:
        _, ax = plt.subplots()
    ax.imshow(np.array(raw_image))
    ax.set_autoscale_on(False)
    for segment in segments:
        if segment['trav_score'] is None:
            continue
        show_mask(segment['mask'], ax)
        show_score(segment['point'], segment['trav_score'], ax)
    ax.axis("off")
    if standalone:
        plt.show()
    #del mask
    #gc.collect()


def show_traversability(grid, robot_pose: tuple[float] | list[float] = (0,0,0), ax: Axes | None = None):
    cmap = LinearSegmentedColormap.from_list(
        "traversability", ["#082818", "#63ef88"]
    )
    cmap.set_bad("#909090")

    rows, cols = grid.data.shape
    extent = (
        grid.offset_x,
        grid.offset_x + cols * grid.resolution,
        grid.offset_y,
        grid.offset_y + rows * grid.resolution,
    )

    standalone = ax is None
    if standalone:
        _, ax = plt.subplots()
    fig = ax.figure
    img = ax.imshow(
        np.ma.masked_less(grid.data, 0),
        cmap=cmap,
        vmin=0,
        vmax=100,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        aspect="equal",
    )
    fig.colorbar(img, ax=ax, label="Traversability")
    ax.set(xlabel="x [m]", ylabel="y [m]")

    # Show robot pose
    x,y,yaw = robot_pose
    arrow_length = 4.0  # metres

    ax.annotate(
        "",
        xy=(x + arrow_length * np.cos(yaw),
            y + arrow_length * np.sin(yaw)),
        xytext=(x, y),
        arrowprops=dict(arrowstyle="->", color="red", lw=2),
    )
    ax.plot(x, y, "ro", markersize=8)
    if standalone:
        plt.show()
