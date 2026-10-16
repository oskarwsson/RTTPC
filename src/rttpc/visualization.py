import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from pathlib import Path
import shutil
import subprocess


def create_gif(frame_dir: str | Path, output_path: str | Path = "pipeline.gif",
               fps: float = 10) -> Path:
    """Encode frame_000000.png, frame_000001.png, ... into a looping GIF.

    Frames must be contiguous, start at zero, and have matching dimensions.
    Requires FFmpeg on PATH. FFmpeg reads frames incrementally from disk,
    without retaining the complete animation in Python memory.
    Existing output is overwritten; source PNGs are preserved.
    """
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    frame_dir = Path(frame_dir).resolve()
    frames = sorted(frame_dir.glob("frame_*.png"))
    if not frames:
        raise ValueError(f"No PNG frames found in {frame_dir}")
    if any(frame.name != f"frame_{index:06d}.png"
           for index, frame in enumerate(frames)):
        raise ValueError("Frames must be numbered consecutively from frame_000000.png")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("GIF creation requires FFmpeg installed and available on PATH")
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
         "-framerate", str(fps), "-start_number", "0",
         "-i", str(frame_dir / "frame_%06d.png"),
         "-loop", "0", "-f", "gif", str(output_path)],
        check=True,
    )
    return output_path

def visualize_processing_times(processing_times: dict[str, list[float]], VLM: str):
    # Show 4 histograms with processing times as X axis and frequency as Y axis
    da: list = processing_times["DA"]
    sam: list = processing_times["SAM"]
    vlm: list = processing_times[VLM]
    total: list = processing_times["total"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, title, times in zip(
        axes.flat, ("DA", "SAM", VLM, "total"), (da, sam, vlm, total)
    ):
        ax.hist(times, bins=30)
        ax.set(title=title, xlabel="Processing time", ylabel="Frequency")
    fig.tight_layout()
    plt.show()


def show_mask(mask: np.ndarray, ax: Axes, random_color: bool = False):
    if random_color:
        color = (*np.random.randint(0, 256, size=3), 128)
    else:
        color = (30, 144, 255, 102)
    h, w = mask.shape[-2:]
    mask_image = np.zeros((h, w, 4), dtype=np.uint8)
    mask_image[mask.reshape(h, w).astype(bool)] = color
    ax.imshow(mask_image)


def _show_segment_overlay(segments: list[dict], raw_image: Image.Image,
                          ax: Axes) -> None:
    """Keep one byte RGBA overlay per panel, regardless of mask count.

    Accumulating opacity preserves the stronger overlap of repeated blue masks.
    Only a float32 opacity plane is needed while composing the overlay.
    """
    if not segments:
        return
    alpha = np.zeros((raw_image.height, raw_image.width), dtype=np.float32)
    for segment in segments:
        mask = segment['mask']
        alpha[mask] = 1.0 - (1.0 - alpha[mask]) * 0.6
    overlay = np.empty((*alpha.shape, 4), dtype=np.uint8)
    overlay[..., :3] = (30, 144, 255)
    overlay[..., 3] = np.rint(alpha * 255).astype(np.uint8)
    ax.imshow(overlay)
    for segment in segments:
        show_score(segment['point'], segment['trav_score'], ax)


def show_score(point: list[int], score: float, ax: Axes):
    x,y, = point
    ax.text(x,y,f"{score:.2f}")


def visualize_traversable_segments(segments: list[dict], raw_image: Image.Image, threshold: float = 0.01, ax: Axes | None = None) -> None:
    standalone = ax is None
    if standalone:
        _, ax = plt.subplots()
    ax.imshow(np.array(raw_image))
    ax.set_autoscale_on(False)
    selected = [segment for segment in segments
                if segment['trav_score'] is not None and segment['trav_score'] > threshold]
    _show_segment_overlay(selected, raw_image, ax)
    ax.axis("off")
    if standalone:
        plt.show()


def visualize_segment_scores(segments: list[dict], raw_image: Image.Image, ax: Axes | None = None) -> None:
    standalone = ax is None
    if standalone:
        _, ax = plt.subplots()
    ax.imshow(np.array(raw_image))
    ax.set_autoscale_on(False)
    selected = [segment for segment in segments if segment['trav_score'] is not None]
    _show_segment_overlay(selected, raw_image, ax)
    ax.axis("off")
    if standalone:
        plt.show()


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
