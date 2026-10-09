from PIL import Image
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from collections.abc import Iterator

def rellis3d_frames(session: int=0, start: int=0, end: int|None=None, stride: int=10, ) -> Iterator[tuple[Image.Image, np.ndarray, str]]:
    """Load frames (one by one) from the Rellis-3D dataset. Select a session (0-4), which frame you want to start from, what stride you want and when to end.
    If no end is provided, all available frames will be gone through.
    """
    if session not in [0,1,2,3,4]: raise ValueError("Must be an integer between 0-4 (inclusive)")
    if stride <= 0: raise ValueError("Must be a value greater than 0")

    frame = start
    T_base2map = np.eye(4)
    while True:
        if end is not None:
            if frame >= end: break
        image_path = sorted(Path(f"./Rellis-3D/{session:05d}/pylon_camera_node/").glob(f"frame{frame:06d}-*.jpg"))
        if not image_path: break
        image = Image.open(image_path[0])
        # Future: add code to also read the TF
        T_base2map = np.eye(4)
        frame += stride

        yield (image, T_base2map, str(image_path[0]))


if __name__=="__main__":
    for  image, T, path in rellis3d_frames(stride=10,stop=200):
        plt.imshow(np.array(image))
        plt.title(f"{path}")
        plt.pause(0.2)