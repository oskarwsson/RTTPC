from PIL import Image
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from collections.abc import Iterator
import csv

def quaternion_to_rotation_matrix(q: tuple[float] | list[float]) -> np.ndarray:
    q0, q1, q2, q3 = q
    return np.array([
        [2*(q0**2 + q1**2) - 1, 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3), 2*(q0**2 + q2**2) - 1, 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 2*(q0**2 + q3**2) - 1]
    ])

def load_TF(path_to_csv: str, frame: int) -> np.ndarray:
    """Load the transform mapping base_link coordinates into odom."""
    with open(path_to_csv, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if int(row["frame"]) != frame:
                continue

            q = np.array(
                [float(row[key]) for key in ("qw", "qx", "qy", "qz")]
            )
            norm = np.linalg.norm(q)
            if not np.isfinite(norm) or norm == 0:
                raise ValueError(f"Invalid quaternion for frame {frame}")

            T_base2map = np.eye(4)
            T_base2map[:3, :3] = quaternion_to_rotation_matrix(q / norm)
            T_base2map[:3, 3] = [
                float(row[key]) for key in ("x", "y", "z")
            ]
            return T_base2map

    raise ValueError(f"Frame {frame} not found in {path_to_csv}")

def rellis3d_frames(session: int=0, start: int=0, end: int|None=None, stride: int=10, ) -> Iterator[tuple[Image.Image, np.ndarray, str]]:
    """Load frames (one by one) from the Rellis-3D dataset. Select a session (0-4), which frame you want to start from, what stride you want and when to end.
    If no end is provided, all available frames will be gone through.
    Unless its transform is already close to identity, the first yielded frame
    defines the map origin and orientation. All poses use that same reference.
    """
    if session not in [0,1,2,3,4]: raise ValueError("Must be an integer between 0-4 (inclusive)")
    if stride <= 0: raise ValueError("Must be a value greater than 0")

    frame = start
    T_odom2map = None
    while True:
        if end is not None:
            if frame >= end: break
        image_path = sorted(Path(f"./Rellis-3D/{session:05d}/pylon_camera_node/").glob(f"frame{frame:06d}-*.jpg"))
        if not image_path: break
        image = Image.open(image_path[0])
        T_base2map = load_TF(f"./Rellis-3D/{session:05d}/tfs.csv", frame)
        if T_odom2map is None:
            T_odom2map = (
                np.eye(4) if np.allclose(T_base2map, np.eye(4))
                else np.linalg.inv(T_base2map)
            )
        T_base2map = T_odom2map @ T_base2map
        frame += stride

        yield (image, T_base2map, str(image_path[0]))


if __name__=="__main__":
    for  image, T, path in rellis3d_frames(stride=10,stop=200):
        plt.imshow(np.array(image))
        plt.title(f"{path}")
        plt.pause(0.2)
