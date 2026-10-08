import numpy as np
import torch
import open3d
from PIL import Image
import time
import matplotlib.pyplot as plt
from load_models import load_da_model
from pipeline import get_image_depth, image_to_world

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    force=True
)
logger = logging.getLogger(__name__)

def get_ground_plane(pcd: open3d.geometry.PointCloud) -> tuple[float]:

    candidates = pcd

    coarse = candidates.voxel_down_sample(voxel_size=0.10)

    if len(coarse.points) < 3:
        raise ValueError("Not enough points to fit a plane")

    # detect_planar_patches may be more robust and also might not limit us to a single ground plane
    plane, inliers = coarse.segment_plane(
        distance_threshold=0.10, # Within 10 cm of plane
        ransac_n=3, # three points define a plane
        num_iterations=500,
        probability=0.999,
    )

    # Plane sanity checks (should be reasonable)
    a, b, c, d = plane
    sus = False
    if abs(a) > 0.5: sus = True # ~ tilt along z axis
    if abs(b) < 0.0: sus = True # ~ tilt along x axis
    if abs(c) > 0.5: sus = True # ~ also tilt along x axis
    if d > 0: sus = True # ground should not be a the same height as the camera

    if sus: print("RANSAC plane has some HEAVY tilt")

    #plane_points = coarse.select_by_index(inliers)
    #other_points = coarse.select_by_index(inliers, invert=True)

    return plane

def gather_ground_pts(
        pcd: open3d.geometry.PointCloud, 
        plane: tuple[float], 
        distance_threshold: float = 0.2,
    ) -> open3d.geometry.PointCloud:
    
    a,b,c,d = plane
    # Convert point cloud to numpy array
    points = np.asarray(pcd.points)

    # Calculate distance from each point to the plane
    distances = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d)
    indices = np.where(distances <= distance_threshold)[0]

    return pcd.select_by_index(indices)

def main() -> None:

    # Future TODO: Load image from some kind of buffer maybe?
    image = Image.open("./Rellis_3D_image_example/pylon_camera_node/frame000002-1581624652_949.jpg")
    camera_intrinsics = [2813.643275, 2808.326079, 969.285772, 624.049972]

    da_model, da_processor, da_device = load_da_model() # Maybe want to load both indoor and outdoor models and use CLIP to select which one to use

    logger.info(f"All models loaded.")

    # Depth time :)
    start_processing_time = time.time()

    depth = get_image_depth(image, da_model, da_processor, da_device)

    pcd: open3d.geometry.PointCloud = image_to_world(image, depth, camera_intrinsics)

    plane = get_ground_plane(pcd)

    ground_pcd = gather_ground_pts(pcd, plane)

    logger.info(f"Total processing time: {(time.time()-start_processing_time):.4f}s")

    open3d.visualization.draw_geometries([ground_pcd])

if __name__=="__main__":
    main()