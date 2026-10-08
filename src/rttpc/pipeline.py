import numpy as np
import torch
import open3d
from PIL import Image
import time
import matplotlib.pyplot as plt

from rttpc.load_models import load_sam_model, load_clip_model, load_da_model
from rttpc.visualization import show_traversability, visualize_segment_scores, visualize_traversable_segments
from rttpc.trav_map import TraversabilityMap

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    force=True
)
logger = logging.getLogger(__name__)

@torch.inference_mode()
def generate_sam_masks(
        image: Image.Image,
        sam_model: object,
        sam_processor: object,
        device: str,
        grid_size: int = 6,
        mask_quality_threshold: float = 0.6,
        max_area_ratio: float = 0.4,
        points_per_batch: int = 16,
    ) -> list[dict]:
    """Generate proposals, encoding the image once and decoding prompts in batches.

    Each grid point is an independent foreground prompt. ``points_per_batch``
    controls decoder memory usage without changing the grid coverage.
    Returns dictionaries containing mask, area, point, and confidence.
    """
    if grid_size < 1 or points_per_batch < 1:
        raise ValueError("grid_size and points_per_batch must be positive")

    width, height = image.size
    x_points = np.linspace(width * 0.1, width * 0.9, grid_size)
    y_points = np.linspace(height * 0.1, height * 0.9, grid_size)
    points = [[float(x), float(y)] for x in x_points for y in y_points]
    proposals = []
    processed_masks = []
    logger.info(f"Generating SAM proposals with {grid_size}x{grid_size} grid...")

    # Shape: [one image, independent prompts, one point per prompt, xy].
    inputs = sam_processor(
        images=image,
        input_points=[[ [point] for point in points ]],
        return_tensors="pt",
    )
    image_embeddings = sam_model.get_image_embeddings(
        inputs["pixel_values"].to(device)
    )
    input_points = inputs["input_points"].to(device)

    for start in range(0, len(points), points_per_batch):
        outputs = sam_model(
            image_embeddings=image_embeddings,
            input_points=input_points[:, start:start + points_per_batch],
            multimask_output=True,
            return_dict=True,
        )
        best_scores, best_indices = outputs.iou_scores[0].max(dim=-1)
        keep = best_scores >= mask_quality_threshold
        kept_indices = keep.nonzero(as_tuple=True)[0]
        if kept_indices.numel() == 0:
            continue

        # Select one candidate per accepted prompt before transfer and upsampling.
        selected_masks = outputs.pred_masks[0, kept_indices, best_indices[keep]]
        masks = sam_processor.image_processor.post_process_masks(
            selected_masks[None, :, None].cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )[0][:, 0].numpy().astype(bool)
        scores = best_scores[keep].cpu().tolist()
        indices = kept_indices.cpu().tolist()

        for mask_np, best_score, point_index in zip(masks, scores, indices):
            area = np.count_nonzero(mask_np)
            if area <= 100 or area / mask_np.size > max_area_ratio:
                continue

            is_duplicate = False
            for existing_mask in processed_masks:
                overlap = np.count_nonzero(mask_np & existing_mask)
                union = np.count_nonzero(mask_np | existing_mask)
                if union > 0 and overlap / union > 0.8:
                    is_duplicate = True
                    break

            if not is_duplicate:
                proposals.append({
                    'mask': mask_np,
                    'area': area,
                    'point': points[start + point_index],
                    'confidence': best_score,
                })
                processed_masks.append(mask_np)

    logger.info(f"Generated {len(proposals)} unique segment proposals")
    return proposals

def score_with_clip(
        image: Image.Image, 
        mask: np.ndarray, 
        clip_model: object, 
        clip_processor: object, 
        device: str,
        positive_embeddings: list[torch.Tensor],
        negative_embeddings: list[torch.Tensor],
        padding_ratio: float = 0.05, #0.01
    ) -> float:
    """Extract and score CLIP features from a segmented region."""
    try:
        # Find bounding box of mask
        coords = np.where(mask)
        if len(coords[0]) == 0:
            return None
        min_x, max_x = np.min(coords[0]), np.max(coords[0])
        min_y, max_y = np.min(coords[1]), np.max(coords[1])
        pad_x = int((max_x - min_x) * padding_ratio)
        pad_y = int((max_y - min_y) * padding_ratio)

        pad_min_x = min_x-pad_x if min_x-pad_x > 0 else 0 
        pad_max_x = max_x+pad_x if max_x+pad_x < image.height else image.height-1

        pad_min_y = min_y-pad_y if min_y-pad_y > 0 else 0 
        pad_max_y = max_y+pad_y if max_y+pad_y < image.width else image.width-1

        image_crop = image.crop((pad_min_y, pad_min_x, pad_max_y, pad_max_x)) 
        
        # Process with CLIP
        inputs = clip_processor(images=image_crop, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs)
            if not isinstance(image_features, torch.Tensor):
                image_features = image_features.pooler_output
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            #image_features /= torch.linalg.vector_norm(image_features, ord=2)  # Normalize the feature vector to unit L2 length

        image_features = image_features.cpu().numpy().squeeze()

        positive_score = max(image_features @ t for t in positive_embeddings)
        negative_score = max(image_features @ t for t in negative_embeddings)

        # The strongest score, either positive or negative, is returned. Like, we assume if the strongest negative result is weaker than the
        # strongest positive result, it is less likely to be a hazard
        return positive_score if positive_score - negative_score > 0 else -negative_score
        
    except Exception as e:
        logger.error(f"Failed to extract CLIP features: {e}")
        return None

def get_image_depth(
        image: Image.Image, 
        da_model: object, 
        da_processor: object,
        da_device: object,
    ) -> torch.Tensor:
    # prepare image for the model
    inputs = da_processor(images=image, return_tensors="pt")
    inputs = {k: v.to(da_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = da_model(**inputs)

    # interpolate to original size
    post_processed_output = da_processor.post_process_depth_estimation(
        outputs,
        target_sizes=[(image.height, image.width)],
    )

    predicted_depth = post_processed_output[0]["predicted_depth"]
    #depth = predicted_depth.detach().cpu().numpy()
    #depth = Image.fromarray(depth.astype("uint8"))
    
    return predicted_depth


def image_to_world(
    image: Image.Image,
    depth: torch.Tensor,
    camera_intrinsics: list[float],
) -> tuple[open3d.geometry.PointCloud, np.ndarray]:
    
    width, height = image.size

    color_np = np.ascontiguousarray(image.convert("RGB"), dtype=np.uint8)
    depth_np = np.ascontiguousarray(
        depth.detach().cpu().float().numpy()
    )

    if depth_np.shape != (height, width):
        raise ValueError("Depth must have shape (image.height, image.width)")

    # Construct points and their pixel mapping together so skipped depths cannot
    # shift the correspondence
    pixel_indices = np.flatnonzero(np.isfinite(depth_np) & (depth_np > 0))
    v, u = np.unravel_index(pixel_indices, (height, width))
    z = depth_np.ravel()[pixel_indices].astype(np.float64)
    fx, fy, cx, cy = camera_intrinsics
    points = np.column_stack(((u - cx) * z / fx, (v - cy) * z / fy, z))

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    pcd.colors = open3d.utility.Vector3dVector(
        color_np.reshape(-1, 3)[pixel_indices] / 255.0
    )

    return pcd, pixel_indices

def get_ground_plane(pcd: open3d.geometry.PointCloud) -> tuple[float]:

    candidates = pcd

    coarse = candidates.voxel_down_sample(voxel_size=0.10)

    if len(coarse.points) < 3:
        raise ValueError("Not enough points to fit a plane")

    # detect_planar_patches may be more robust and also might not limit us to a single ground plane
    plane, inliers = coarse.segment_plane(
        distance_threshold=0.20, # Within 10 cm of plane
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

    if sus: logger.warning("RANSAC plane has some HEAVY tilt!")

    #plane_points = coarse.select_by_index(inliers)
    #other_points = coarse.select_by_index(inliers, invert=True)

    return plane

def gather_ground_indices(
        pcd: open3d.geometry.PointCloud, 
        plane: tuple[float], 
        distance_threshold: float = 0.2,
    ) -> np.ndarray:
    
    a,b,c,d = plane
    # Convert point cloud to numpy array
    points = np.asarray(pcd.points)

    # Calculate distance from each point to the plane
    distances = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d)
    indices = np.where(distances <= distance_threshold)[0]

    return indices

def prepare_text_embeddings(
        positive_queries: list[str], 
        negative_queries: list[str], 
        clip_model: object, 
        clip_processor: object, 
        clip_device: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:

    # Prepare negative embeddings
    negative_embeddings = [] 
    for text_query in negative_queries:
        inputs = clip_processor(text=[text_query], return_tensors="pt")
        inputs = {k: v.to(clip_device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_features = clip_model.get_text_features(**inputs)

            if not isinstance(text_features, torch.Tensor):
                text_features = text_features.pooler_output

            # Normalize text features for cosine-similarity queries.
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
        negative_embeddings.append(text_features.cpu().numpy().squeeze())

    # Prepare positive embeddings
    positive_embeddings = []
    for text_query in positive_queries:
        inputs = clip_processor(text=[text_query], return_tensors="pt")
        inputs = {k: v.to(clip_device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_features = clip_model.get_text_features(**inputs)

            if not isinstance(text_features, torch.Tensor):
                text_features = text_features.pooler_output

            # Normalize text features for cosine-similarity queries.
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
        positive_embeddings.append(text_features.cpu().numpy().squeeze())

    return positive_embeddings, negative_embeddings

# https://www.johndcook.com/blog/2025/05/07/quaternions-and-rotation-matrices/
def quaternion_to_rotation_matrix(q: tuple[float] | list[float]) -> np.ndarray:
    q0, q1, q2, q3 = q
    return np.array([
        [2*(q0**2 + q1**2) - 1, 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3), 2*(q0**2 + q2**2) - 1, 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 2*(q0**2 + q3**2) - 1]
    ])

def world_coords_to_map_indices(points_world: np.ndarray, grid: TraversabilityMap) -> tuple[np.ndarray[int]]:
    col = np.floor((points_world[:, 0] - grid.offset_x) / grid.resolution).astype(int)
    row = np.floor((points_world[:, 1] - grid.offset_y) / grid.resolution).astype(int)
    return (row, col)


def pipeline() -> None:

    # Future TODO: Load image from some kind of buffer maybe? (for testing)
    image = Image.open(
        "./Rellis_3D_image_example/pylon_camera_node/frame000002-1581624652_949.jpg"
        #"./Rellis_3D_image_example/pylon_camera_node/frame000000-1581623790_349.jpg"
    )
    camera_intrinsics = [2813.643275, 2808.326079, 969.285772, 624.049972]
    """
    q:
        w: -0.50507811
        x: 0.51206185
        y: 0.49024953
        z: -0.49228464
    t:
        x: -0.13165462
        y: 0.03870398
        z: -0.17253834
    """
    T_cam2lidar = np.eye(4)
    T_cam2lidar[0:3,0:3] = quaternion_to_rotation_matrix([-0.50507811, 0.51206185, 0.49024953, -0.49228464])
    T_cam2lidar[0,3] = -0.13165462
    T_cam2lidar[1,3] = 0.03870398
    T_cam2lidar[2,3] = -0.17253834

    T_lidar2base = np.diag([-1.0, 1.0, -1.0, 1.0])
    T_cam2base = T_lidar2base @ T_cam2lidar
    

       
    positive_queries = [
        "a photo of a flat clear road",
        "a photo of a walkable dirt path",
        "a photo of flat grassy ground",
    ]

    negative_queries = [
        "a photo of the sky",
        "a photo of a wall",
        "a photo of deep water",
        "a photo of a puddle",
        "a photo of a tree",
        "a photo of trees",
        "a photo of a large rock",
        "a photo of a person",
        "a photo of a cliff",
        "a photo of a bush"
    ]

    sam_model, sam_processor, sam_device = load_sam_model()

    clip_model, clip_processor, clip_device = load_clip_model()

    da_model, da_processor, da_device = load_da_model() # Maybe want to load both indoor and outdoor models and use CLIP to select which one to use

    # Pre-process queries for faster scoring (shaves off roughly 1 second of processing time per image)
    safety_embeddings, trav_embeddings = prepare_text_embeddings(positive_queries, negative_queries, clip_model, clip_processor, clip_device)

    logger.info(f"All models loaded.")

    # Initialize map
    trav_map = TraversabilityMap(
        h = 96,
        w = 96,
        offset_x = -48, # 0 at map center
        offset_y = -48, # 0 at map center
        resolution= 0.10,
        decay_time=3.0
    )

    # Depth time :)
    start_processing_time = time.perf_counter()

    depth = get_image_depth(image, da_model, da_processor, da_device)

    pcd, pixel_indices = image_to_world(image, depth, camera_intrinsics)

    # Fit only nearby points in the bottom half of the image to avoid sky.
    ground_fit_min_row = 0.5 * image.height
    ground_fit_max_depth = 20.0  # metres (camera z)
    fit_mask = (
        (pixel_indices // image.width >= ground_fit_min_row)
        & (np.asarray(pcd.points)[:, 2] <= ground_fit_max_depth)
    )
    plane = get_ground_plane(pcd.select_by_index(np.flatnonzero(fit_mask)))
        
    ground_indices = gather_ground_indices(pcd, plane)

    end_da_processing_time = time.perf_counter()

    logger.info(f"DepthAnything processing time: {(end_da_processing_time-start_processing_time):.4f}s")

    # SAM time :D
    segmentations = generate_sam_masks(image, sam_model, sam_processor, sam_device)

    end_SAM_processing_time = time.perf_counter()
    logger.info(f"SAM processing time: {(end_SAM_processing_time-end_da_processing_time):.4f}s")

    viz_pcd = open3d.geometry.PointCloud()
    for segment in segmentations:
        mask = segment["mask"]
    
        segment["trav_score"] = 100 * score_with_clip(image, mask, clip_model, clip_processor, clip_device, safety_embeddings, trav_embeddings)
        #visualize_segment_scores([segment], image)
    
        if segment["trav_score"] is not None and segment["trav_score"] > 0.01:
            # Sample the image mask at each ground point's source pixel, then
            # retain the corresponding indices into the original cloud.
            keep = mask.ravel()[pixel_indices[ground_indices]]
            trav_indices = ground_indices[keep]
            trav_pcd = pcd.select_by_index(trav_indices)
            # Transform trav_pcd into world frame (skipped for now, robot->world will be defined later)
            points_world = trav_pcd.transform(T_cam2base)
            map_indices = world_coords_to_map_indices(np.asarray(points_world.points), trav_map)
            # Right now, we just overwrite cells directly. Realistically, I'd rather add the scores to already existing scores to increase the confidence,
            # since multiple observations should make us more confident. 
            trav_map[map_indices] = segment["trav_score"]

            viz_pcd.points.extend(points_world.points)

    end_CLIP_processing_time = time.perf_counter()
    logger.info(f"CLIP processing time: {(end_CLIP_processing_time-end_SAM_processing_time):.4f}s (avg {(end_CLIP_processing_time-end_SAM_processing_time)/len(segmentations):.4f}s per segment)")
    
    logger.info(f"Total processing time: {(end_CLIP_processing_time-start_processing_time):.4f}s")

    logger.info("Visualizing results...")
    visualize_segment_scores(segmentations, image)
    visualize_traversable_segments(segmentations, image)
    open3d.visualization.draw_geometries([pcd.transform(T_cam2base), viz_pcd.paint_uniform_color([0,1,0])])

    pose = (0,0,0)
    show_traversability(trav_map, pose)

if __name__=="__main__":
    pipeline()
