import numpy as np
import torch
import open3d
from PIL import Image
import time
import matplotlib.pyplot as plt
from load_models import load_sam_model, load_clip_model, load_da_model
from visualization import visualize_segment_scores, visualize_traversable_segments
from trav_map import TraversabilityMap

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

def pipeline() -> None:

    # Future TODO: Load image from some kind of buffer maybe?
    image = Image.open("./Rellis_3D_image_example/pylon_camera_node/frame000002-1581624652_949.jpg")
    camera_intrinsics = [2813.643275, 2808.326079, 969.285772, 624.049972]

       
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

    # Depth time :)
    start_processing_time = time.perf_counter()

    depth = get_image_depth(image, da_model, da_processor, da_device)

    pcd, pixel_indices = image_to_world(image, depth, camera_intrinsics)

    plane = get_ground_plane(pcd)
        
    ground_indices = gather_ground_indices(pcd, plane)

    end_da_processing_time = time.perf_counter()

    logger.info(f"DepthAnything processing time: {(end_da_processing_time-start_processing_time):.4f}s")

    # SAM time :D
    segmentations = generate_sam_masks(image, sam_model, sam_processor, sam_device)

    end_SAM_processing_time = time.perf_counter()
    logger.info(f"SAM processing time: {(end_SAM_processing_time-end_da_processing_time):.4f}s")

    trav_pcd: list[open3d.geometry.PointCloud] = list()
    for segment in segmentations:
        mask = segment["mask"]
    
        segment["trav_score"] = 100 * score_with_clip(image, mask, clip_model, clip_processor, clip_device, safety_embeddings, trav_embeddings)
        #visualize_segment_scores([segment], image)
    
        if segment["trav_score"] is not None and segment["trav_score"] > 0.01:
            # Sample the image mask at each ground point's source pixel, then
            # retain the corresponding indices into the original cloud.
            keep = mask.ravel()[pixel_indices[ground_indices]]
            trav_indices = ground_indices[keep]
            trav_pcd.append(pcd.select_by_index(trav_indices))


    end_CLIP_processing_time = time.perf_counter()
    logger.info(f"CLIP processing time: {(end_CLIP_processing_time-end_SAM_processing_time):.4f}s (avg {(end_CLIP_processing_time-end_SAM_processing_time)/len(segmentations):.4f}s per segment)")
    
    logger.info(f"Total processing time: {(end_CLIP_processing_time-start_processing_time):.4f}s")

    #logger.info("Visualizing results...")
    visualize_segment_scores(segmentations, image)
    visualize_traversable_segments(segmentations, image)
    """
    open3d.visualization.draw_geometries(
        [pcd, pcd.select_by_index[ground_indices].paint_uniform_color([1,0,0])],
        #zoom=0.3412,
        #front=[0.4257, -0.2125, -0.8795],
        #lookat=[2.6172, 2.0475, 1.532],
        #up=[-0.0694, -0.9768, 0.2024]
    )
    """
    trav_pcd.insert(0,pcd.paint_uniform_color([1,0,0]))
    open3d.visualization.draw_geometries(trav_pcd)


if __name__=="__main__":
    pipeline()