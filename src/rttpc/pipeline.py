import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import logging
from load_models import *

logger = logging.getLogger(__name__)

def generate_sam_masks(
        image: Image.Image,
        sam_model: object,
        sam_processor: object,
        device: str,
        grid_size: int = 6,
        mask_quality_threshold: float = 0.5,
    ) -> List[Dict]:
    """
    Generate masks from image using SAM model.

    Args:
        image: image to generate masks for
        sam_model: model
        sam_processor: processor
        device: Device to use (cuda/cpu)
        grid_size: size of grid
        mask_quality_threshold: quality threshold for keeping masks

    Returns:
        Tuple of (model, processor, device)
    """

    """
    Model configurations from HuggingFace
        'base': 'facebook/sam-vit-base',
        'large': 'facebook/sam-vit-large',
        'huge': 'facebook/sam-vit-huge'
    """
    """Generate object proposals using SAM with a grid of point prompts."""
    
    width, height = image.size
    
    # Generate grid of point prompts
    x_points = np.linspace(width * 0.1, width * 0.9, grid_size)
    y_points = np.linspace(height * 0.1, height * 0.9, grid_size)
    
    proposals = []
    processed_masks = []
    
    logger.info(f"Generating SAM proposals with {grid_size}x{grid_size} grid...")
    
    for i, x in enumerate(x_points):
        for j, y in enumerate(y_points):
            input_points = [[[x, y]]]
            
            try:
                inputs = sam_processor(
                    images=image,
                    input_points=input_points,
                    return_tensors="pt"
                )
                
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                         for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = sam_model(**inputs)
                
                masks = sam_processor.image_processor.post_process_masks(
                    outputs.pred_masks.cpu(),
                    inputs["original_sizes"].cpu(),
                    inputs["reshaped_input_sizes"].cpu()
                )
                
                batch_masks = masks[0]
                if len(batch_masks) == 0:
                    continue
                
                point_masks = batch_masks[0]
                if len(point_masks) == 0:
                    continue
                
                # Use SAM quality scores to select the strongest mask proposal.
                best_mask_idx = 0
                best_score = 0.5
                if hasattr(outputs, 'iou_scores') and outputs.iou_scores is not None:
                    try:
                        iou_scores = outputs.iou_scores[0][0]
                        if len(iou_scores) > 0:
                            best_mask_idx = torch.argmax(iou_scores).item()
                            best_score = iou_scores[best_mask_idx].item()
                            
                            if best_score < mask_quality_threshold:
                                continue
                    except:
                        pass
                
                mask = point_masks[best_mask_idx]
                if isinstance(mask, torch.Tensor):
                    mask_np = mask.cpu().numpy().astype(bool)
                else:
                    mask_np = np.array(mask).astype(bool)
                
                # Check for duplicates
                is_duplicate = False
                for existing_mask in processed_masks:
                    overlap = np.sum(mask_np & existing_mask)
                    union = np.sum(mask_np | existing_mask)
                    if union > 0 and overlap / union > 0.8:
                        is_duplicate = True
                        break
                
                if not is_duplicate and np.sum(mask_np) > 100:
                    proposals.append({
                        'mask': mask_np,
                        'area': np.sum(mask_np),
                        'point': [x, y],
                        'confidence': best_score
                    })
                    processed_masks.append(mask_np)
                    
            except Exception as e:
                logger.warn(f"Failed to do segmentation: {e}")
    
    logger.info(f"Generated {len(proposals)} unique segment proposals")
    return proposals

def score_with_clip(
        image: Image.Image, 
        masks: np.ndarray, 
        clip_model: object, 
        clip_processor: object, 
        device: str,
        padding_ratio: float = 0.1,
    ) -> tuple:
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
            image_features /= torch.linalg.vector_norm(image_features, ord=2)  # Normalize the feature vector to unit L2 length
        
        # Cosine similarity with predetermined feature vectors

        # Safety checks (humans are NOT traversible)

        return #image_features.cpu().numpy().squeeze()
        
    except Exception as e:
        logger.error(f"Failed to extract CLIP features: {e}")
        return None
    pass

def pipeline() -> None:

    # Load image from some kind of buffer maybe?
    image = Image.open("./Rellis_3D_image_example/pylon_camera_node/frame000002-1581624652_949.jpg")


    sam_model, sam_processor, sam_device = load_sam_model()

    segmentations = generate_sam_masks(image, sam_model, sam_processor, sam_device)

    for segment in segmentations:
        
        mask = segment["mask"]

        segment["clip_score"] = score_with_clip(image, mask, clip_model, clip_processor, clip_device)

    # Huh

