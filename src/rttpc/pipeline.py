import numpy as np
import torch
from PIL import Image
import time
from load_models import load_sam_model, load_clip_model, load_da_model
from visualization import visualize_segment_scores

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    force=True
)
logger = logging.getLogger(__name__)

def generate_sam_masks(
        image: Image.Image,
        sam_model: object,
        sam_processor: object,
        device: str,
        grid_size: int = 6,
        mask_quality_threshold: float = 0.5,
    ) -> list[dict]:
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
                logger.warning(f"Failed to do segmentation: {e}")
    
    logger.info(f"Generated {len(proposals)} unique segment proposals")
    return proposals

def score_with_clip(
        image: Image.Image, 
        mask: np.ndarray, 
        clip_model: object, 
        clip_processor: object, 
        device: str,
        safety_embeddings: torch.Tensor,
        trav_embeddings: torch.Tensor,
        padding_ratio: float = 0.1,
        safety_threshold: float = 0.3,
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
            if not isinstance(image_features, torch.Tensor):
                image_features = image_features.pooler_output
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            #image_features /= torch.linalg.vector_norm(image_features, ord=2)  # Normalize the feature vector to unit L2 length

        image_features = image_features.cpu().numpy().squeeze()
        # Safety checks (humans are NOT traversible)
        for text_features in safety_embeddings:
            if np.dot(image_features, text_features).squeeze() > safety_threshold:
                return 0

        scores = []
        # Cosine similarity with predetermined feature vectors
        for text_features in trav_embeddings:
            scores.append(np.dot(image_features, text_features).squeeze())

        return max(scores)
        
    except Exception as e:
        logger.error(f"Failed to extract CLIP features: {e}")
        return None

def prepare_text_embeddings(
        safety_queries: list[str], 
        trav_queries: list[str], 
        clip_model: object, 
        clip_processor: object, 
        clip_device: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:

    # Prepare "safety" embeddings
    safety_embeddings = [] 
    for text_query in safety_queries:
        inputs = clip_processor(text=[text_query], return_tensors="pt")
        inputs = {k: v.to(clip_device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_features = clip_model.get_text_features(**inputs)

            if not isinstance(text_features, torch.Tensor):
                text_features = text_features.pooler_output

            # Normalize text features for cosine-similarity queries.
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
        safety_embeddings.append(text_features.cpu().numpy().squeeze())

    # Prepare "traversablity" embeddings
    trav_embeddings = []
    for text_query in trav_queries:
        inputs = clip_processor(text=[text_query], return_tensors="pt")
        inputs = {k: v.to(clip_device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_features = clip_model.get_text_features(**inputs)

            if not isinstance(text_features, torch.Tensor):
                text_features = text_features.pooler_output

            # Normalize text features for cosine-similarity queries.
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
        trav_embeddings.append(text_features.cpu().numpy().squeeze())

    return safety_embeddings, trav_embeddings

def pipeline() -> None:

    # Future TODO: Load image from some kind of buffer maybe?
    image = Image.open("./Rellis_3D_image_example/pylon_camera_node/frame000002-1581624652_949.jpg")
    safety_queries = ["human", "person", "animal", "man", "woman", "child", "baby", "cliff"]
    trav_queries = ["path", "ground", "walkable", "floor", "grass", "road", "walkway"]

    sam_model, sam_processor, sam_device = load_sam_model()

    clip_model, clip_processor, clip_device = load_clip_model()

    # Pre-process queries for faster scoring (shaves off roughly 1 second of processing time per image)
    safety_embeddings, trav_embeddings = prepare_text_embeddings(safety_queries, trav_queries, clip_model, clip_processor, clip_device)

    logger.info(f"All models loaded.")

    start_processing_time = time.time()

    segmentations = generate_sam_masks(image, sam_model, sam_processor, sam_device)

    end_SAM_processing_time = time.time()
    logger.info(f"SAM processing time: {(end_SAM_processing_time-start_processing_time):.4f}s")

    for segment in segmentations:
        mask = segment["mask"]
    
        segment["clip_score"] = score_with_clip(image, mask, clip_model, clip_processor, clip_device, safety_embeddings, trav_embeddings)
        # TODO: Maybe scale final CLIP score with SAM confidence? Example:
        segment["trav_score"] = segment["confidence"] * segment["clip_score"]

    end_CLIP_processing_time = time.time()
    logger.info(f"CLIP processing time: {(end_CLIP_processing_time-end_SAM_processing_time):.4f}s (avg {(end_CLIP_processing_time-end_SAM_processing_time)/len(segmentations):.4f}s per segment)")
    logger.info(f"Total processing time: {(end_CLIP_processing_time-start_processing_time):.4f}s")

    logger.info("Visualizing results...")
    visualize_segment_scores(segmentations, image)


def main() -> None:
    pipeline()

if __name__=="__main__":
    main()