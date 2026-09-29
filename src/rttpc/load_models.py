from transformers import (
    SamModel,
    SamProcessor,
    CLIPModel,
    CLIPProcessor,
    AutoImageProcessor, 
    AutoModelForDepthEstimation,
)

import logging
logger = logging.getLogger(__name__)

def load_sam_model(device: str|None = None) -> Tuple[SamModel, SamProcessor, str]:
    """
    Load SAM model from Hugging Face using transformers.

    Args:
        device: Device to use (cuda/cpu)

    Returns:
        Tuple of (model, processor, device)
    """

    """
    Model configurations from HuggingFace
        'base': 'facebook/sam-vit-base',
        'large': 'facebook/sam-vit-large',
        'huge': 'facebook/sam-vit-huge'
    """

    model_id = 'facebook/sam-vit-base'

    logger.info(f"Loading SAM model: {model_id}...")

    processor = SamProcessor.from_pretrained(model_id)
    model = SamModel.from_pretrained(model_id)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model = model.to(device)
    model.eval()

    logger.info(f"SAM model loaded on device: {device}")
    
    return model, processor, str(device)


def load_clip_model(device: str|None = None) -> Tuple[CLIPModel, CLIPProcessor, str]:
    """
    Load CLIP model from Hugging Face using transformers.

    Args:
        device: Device to use (cuda/cpu)

    Returns:
        Tuple of (model, processor, device)
    """

    model_name = "openai/clip-vit-base-patch32"

    logger.info(f"Loading CLIP model: {model_name}...")

    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model = model.to(device)
    model.eval()

    logger.info(f"CLIP model loaded on device: {device}")

    return model, processor, str(device)


def load_da_model(device: str|None = None) -> Tuple[AutoModelForDepthEstimation, AutoImageProcessor, str]:
    """
    Load DepthAnything model from Hugging Face using transformers..

    Args:
        device: Device to use (cuda/cpu)

    Returns:
        Tuple of (model, processor, device)
    """

    model_name = "LiheYoung/depth-anything-small-hf"

    logger.info(f"Loading DepthAnything model: {model_name}...")

    
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model = model.to(device)
    model.eval()

    logger.info(f"DepthAnything model loaded on device: {device}")

    return model, processor, str(device)