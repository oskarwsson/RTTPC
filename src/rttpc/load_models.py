from transformers import (
    SamModel,
    SamProcessor,
    SamConfig,
    CLIPModel,
    CLIPProcessor,
    CLIPConfig,
    AutoImageProcessor, 
    AutoModelForDepthEstimation,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
)
import torch

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

def load_qwen_model(
    device: str | None = None,
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
) -> tuple[Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor, str]:
    """Load a Qwen2.5-VL checkpoint; return (model, processor, device).

    Defaults to CUDA when available, otherwise CPU. ``model_id`` can select
    another Qwen2.5-VL instruction checkpoint or a local checkpoint directory.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float32
    if device.type == "cuda":
        with torch.cuda.device(device):
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    logger.info("Loading Qwen model: %s...", model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, dtype=dtype)
    model = model.to(device)
    model.eval()
    logger.info("Qwen model loaded on device: %s", device)
    return model, processor, str(device)

def load_sam_model(device: str|None = None) -> tuple[SamModel, SamProcessor, str]:
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

    config = SamConfig(return_dict=False)
    processor = SamProcessor.from_pretrained(model_id)
    model = SamModel(config).from_pretrained(model_id)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model = model.to(device)
    model.eval()

    logger.info(f"SAM model loaded on device: {device}")
    
    return model, processor, str(device)


def load_clip_model(device: str|None = None) -> tuple[CLIPModel, CLIPProcessor, str]:
    """
    Load CLIP model from Hugging Face using transformers.

    Args:
        device: Device to use (cuda/cpu)

    Returns:
        Tuple of (model, processor, device)
    """

    model_name = "openai/clip-vit-base-patch32"

    logger.info(f"Loading CLIP model: {model_name}...")

    config = CLIPConfig(return_dict=False)
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel(config).from_pretrained(model_name)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model = model.to(device)
    model.eval()

    logger.info(f"CLIP model loaded on device: {device}")

    return model, processor, str(device)


def load_da_model(device: str|None = None, outdoor: bool = True) -> tuple[AutoModelForDepthEstimation, AutoImageProcessor, str]:
    """
    Load DepthAnything model from Hugging Face using transformers..

    Args:
        device: Device to use (cuda/cpu)

    Returns:
        Tuple of (model, processor, device)
    """
    model_name = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
    if not outdoor:
        model_name = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"

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
