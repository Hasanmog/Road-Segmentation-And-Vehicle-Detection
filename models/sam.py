import torch
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor
from typing import List


def load_sam(checkpoint_path: str, model_type: str = "vit_h") -> SamPredictor:
    """
    Load the SAM model and return the predictor.
    Args:
        checkpoint_path (str): Path to the SAM .pth checkpoint file.
        model_type (str): SAM model type ("vit_h", "vit_l", or "vit_b").

    Returns:
        SamPredictor: A SAM predictor instance.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)

    predictor = SamPredictor(sam)
    return predictor


def sam_predict(predictor: SamPredictor, image: np.ndarray, boxes: List[List[float]], multimask: bool = True) -> List[np.ndarray]:
    """
    Predict masks using SAM from bounding boxes.

    Args:
        predictor (SamPredictor): Initialized SAM predictor.
        image (np.ndarray): Input image (RGB).
        boxes (list): List of boxes in [x1, y1, x2, y2] format.
        multimask (bool): Whether to return multiple masks per box.

    Returns:
        List[np.ndarray]: List of binary masks.
    """
    predictor.set_image(image)

    transformed_boxes = predictor.transform.apply_boxes(np.array(boxes), image.shape[:2])
    masks = []
    boxes = np.array(boxes)
    for box in boxes:
        mask, scores, logits = predictor.predict(box=box, multimask_output=multimask)
        best_idx = np.argmax(scores)
        masks.append(mask[best_idx])

    return masks , boxes.tolist()


