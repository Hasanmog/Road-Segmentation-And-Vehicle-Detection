# utils/visualization.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_boxes(image: np.ndarray, boxes: list, confidences: list = None, classes: list = None,
                   color: tuple = (0, 255, 0), thickness: int = 2):
    """
    Visualize bounding boxes on the image.
    
    Args:
        image: Input image (BGR format)
        boxes: List of boxes in [x1, y1, x2, y2] format
        confidences: Optional list of confidence scores
        classes: Optional list of class IDs
        color: BGR color tuple for boxes
        thickness: Line thickness for boxes
    
    Returns:
        np.ndarray: Image with drawn boxes
    """
    img_with_boxes = image.copy()
    
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        
        # Draw the box
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, thickness)
        
        # If we have confidence and class info, draw them
        if confidences is not None and classes is not None:
            label = f"Class {classes[idx]}: {confidences[idx]:.2f}"
            cv2.putText(img_with_boxes, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    
    return img_with_boxes

def visualize_masks(image: np.ndarray, masks: list, alpha: float = 0.5):
    """
    Visualize segmentation masks on the image.
    
    Args:
        image: Input image (BGR format)
        masks: List of binary masks
        alpha: Transparency factor for the masks
    
    Returns:
        np.ndarray: Image with drawn masks
    """
    img_with_masks = image.copy()
    
    # Generate random colors for each mask
    colors = np.random.randint(0, 255, size=(len(masks), 3))
    
    for mask, color in zip(masks, colors):
        # Create colored mask
        colored_mask = np.zeros_like(image)
        colored_mask[mask] = color
        
        # Blend the colored mask with the image
        img_with_masks = cv2.addWeighted(img_with_masks, 1, colored_mask, alpha, 0)
    
    return img_with_masks

def display_results(image: np.ndarray, boxes: list, masks: list, 
                   confidences: list = None, classes: list = None):
    """
    Display the original image, image with boxes, and image with masks side by side.
    
    Args:
        image: Input image (BGR format)
        boxes: List of boxes
        masks: List of masks
        confidences: Optional list of confidence scores
        classes: Optional list of class IDs
    """
    # Convert BGR to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create visualizations
    img_with_boxes = visualize_boxes(image_rgb, boxes, confidences, classes)
    img_with_masks = visualize_masks(image_rgb, masks)
    
    # Display results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.title('Original Image')
    plt.imshow(image_rgb)
    plt.axis('off')
    
    plt.subplot(132)
    plt.title('Bounding Boxes')
    plt.imshow(img_with_boxes)
    plt.axis('off')
    
    plt.subplot(133)
    plt.title('Segmentation Masks')
    plt.imshow(img_with_masks)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    
    
# def visualize_masks_with_boxes(image: np.ndarray, masks: list, boxes: list,
#                                confidences: list = None, classes: list = None,
#                                mask_alpha: float = 0.5, box_color: tuple = (0, 255, 0), box_thickness: int = 2):
#     """
#     Visualize segmentation masks and bounding boxes on the same image.

#     Args:
#         image (np.ndarray): Input image (BGR format)
#         masks (list): List of binary masks
#         boxes (list): List of bounding boxes in [x1, y1, x2, y2]
#         confidences (list): Optional confidence scores
#         classes (list): Optional class indices
#         mask_alpha (float): Opacity for mask overlay
#         box_color (tuple): Color of the bounding boxes (BGR)
#         box_thickness (int): Thickness of bounding box lines

#     Returns:
#         np.ndarray: Image with both masks and boxes drawn
#     """
#     img_combined = visualize_masks(image, masks, alpha=mask_alpha)
#     img_combined = visualize_boxes(img_combined, boxes, confidences, classes,
#                                    color=box_color, thickness=box_thickness)
#     return img_combined
