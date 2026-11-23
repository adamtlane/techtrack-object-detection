import numpy as np
from typing import List, Tuple
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) # Add the parent "modules" directory to the Python path.

from utils.metrics import calculate_iou  # Function to compute Intersection over Union (IoU).

class NMS:
    """
    Implements Non-Maximum Suppression (NMS) to reduce redundant bounding boxes
    in object detection results.

    The NMS process keeps only the most confident bounding box per object, suppressing
    others that significantly overlap (i.e., have a high IoU).
    """

    def __init__(self, score_threshold: float, nms_iou_threshold: float) -> None:
        """
        Initializes the NMS filter with confidence and IoU thresholds.

        Parameters:
            score_threshold (float): The minimum confidence score for a detection to be considered.
            nms_iou_threshold (float): The IoU threshold above which overlapping boxes are suppressed.
        """
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold

    def filter(
        self,
        bboxes: List[List[int]],
        class_ids: List[int],
        scores: List[float],
        class_scores: List[float],
    ) -> Tuple[List[List[int]], List[int], List[float], List[float]]:
        """
        Applies NMS on a list of predicted bounding boxes.

        Parameters:
            bboxes (List[List[int]]): The list of bounding boxes in the [x, y, width, height] format.
            class_ids (List[int]): The list of predicted class IDs corresponding to each box.
            scores (List[float]): The list of confidence (objectness) scores for each box.
            class_scores (List[float]): The list of class-specific scores for each box.

        Returns:
            filtered_bboxes (List[List[int]]): The bounding boxes retained after NMS.
            filtered_class_ids (List[int]): The corresponding class IDs of retained boxes.
            filtered_scores (List[float]): The confidence scores of retained boxes.
            filtered_class_scores (List[float]): The class-specific scores of retained boxes.
        """
        
        filtered_bboxes = [] # Output containers for retained predictions.
        filtered_class_ids = []
        filtered_scores = []
        filtered_class_scores = []
        
        bboxes = np.array(bboxes) # Convert input lists to NumPy arrays for efficient indexing and operations.
        class_ids = np.array(class_ids)
        scores = np.array(scores)
        class_scores = np.array(class_scores)

        unique_classes = np.unique(class_ids) # Loop through each unique class label.

        for cls in unique_classes: # Get indices of boxes belonging to the current class.
            cls_indices = np.where(class_ids == cls)[0]
            
            cls_scores = scores[cls_indices] # Filter out boxes below score threshold.
            keep_indices = cls_scores >= self.score_threshold
            cls_indices = cls_indices[keep_indices]

            if len(cls_indices) == 0:
                continue # Skip if no valid boxes remain.

            cls_bboxes = bboxes[cls_indices] # Extract corresponding bounding boxes and scores.
            cls_scores = scores[cls_indices]
            cls_class_scores = class_scores[cls_indices]

            sorted_indices = np.argsort(-cls_scores) # Sort boxes by descending confidence.
            cls_bboxes = cls_bboxes[sorted_indices]
            cls_scores = cls_scores[sorted_indices]
            cls_class_scores = cls_class_scores[sorted_indices]
            cls_indices = cls_indices[sorted_indices]

            keep = []

            while len(cls_bboxes) > 0: # Perform iterative NMS.
                curr_box = cls_bboxes[0] # Select the highest-scoring box.
                curr_score = cls_scores[0]
                curr_class_score = cls_class_scores[0]
                curr_idx = cls_indices[0]

                keep.append((curr_box, cls, curr_score, curr_class_score)) # Retain the box.
 
                if len(cls_bboxes) == 1:
                    break  # No more boxes to compare.

                rest_boxes = cls_bboxes[1:] # Compare current box to the rest.

                ious = np.array([calculate_iou(curr_box, box) for box in rest_boxes]) # Compute IoUs between current box and remaining boxes.

                inds_to_keep = np.where(ious <= self.nms_iou_threshold)[0] # Keep only boxes with an IoU greater or equal than the threshold.

                cls_bboxes = rest_boxes[inds_to_keep] # Update the lists for the next round.
                cls_scores = cls_scores[1:][inds_to_keep]
                cls_class_scores = cls_class_scores[1:][inds_to_keep]
                cls_indices = cls_indices[1:][inds_to_keep]

            for box, c_id, score, c_score in keep: # Append the retained boxes to output lists.
                filtered_bboxes.append(box.tolist())
                filtered_class_ids.append(c_id)
                filtered_scores.append(float(score))
                filtered_class_scores.append(c_score)

        return filtered_bboxes, filtered_class_ids, filtered_scores, filtered_class_scores