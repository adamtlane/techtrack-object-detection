import numpy as np
from ..utils.metrics import calculate_iou # Ensure the correct relative path.

class Loss:
    """
    A YOLO-style loss class that computes loss terms for object detection models.

    Attributes:
        iou_threshold (float): The minimum IoU to match a prediction to a ground truth box.
        lambda_coord (float): The weight for localization loss.
        lambda_obj (float): The weight for object confidence loss (when matched).
        lambda_noobj (float): The weight for object confidence loss (when unmatched).
        lambda_cls (float): The weight for classification loss.
        num_classes (int): The number of target classes.
        use_cross_entropy (bool): Whether to use cross-entropy or MSE for classification loss.
        columns (list): The names of individual loss components.
    """
    def __init__(
        self, 
        iou_threshold=0.5, 
        lambda_coord=5.0, 
        lambda_obj=1.0, 
        lambda_noobj=0.5, 
        lambda_cls=1.0, 
        num_classes=20,
        use_cross_entropy=False
    ):
        """
        Initialize the loss function with configurable weights and IoU threshold.

        Parameters:
            iou_threshold (float): The IoU threshold for determining positive matches.
            lambda_coord (float): The multiplier for localization loss.
            lambda_obj (float): The multiplier for object confidence loss (positive match).
            lambda_noobj (float): The multiplier for object confidence loss (negative match).
            lambda_cls (float): The multiplier for classification loss.
            num_classes (int): The number of classification labels.
            use_cross_entropy (bool): Use cross-entropy instead of MSE for class loss.
        """
        self.iou_threshold = iou_threshold
        self.lambda_coord = lambda_coord
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_cls = lambda_cls
        self.num_classes = num_classes
        self.use_cross_entropy = use_cross_entropy
        self.columns = ['total_loss', 'loc_loss', 'conf_loss_obj', 'conf_loss_noobj', 'class_loss']

    def get_predictions(self, predictions):
        """
        Parse the raw prediction tensor into bounding boxes, objectness scores, and class scores.

        Parameters:
            predictions (np.ndarray or list): The prediction array of shape (N, M>=6), or (N, 1, M).

        Returns:
            boxes (np.ndarray): The shape (N, 4), coordinates of predicted boxes.
            objectness (np.ndarray): The shape (N,), object confidence scores.
            class_scores (np.ndarray): The shape (N, num_classes), raw class scores.
        """
        if isinstance(predictions, list):
            predictions = np.array(predictions) # Convert the list to a NumPy array.
            
        
        if predictions.ndim == 3 and predictions.shape[1] == 1: # Handle predictions shaped (N, 1, M) by removing axis 1.
            predictions = predictions.squeeze(1)
        
        if predictions.ndim != 2 or predictions.shape[1] < 6: # Ensure input has at least 6 columns (x, y, w, h, obj_score, class1, ..., classN).
            raise ValueError(f"Predictions must have shape (N, M>=6), got shape {predictions.shape}.")

        # Split predictions into various components.
        boxes = predictions[:, 0:4] # Bounding box in the format [x, y, w, h].
        obj_scores = predictions[:, 4] # Objectness confidence.
        class_scores = predictions[:, 5:] # Class probability scores.

        return boxes, obj_scores, class_scores

    def get_annotations(self, annotations):
        """
        Parse ground truth annotations into bounding boxes and class labels.

        Parameters:
            annotations (np.ndarray or list): Each annotation in the form [x, y, w, h, class_id].

        Returns:
            boxes (np.ndarray): The shape (N, 4), ground truth box coordinates.
            labels (np.ndarray): The shape (N,), class indices.
        """
        annotations = np.array(annotations)
        
        if annotations.size == 0: # Handle empty annotations.
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=int)

        if annotations.ndim == 1: # Expand 1D array to 2D if only one annotation exists.
            annotations = np.expand_dims(annotations, axis=0)

        boxes = annotations[:, :4].astype(np.float32)
        labels = annotations[:, 4].astype(int)

        return boxes, labels

    def compute(self, predictions, annotations):
        """
        Compute YOLO-style loss for a single image.

        Parameters:
            predictions (list or np.ndarray): Model predictions for one image.
            annotations (list or np.ndarray): Ground truth annotations.

        Returns:
            dict: The loss values with keys: 'total_loss', 'loc_loss', 'conf_loss_obj', 'conf_loss_noobj', 'class_loss'.
        """
        loc_loss = 0.0 # Initialize the loss components.
        conf_loss_obj = 0.0
        conf_loss_noobj = 0.0
        class_loss = 0.0

        pred_boxes, pred_obj_scores, pred_class_scores = self.get_predictions(predictions) # Extract components from model output and ground truth.
        gt_boxes, gt_labels = self.get_annotations(annotations)

        matched_pred_indices = set()
        matched_gt_indices = set()

        for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)): # Match each ground truth to the best prediction with IoU above threshold.
            ious = np.array([calculate_iou(gt_box, pred_box) for pred_box in pred_boxes])
            best_pred_idx = np.argmax(ious)
            best_iou = ious[best_pred_idx]

            if best_iou >= self.iou_threshold and best_pred_idx not in matched_pred_indices:
                matched_pred_indices.add(best_pred_idx)
                matched_gt_indices.add(gt_idx)

                pred = pred_boxes[best_pred_idx] # Localization loss - use (x, y, sqrt(w), sqrt(h)) to balance small and large boxes.
                gt = gt_box
                pred_transformed = np.concatenate([pred[:2], np.sqrt(np.clip(pred[2:], 1e-6, None))])
                gt_transformed = np.concatenate([gt[:2], np.sqrt(np.clip(gt[2:], 1e-6, None))])
                loc_loss += np.sum((pred_transformed - gt_transformed) ** 2)

                conf_loss_obj += (1.0 - pred_obj_scores[best_pred_idx]) ** 2 # Confidence loss (object present).

                if self.use_cross_entropy: #  Classification loss.
                    scores = pred_class_scores[best_pred_idx]
                    probs = np.exp(scores) / np.sum(np.exp(scores)) # Softmax function.
                    class_loss -= np.log(np.clip(probs[gt_label], 1e-7, 1 - 1e-7))
                else: # MSE between predicted scores and one-hot vector.
                    true_class = np.eye(self.num_classes)[gt_label] 
                    class_loss += np.sum((pred_class_scores[best_pred_idx] - true_class) ** 2)
        
        for i in range(len(pred_boxes)): # Confidence loss (no object), penalize unmatched predictions as background.
            if i not in matched_pred_indices:
                conf_loss_noobj += (pred_obj_scores[i]) ** 2

        total_loss = (
            self.lambda_coord * loc_loss +
            self.lambda_obj * conf_loss_obj +
            self.lambda_noobj * conf_loss_noobj +
            self.lambda_cls * class_loss
        ) # Total loss calculations.

        return {
            'total_loss': total_loss,
            'loc_loss': loc_loss,
            'conf_loss_obj': conf_loss_obj,
            'conf_loss_noobj': conf_loss_noobj,
            'class_loss': class_loss
        }