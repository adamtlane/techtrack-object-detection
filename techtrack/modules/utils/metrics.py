import numpy as np
from sklearn.preprocessing import label_binarize

def calculate_iou(boxA, boxB):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Parameters:
        boxA (list or np.ndarray): The bounding box A in the [x1, y1, x2, y2] format.
        boxB (list or np.ndarray): The bounding box B in the [x1, y1, x2, y2] format.

    Returns:
        float: The IoU value between 0 and 1, representing the overlap between boxA and boxB.
    """
    boxA = np.squeeze(np.asarray(boxA)).tolist()
    boxB = np.squeeze(np.asarray(boxB)).tolist()

    xA1, yA1, xA2, yA2 = boxA # Extract the coordinates.
    xB1, yB1, xB2, yB2 = boxB

    x_left = max(xA1, xB1) # Calculate the intersection coordinates.
    y_top = max(yA1, yB1)
    x_right = min(xA2, xB2)
    y_bottom = min(yA2, yB2)

    if x_right < x_left or y_bottom < y_top: # Return 0 IoU if there is no overlap.
        return 0.0

    inter_area = (x_right - x_left) * (y_bottom - y_top) # Area of intersection rectangle.

    boxA_area = (xA2 - xA1) * (yA2 - yA1) # Areas of the original boxes.
    boxB_area = (xB2 - xB1) * (yB2 - yB1)
    
    iou = inter_area / float(boxA_area + boxB_area - inter_area) # Compute the IoU.

    return iou

def evaluate_detections(boxes, classes, scores, cls_scores, gt_boxes, gt_classes, map_iou_threshold, eval_type="class_scores"):
    """
    Match predicted boxes with ground truth and collect true labels and scores for mAP calculation.

    Parameters:
        boxes (list): The predicted bounding boxes per image.
        classes (list): The predicted class IDs per image.
        scores (list): The confidence scores per image.
        cls_scores (list): The class probability vectors per image.
        gt_boxes (list): The ground truth boxes per image.
        gt_classes (list): The ground truth class IDs per image.
        map_iou_threshold (float): The IoU threshold for matching predicted and ground truth boxes.
        eval_type (str): The evaluation mode, one of {'objectness', 'class_scores', 'combined'}.

    Returns:
         y_true (list): The true class labels (or -1 for false positives).
         pred_scores (list): The corresponding prediction scores.
    """
    y_true = []
    pred_scores = []

    num_images = len(boxes)

    for i in range(num_images):
        detected_boxes = boxes[i]
        detected_classes = classes[i]
        detected_scores = scores[i]
        detected_cls_scores = cls_scores[i]
        gt_bxs = gt_boxes[i]
        gt_cls = gt_classes[i]

        num_dets = len(detected_boxes)
        num_gts = len(gt_bxs)

        if num_dets == 0 and num_gts == 0: # Skip any empty images.
            continue
        
        iou_matrix = np.zeros((num_dets, num_gts)) if num_dets > 0 and num_gts > 0 else np.zeros((num_dets, 0)) # Initialize the IoU matrix.

        for d in range(num_dets):
            for g in range(num_gts):
                iou_matrix[d, g] = calculate_iou(detected_boxes[d], gt_bxs[g])

        det_order = np.argsort(detected_scores)[::-1] # Sort the detections by descending score.
        gt_matched = np.zeros(num_gts, dtype=bool)

        for d_idx in det_order:
            best_iou = 0
            best_gt_idx = -1

            for g in range(num_gts): # Find the best unmatched GT box for the current detection.
                if not gt_matched[g] and iou_matrix[d_idx, g] >= map_iou_threshold:
                    if iou_matrix[d_idx, g] > best_iou:
                        best_iou = iou_matrix[d_idx, g]
                        best_gt_idx = g

            if best_gt_idx >= 0: # Condition case for a true positive match.
                gt_matched[best_gt_idx] = True
                true_label = gt_cls[best_gt_idx]

                if eval_type == "objectness": # Obtain the appropriate score.
                    pred_score = detected_scores[d_idx]
                elif eval_type == "class_scores":
                    class_id = true_label
                    cls_vec = detected_cls_scores[d_idx]
                    pred_score = cls_vec[class_id] if len(cls_vec) > class_id else 0.0
                elif eval_type == "combined":
                    class_id = true_label
                    cls_vec = detected_cls_scores[d_idx]
                    cls_score = cls_vec[class_id] if len(cls_vec) > class_id else 0.0
                    pred_score = detected_scores[d_idx] * cls_score
                else:
                    raise ValueError(f"Unknown eval_type '{eval_type}'")

                y_true.append(true_label)
                pred_scores.append(pred_score)

            else: # Condition case for a false positive.
                pred_label = detected_classes[d_idx]

                if eval_type == "objectness":
                    pred_score = detected_scores[d_idx]
                elif eval_type == "class_scores":
                    cls_vec = detected_cls_scores[d_idx]
                    pred_score = cls_vec[pred_label] if len(cls_vec) > pred_label else 0.0
                elif eval_type == "combined":
                    cls_vec = detected_cls_scores[d_idx]
                    cls_score = cls_vec[pred_label] if len(cls_vec) > pred_label else 0.0
                    pred_score = detected_scores[d_idx] * cls_score
                else:
                    raise ValueError(f"Unknown eval_type '{eval_type}'")

                y_true.append(-1) # Mark as false positive.
                pred_scores.append(pred_score)

        for g in range(num_gts): # Handle the false negatives.
            if not gt_matched[g]:
                y_true.append(gt_cls[g])
                pred_scores.append(0.0)

    return y_true, pred_scores

def calculate_precision_recall_curve(y_true, pred_scores, num_classes=20):
    """
    Generate precision-recall curve for each class based on predictions.

    Parameters:
        y_true (list): THe true labels (or -1 for false positives).
        pred_scores (list or np.ndarray): The redicted scores (list of floats or 2D array for multi-class).
        num_classes (int): The number of classes.

    Returns:
        precision (dict): The class-wise precision values.
        recall (dict): The class-wise recall values.
        thresholds (dict): The thresholds used to compute PR points.
    """
    y_true_bin = label_binarize(y_true, classes=np.arange(num_classes)) # Convert ground truth labels to one-hot format.
    precision = {}
    recall = {}
    thresholds = {}

    for c in range(num_classes):
        scores_c = pred_scores[:, c] if isinstance(pred_scores, np.ndarray) and pred_scores.ndim > 1 else None
        true_c = y_true_bin[:, c]

        order = np.argsort(-pred_scores[:, c]) if scores_c is not None else np.argsort(-np.array(pred_scores)) # Sort scores in descending order.
        true_sorted = true_c[order]
        scores_sorted = np.array(pred_scores)[order, c] if scores_c is not None else np.array(pred_scores)[order]

        tps = 0
        fps = 0
        fn = np.sum(true_c)

        prec_curve = []
        rec_curve = []
        thresh = []

        for i in range(len(scores_sorted)):
            if true_sorted[i] == 1:
                tps += 1
                fn -= 1
            else:
                fps += 1

            prec = tps / (tps + fps) if (tps + fps) > 0 else 0
            rec = tps / (tps + fn) if (tps + fn) > 0 else 0

            prec_curve.append(prec)
            rec_curve.append(rec)
            thresh.append(scores_sorted[i])

        prec_curve.append(0) # Add the final point (0 precision/recall).
        rec_curve.append(0)
        thresh.append(0)

        precision[c] = prec_curve
        recall[c] = rec_curve
        thresholds[c] = np.array(thresh)

    return precision, recall, thresholds

def calculate_map_x_point_interpolated(precision_recall_points, num_classes, num_interpolated_points=11):
    """
    Compute mean Average Precision (mAP) with x-point interpolation over recall values.

    Parameters:
        precision_recall_points (dict): A dictionary mapping class indices to lists of (recall, precision) tuples.
        num_classes (int): The number of classes.
        num_interpolated_points (int): The number of recall points for interpolation (default = 11 for Pascal VOC style).

    Returns:
        float: The final mean Average Precision (mAP) score over all classes.
    """
    mean_average_precisions = []

    for i in range(num_classes):
        points = precision_recall_points[i]

        points = sorted(points, key=lambda x: x[1], reverse=True) # Sort by descending precision.

        interpolated_precisions = []

        for recall_threshold in [j * 0.1 for j in range(num_interpolated_points)]: # Find the max precision for all recalls greater than or equal to the threshold.
            precisions_above = [p for r, p in points if r >= recall_threshold]
            if precisions_above:
                interpolated_precisions.append(max(precisions_above))
            else:
                interpolated_precisions.append(0)

        mean_ap = np.mean(interpolated_precisions) # The mean of interpolated precisions for the current class.
        mean_average_precisions.append(mean_ap)

    return np.mean(mean_average_precisions) # Average over all the classes.