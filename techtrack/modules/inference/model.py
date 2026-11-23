import cv2
import numpy as np
from typing import List, Tuple
import sys
import os
from inference.nms import NMS

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) # Add "techtrack/modules" to sys.path so we can import local modules.

class Detector:
    """
    A class for object detection using OpenCV's DNN module with a YOLO-based architecture.

    Attributes:
        net: The OpenCV DNN network loaded from YOLO weights and config.
        output_layers: The names of the YOLO output layers.
        classes: The list of class names loaded from the class_path file.
        score_threshold: The confidence threshold for filtering weak detections.
        nms_iou_threshold: The IoU threshold for Non-Maximum Suppression.
        img_height: The height of the input image.
        img_width: The width of the input image.
        input_size: The fixed input size expected by the model (typically 416x416).
    """

    def __init__(self, weights_path: str, config_path: str, class_path: str,
                 score_threshold: float = 0.5, nms_iou_threshold: float = 0.4) -> None:
        """
        Initializes the detector with YOLO weights, config, and class names.

        Parameters:
            weights_path (str): The path to the YOLO weights file.
            config_path (str): The path to the YOLO config file.
            class_path (str): The path to the file containing the class names.
            score_threshold (float): The minimum score needed to consider a detection.
            nms_iou_threshold (float): The IoU threshold for Non-Maximum Suppression.
        """
        self.net = cv2.dnn.readNet(weights_path, config_path) # Load the YOLO model from the associated weights and config.
        self.output_layers = self.net.getUnconnectedOutLayersNames()

        with open(class_path, "r") as f: # Load the class labels.
            self.classes = [line.strip() for line in f.readlines()]

        self.score_threshold = score_threshold # Set the thresholds.
        self.nms_iou_threshold = nms_iou_threshold
        
        self.img_height: int = 0 # Initialize the image dimensions and model input size.
        self.img_width: int = 0
        self.input_size: Tuple[int, int] = (416, 416) # In the form (width, height).

    def predict(self, frame: np.ndarray) -> List[np.ndarray]:
        """
        Runs the YOLO model on a single input frame and returns raw predictions.

        Parameters:
            frame (np.ndarray): The input image as a NumPy array.

        Returns:
            List[np.ndarray]: The list of raw detection outputs from the YOLO layers.
        """
        self.img_height, self.img_width = frame.shape[:2] # Capture the input image size.

        
        blob = cv2.dnn.blobFromImage(frame,
                                     scalefactor=1 / 255.0,
                                     size=self.input_size,
                                     swapRB=True,
                                     crop=False)
        self.net.setInput(blob) # Prepare the image for YOLO by normalizing and resizing it.

        layer_names = self.net.getLayerNames() # Retrieve the names of the YOLO output layers using index-based lookup.
        unconnected = self.net.getUnconnectedOutLayers()
        
        outputs = self.net.forward([layer_names[i - 1] for i in unconnected.flatten()]) # Run the forward pass through the output layers.

        return outputs

    def post_process(self, outputs, score_threshold=0.5, nms_iou_threshold=0.5):
        """
        Converts raw outputs to filtered bounding boxes using Non-Maximum Suppression (NMS).
        Filters out malformed or low-confidence detections.

        Parameters:
            outputs (list): The raw outputs from YOLO's detection layers.
            score_threshold (float): The confidence threshold for filtering.
            nms_iou_threshold (float): The IoU threshold for NMS filtering.

        Returns:
            filtered_bboxes (List[List[int]]): The final bounding boxes in the form [x, y, w, h].
            filtered_class_ids (List[int]): The detected class indices.
            filtered_confidences (List[float]): The objectness scores.
            filtered_class_scores (List[float]): The final confidence scores after the class probabilities.
        """
        if not isinstance(outputs, (list, tuple)): # Ensure outputs are in the expected format.
            raise ValueError(f"[ERROR] outputs must be a list or tuple, got {type(outputs)}")
        
        outputs = list(outputs)

        detection_outputs = [o for o in outputs if isinstance(o, np.ndarray) and o.ndim == 2 and o.shape[1] == 25] # Filter for valid YOLO detection layers (shape == [?, 25]).

        
        if len(detection_outputs) == 0: # Return empty results if no valid outputs are found.
            print(f"[WARNING] No valid YOLO outputs found. Skipping post-processing.")
            return [], [], [], []

        bboxes = [] # Initialize the detection result containers.
        class_ids = []
        confidence_scores = []
        class_scores = []

        malformed_count = 0 # Handle any malformed detections gracefully with capped logging.
        max_malformed_warnings = 100

        for output in detection_outputs: # Process each detection entry.
            for detection in output:
                if not isinstance(detection, (np.ndarray, list)) or len(detection) != 25:
                    if malformed_count < max_malformed_warnings:
                        print(f"[WARNING] Skipping malformed detection of length {len(detection)}: {detection}")
                    malformed_count += 1
                    continue

                try:
                    cx, cy, w, h = detection[0:4] # Extract the bounding box and confidence components.
                    objectness = detection[4]
                    scores = detection[5:]
                    
                    class_id = int(np.argmax(scores)) # Identify the most probable class and compute the combined score.
                    class_prob = scores[class_id]
                    score = objectness * class_prob
                    
                    if objectness > self.score_threshold and score > score_threshold: # Apply the score threshold filtering.
                        box_x = int((cx - w / 2) * self.img_width) # Convert the YOLO center-based box to the top-left corner, a new format.
                        box_y = int((cy - h / 2) * self.img_height)
                        box_w = int(w * self.img_width)
                        box_h = int(h * self.img_height)

                        bboxes.append([box_x, box_y, box_w, box_h]) # Store the results.
                        confidence_scores.append(float(objectness))
                        class_scores.append(float(score))
                        class_ids.append(class_id)

                except Exception as e:
                    if malformed_count < max_malformed_warnings:
                        print(f"[ERROR] Failed to parse detection: {detection} | Error: {e}")
                    malformed_count += 1

        if malformed_count > max_malformed_warnings:
            print(f"[WARNING] Skipped {malformed_count} malformed detections. "
                  f"Only showing first {max_malformed_warnings}.")

        nms_filter = NMS(score_threshold=score_threshold, nms_iou_threshold=nms_iou_threshold) # Apply Non-Maximum Suppression using the custom NMS wrapper.
        filtered_bboxes, filtered_class_ids, filtered_confidences, filtered_class_scores = nms_filter.filter(
            bboxes, class_ids, confidence_scores, class_scores
        )

        return filtered_bboxes, filtered_class_ids, filtered_confidences, filtered_class_scores

if __name__ == "__main__":
    weights_path = "../../storage/yolo_models/yolo_model_1/yolov4-tiny-logistics_size_416_1.weights" # Paths to the model and sample image.
    config_path = "../../storage/yolo_models/yolo_model_1/yolov4-tiny-logistics_size_416_1.cfg"
    class_path = "../../storage/yolo_models/yolo_model_1/logistics.names"
    frame_path = "../../storage/logistics/p1_8760_jpg.rf.f174b5ef18011cac19021f9df63ecb7c.jpg"

    model = Detector(weights_path, config_path, class_path) # Load the detector and sample frame.
    frame = cv2.imread(frame_path)

    predictions = model.predict(frame) # Run the raw prediction and post-processing.
    bboxes, class_ids, confidence_scores, class_scores = model.post_process(predictions)

    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(1) # Visualize detections using matplotlib.
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    for bbox in bboxes:
        x, y, width, height = bbox
        r = Rectangle((x, y), width, height, facecolor="none", edgecolor="r")
        ax.add_patch(r)
        
    plt.title("Detections")
    plt.show()