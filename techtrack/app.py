import cv2
import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "modules"))) # Add the "modules" directory to the system path so custom modules can be imported.

from modules.inference.nms import NMS
from modules.inference.model import Detector
from modules.inference.preprocessing import Preprocessing

class InferenceService:
    """
    Handles real-time video inference by integrating preprocessing, object detection, 
    and Non-Maximum Suppression (NMS) filtering.

    This service captures frames from a video stream, performs object detection on each frame,
    applies NMS to eliminate redundant detections, and saves annotated frames to disk.
    """

    def __init__(self, stream: Preprocessing, detector: Detector, nms: NMS, save_dir: str, drop_rate: int = 10) -> None:
        """
        Initializes the inference service.

        Parameters:
            stream (Preprocessing): The preprocessing object for capturing video frames.
            detector (Detector): The object detection model.
            nms (NMS): The Non-Maximum Suppression filter.
            save_dir (str): The directory where annotated frames will be saved.
            drop_rate (int): The number of frames to skip between processing (default is 10).

        Attributes:
            self.stream: The video stream processor.
            self.detector: The object detection model instance.
            self.nms: The NMS processor for post-filtering.
            self.save_dir: The output directory for saving results.
        """
        self.stream = stream
        self.detector = detector
        self.nms = nms
        self.drop_rate = drop_rate
        self.save_dir = save_dir

        os.makedirs(self.save_dir, exist_ok=True) # Ensure the output directory exists.

        print("[INFO] Inference Service initialized.")

    def draw_boxes(self, frame, bboxes, class_ids, scores):
        """
        Draws bounding boxes and associated class labels/confidence scores onto a video frame.

        Parameters:
            frame (ndarray): The original frame to annotate.
            bboxes (List[Tuple[int, int, int, int]]): The detected bounding boxes in the form (x, y, width, height).
            class_ids (List[int]): The detected class IDs.
            scores (List[float]): The corresponding confidence scores.

        Returns:
            ndarray: The annotated frame.
        """
        for i, (x, y, w, h) in enumerate(bboxes): # Draw the bounding box.
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            label = f"Class {class_ids[i]}: {scores[i]:.2f}" # Construct the label text.

            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2) # Draw label text above the box.

        return frame

    def save_frame(self, frame, frame_number):
        """
        Saves the annotated frame to the output directory.

        Parameters:
            frame (ndarray): The frame to save.
            frame_number (int): The sequential number of the frame.
        """
        filename = os.path.join(self.save_dir, f"frame_{frame_number}.jpg")

        cv2.imwrite(filename, frame)

        print(f"[INFO] Saved: {filename}")

    def run(self) -> None:
        """
        Executes the real-time inference pipeline.

        This method:
            Captures frames from the video stream.
            Performs inference using the YOLO detector.
            Applies post-processing to extract bounding boxes.
            Applies Non-Maximum Suppression to reduce overlaps.
            Logs the results to the console.
            Draws bounding boxes on the frame and saves it.
            Saves detection results as JSON files per frame.

        Returns:
            None.
        """
        print("[INFO] Inference loop started")

        frame_count = 0  # Tracks the total frames processed.

        # Ensure JSON output directory exists
        json_dir = os.path.join(self.save_dir, "json")
        os.makedirs(json_dir, exist_ok=True)

        for frame in self.stream.capture_video():  # Iterate over video frames.
            frame_count += 1

            if frame is None:
                print(f"[WARNING] Frame {frame_count} is None — skipping.")
                continue

            raw_preds = self.detector.predict(frame)  # Predict raw outputs from the model.

            bboxes, class_ids, scores, class_scores = self.detector.post_process(outputs=raw_preds)  # Convert raw outputs into bounding boxes, scores, and IDs.

            filtered_bboxes, filtered_ids, filtered_scores, filtered_class_scores = self.nms.filter(
                bboxes, class_ids, scores, class_scores
            )  # Apply NMS to filter overlapping detections.

            if filtered_bboxes:  # Log the filtered detections.
                print(f"[INFO] Frame {frame_count}: {len(filtered_bboxes)} detections")
                for i, bbox in enumerate(filtered_bboxes):
                    print(f"  - BBox: {bbox} | Class: {filtered_ids[i]} | Score: {filtered_scores[i]:.2f}")
            else:
                print(f"[INFO] Frame {frame_count}: No detections")
                print(filtered_class_scores)

            # Save detections as JSON
            # Updated JSON-safe conversion
            detections_for_frame = []
            for bbox, class_id, score in zip(filtered_bboxes, filtered_ids, filtered_scores):
                x, y, w, h = [int(v) for v in bbox]  # Make sure bbox elements are ints
                detections_for_frame.append({
                    "bbox": [x, y, w, h],
                    "class_id": int(class_id),  # ensure plain int
                    "score": float(score)       # ensure plain float
                })

            json_filename = os.path.join(json_dir, f"frame_{frame_count:04d}.json")
            with open(json_filename, "w") as f:
                json.dump({
                    "frame": frame_count,
                    "detections": detections_for_frame
                }, f, indent=2)

            annotated = self.draw_boxes(frame, filtered_bboxes, filtered_ids, filtered_scores)  # Annotate and save the processed frame.
            self.save_frame(annotated, frame_count)

        print("[INFO] Inference loop completed")

if __name__ == "__main__": # Entry point for running the inference pipeline.
    print("[INFO] Launching Inference Pipeline...")
    
    if os.environ.get("DOCKER") == "1": # Choose stream source dynamically.
        VIDEO_SOURCE = "udp://host.docker.internal:23000" # For Docker.
    else:
        VIDEO_SOURCE = "udp://127.0.0.1:23000" # For local run.

    WEIGHTS_PATH = "storage/yolo_models/yolo_model_1/yolov4-tiny-logistics_size_416_1.weights" # Configuration, RTSP/UDP stream input.
    CONFIG_PATH = "storage/yolo_models/yolo_model_1/yolov4-tiny-logistics_size_416_1.cfg"
    CLASS_NAMES_PATH = "storage/yolo_models/yolo_model_1/logistics.names"
    SCORE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.4
    SAVE_DIR = "storage/detections/model_1"
    
    print(f"[INFO] Initializing video stream: {VIDEO_SOURCE}") # Component initialization.
    stream = Preprocessing(VIDEO_SOURCE, drop_rate=60)

    print("[INFO] Loading YOLO detector...")
    detector = Detector(WEIGHTS_PATH, CONFIG_PATH, CLASS_NAMES_PATH, SCORE_THRESHOLD)

    print("[INFO] Initializing NMS module...")
    nms = NMS(score_threshold=SCORE_THRESHOLD, nms_iou_threshold=IOU_THRESHOLD)

    service = InferenceService(stream, detector, nms, save_dir=SAVE_DIR) # Start the inference service.
    service.run()

    print("[INFO] Inference terminated.")