import os
import math
import pandas as pd
import numpy as np

from glob import glob
import cv2

from ..utils.loss import Loss
from ..inference.model import Detector
from ..inference.nms import NMS

class HardNegativeMiner:
    """
    A class to mine hard negative examples from a dataset using a given object detection model and loss function.

    Attributes:
        model (Detector): The object detection model used to generate predictions.
        nms (NMS or None): An optional non-maximum suppression (NMS) module.
        measure (Loss): The loss object used to evaluate predictions.
        dataset_dir (str): A directory containing the dataset (images and annotations).
        table (DataFrame): Stores per-sample evaluation metrics.
    """

    def __init__(self, model: Detector, measure: Loss, dataset_dir: str, nms=None):
        """
        Initializes the HardNegativeMiner with model, loss measure, and dataset directory.

        Parameters:
            model (Detector): The trained detector model.
            measure (Loss): The loss object used to evaluate each sample.
            dataset_dir (str): A directory containing images (.jpg) and annotation files (.txt).
            nms (NMS or None): An otional non-maximum suppression (NMS) instance.
        """
        self.nms = nms
        self.model = model
        self.measure = measure
        self.dataset_dir = dataset_dir

        self.table = pd.DataFrame(columns=['annotation_file', 'image_file'] + self.measure.columns) # Initialize the result table with expected columns from the loss metric.

    def __read_annotations(self, file_path):
        """
        Reads a YOLO-format annotation file.

        Parameters:
            file_path (str): The full path to the annotation .txt file.

        Returns:
            list: A list of annotations, each as a tuple (class_label, x_center, y_center, width, height).
        """
        annotations = []

        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split() # Split line into components.
                class_label = int(parts[0])  # THe first part is the class label.
                bbox = list(map(float, parts[1:])) # The rest are bbox values.
                annotations.append((class_label, *bbox))

        return annotations

    def __predict(self, image):
        """
        Uses the model to make predictions on a single image.

        Parameters:
            image (ndarray): The input image (as a NumPy array, e.g., from cv2.imread).

        Returns:
            Any: The model's prediction output, structure depends on model implementation.
        """
        output = self.model.predict(image) # Perform the forward pass.

        return output

    def __construct_table(self):
        """
        Constructs a DataFrame of performance metrics for the dataset.

        Internal Process:
            Iterates over image and annotation file pairs.
            Runs model predictions on each image.
            Computes evaluation metrics via the loss function.
            Stores results in a DataFrame (`self.table`).
        """
        table_rows = []

        for image_file, annotation_file in zip(
                sorted(glob(os.path.join(self.dataset_dir, "*.jpg"))),
                sorted(glob(os.path.join(self.dataset_dir, "*.txt")))): # Match each image with its corresponding annotation file.

            image = cv2.imread(image_file) # Load the image.
            annotation = self.__read_annotations(annotation_file) # Load the annotation.
            prediction = self.__predict(image) # Run the prediction.

            measures = self.measure.compute(prediction, annotation) # Compute loss metrics for this sample.
            
            table_rows.append({
                'annotation_file': annotation_file,
                'image_file': image_file,
                **measures
            }) # Add the row to the results table.

        self.table = pd.DataFrame(table_rows) # Convert to a DataFrame.

    def sample_hard_negatives(self, num_hard_negatives, criteria):
        """
        Samples hard negative examples from the dataset.

        Parameters:
            num_hard_negatives (int): The number of hard negative samples to return.
            criteria (str): The loss component to sort by (e.g., 'total_loss', 'conf_loss_noobj').

        Returns:
            DataFrame: The top N hardest examples sorted by the given criteria.
        """
        self.__construct_table() # Ensure the results table is constructed.

        if criteria not in self.measure.columns: # Ensure provided criteria is valid.
            raise ValueError(f"Invalid criteria '{criteria}'. Must be one of {self.measure.columns}.")
        
        sorted_table = self.table.sort_values(by=criteria, ascending=False) # Sort the table descending by the selected criteria.
        
        hard_negatives = sorted_table.head(num_hard_negatives) # Return top-N hardest examples.

        return hard_negatives
