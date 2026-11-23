import cv2
import numpy as np
from typing import Generator
import os

class Preprocessing:
    """
    Handles video file reading and frame extraction for object detection inference.

    This class reads a video file and yields frames at a specified interval. It is commonly
    used to reduce processing load by skipping frames and selecting only every nth frame
    based on the specified `drop_rate`.
    """

    def __init__(self, filename: str, drop_rate: int = 10) -> None:
        """
        Initializes the Preprocessing object.

        Parameters:
            filename (str): The path to the input video file.
            drop_rate (int): Extract every `drop_rate`-th frame (the default is 10).

        Attributes:
            self.filename (str): Stores the video file path.
            self.drop_rate (int): The interval at which frames are extracted.
        """

        self.filename = filename
        self.drop_rate = drop_rate

    def capture_video(self) -> Generator[np.ndarray, None, None]:
        """
        Generator that reads the video and yields every `drop_rate`-th frame.

        This function:
            Opens the video file using OpenCV.
            Iterates through frames sequentially.
            Yields only every nth frame, as defined by `drop_rate`.
            Releases the video file resource after processing.

        Returns:
            Generator[np.ndarray, None, None]: A generator yielding video frames (as NumPy arrays).

        Raises:
            ValueError: If the video file cannot be opened.

        Example:
            >>> processor = Preprocessing("input.mp4", drop_rate=10)
            >>> for frame in processor.capture_video():
            >>>     run_detection(frame)
        """
        cap = cv2.VideoCapture(self.filename, cv2.CAP_FFMPEG) # Open the video file.

        if not cap.isOpened(): # Check if the video file was opened successfully.
            raise ValueError(f"Error: Unable to open video file '{self.filename}'.")

        frame_index = 0 # Counter to track the frame position.

        while cap.isOpened(): # Loop to read each frame from the video.
            ret, frame = cap.read() # Here, ret is a bool indicating successm where frame is the image.

            if not ret or frame is None:
                break # End of video or failed the frame read.
            
            if frame_index % self.drop_rate == 0: # Yield the frame only if it's at the drop_rate interval.
                yield frame

            frame_index += 1 # Increment the frame index for the next iteration.
        
        cap.release() # Release the video resource after processing is done.