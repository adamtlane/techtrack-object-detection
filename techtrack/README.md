# ***TechTrack* Rectification and Inference System**

This project implements an end-to-end real-time Object Detection Inference Service using a YOLO-based pipeline. It supports frame preprocessing, model prediction, and Non-Maximum Suppression (NMS), along with evaluation metrics such as mean Average Precision (mAP). The system is designed to process video input streams over UDP and can be deployed locally or as a Dockerized service.

## **Project Structure**

```
techtrack/
├── modules/
│   ├── inference/
│   │   ├── model.py
│   │   ├── nms.py
│   │   └── preprocessing.py
│   ├── rectification/
│   │   └── hard_negative_mining.py
│   └── utils/
│       ├── loss.py
│       └── metrics.py
├── storage/
│   ├── detections/
│   ├── logistics/
│   ├── test_videos/
│   └── yolo_models/
├── app.py
├── Dockerfile
├── requirements.txt
└── README.md
```

## **Brief Overview**

* `modules/inference/model.py`: Implements the `Detector` class, responsible for running the YOLO model inference and applying post-processing to filter raw model predictions based on confidence thresholds.

* `modules/inference/nms.py`: Contains a custom `NumPy`-based implementation of Non-Maximum Suppression (NMS) to eliminate redundant overlapping bounding boxes in object detection outputs.

* `modules/inference/preprocessing.py`: Handles video stream capture and frame sampling logic, enabling real-time processing by yielding every `drop-rate`-th frame from a UDP video stream.


* `modules/rectification/hard_negative_mining.py`: Implements the `sample_hard_negatives()` method to extract the most informative false positives, aiding in model retraining and improving detection accuracy.

* `modules/utils/loss.py`: Defines the YOLO loss function components, including classification, localization, and objectness losses, critical for evaluating model performance during training.

* `modules/utils/metrics.py`: Provides evaluation utilities including Intersection of Unions (IoU), detection matching, and precision-recall curves, culminating in the computation of mean Average Precision (mAP).

* `storage/detections/`: Used at runtime to store annotated frames with bounding boxes drawn around detected objects; contents should not be version-controlled.

* `storage/logistics`: Holds ground truth datasets for evaluating detection performance; required for metrics computation like precision-recall and mAP.

* `storage/test_videos`: Contains raw video files used for real-time inference testing over UDP streams, and is used by the preprocessing pipeline.

* `storage/yolo_models/`: Stores pretrained YOLO model weights and configuration files, which are loaded by the Detector during inference.

* `app.py`: Serves as the orchestration script that integrates preprocessing, detection, and `NMS` modules into a unified pipeline for streaming video inference.

* `Dockerfile`: Defines the containerized runtime environment for the Inference Service, ensuring compatibility (for example, with `opencv-python-headless`) during deployment.

* `README.md`: Provides a quick-start guide for running the Inference Service using Docker, including streaming setup instructions and example commands.

* `requirements.txt`: Specifies the Python dependencies needed to run the project, including `OpenCV`, `NumPy`, and other essential libraries.

## **Local Setup**

Keep in mind that the following requires Python 3.11 and `pip` by default. First, create and activate a virtual environment (if not already done so):

```
python3 -m venv venv

source venv/bin/activate
```

On Windows, the command is changed to `venv\Scripts\activate`., Next, install the necessary dependencies and allowed libraries, located within the `requirements.txt` file:

```
pip install -r requirements.txt
```

## **Running the Program Locally**

This program can be run both locally, and using Docker. Before working through these simple steps, ensure that you are located in the `techtrack` folder (for all terminal windows used). This is easily accomplished with:

```
cd techtrack
```

After doing so, run `ffmpeg` to stream a video to `UDP port 23000`, with the following command:

`ffmpeg -re -stream_loop -1 -i storage/test_videos/worker-zone-detection.mp4 -vcodec libx264 -preset ultrafast -f mpegts udp://127.0.0.1:23000`

In a separate terminal, and still within the `techtrack` folder, run the `app.py`:

`python app.py`

In this second terminal, you should be able to see detection results printed for each sampled frame, including class scores and bounding boxes.

## **Running the Program with Docker**

For the following steps, ensure you have Docker installed. Then, build and run the container inside the `techtrack` folder:

`docker build -t techtrack-inference .`

In the same terminal, run the following command to again start the video stream:

`ffmpeg -re -stream_loop -1 -i storage/test_videos/worker-zone-detection.mp4 -vcodec libx264 -preset ultrafast -f mpegts udp://127.0.0.1:23000`

Now, in a second terminal, run the container and expose `UDP port 23000` (the same one used by the video stream). This will start the video inference pipeline. One thing to keep in mind is that the command differs slightly depending on language. For Git Bash (or Unix/macOS-style shells), run:

`docker run --rm -it -e DOCKER=1 -v "$PWD":/app -w /app techtrack-infer python app.py`

However, for Windows Command Prompt, edit it slightly to:

`docker run --rm -it -e DOCKER=1 -v "%cd%":/app -w /app techtrack-infer python app.py`

After the pipeline is finished running with either method (locally or with Docker), the annotated detection frames will be saved to:

`storage/detections/`

## **Additional Testing**

From the `techtrack/modules/inference` directory, you can also run:

`python model.py`

Recall that `model.py` is responsible for defining and testing the `Detector` class, which is the core object detection engine of the system. Even though it’s part of a larger real-time video pipeline (`app.py`), `model.py` can be used independently to test the model logic and output correctness, without involving live video streaming or UDP input.