# TechTrack Object Detection Inference

A streaming object detection service for logistics scenes. It samples frames from a video stream, runs a YOLOv4-tiny detector through OpenCV's DNN module, filters overlapping boxes with a NumPy implementation of non-maximum suppression, and saves annotated frames. It also includes the evaluation and rectification tools used to compare detectors and choose retraining data.

The detector recognizes 20 classes, including forklifts, freight containers, cardboard boxes, wood pallets, safety vests, helmets, barcodes, QR codes, and license plates.

<p align="center">
  <img src="techtrack/casestudy_images/frame_28.jpg" width="420"/>
  <img src="techtrack/casestudy_images/2_frame_37.jpg" width="420"/>
</p>
<p align="center"><em>Annotated frames from the two YOLO models compared in the case study.</em></p>

## How it works

1. **Frame sampling.** `Preprocessing` reads a video stream and yields every Nth frame.
2. **Detection.** `Detector` runs the YOLO network and converts raw outputs to boxes, class IDs, and confidence scores.
3. **Non-maximum suppression.** `NMS` removes overlapping boxes class by class.
4. **Evaluation.** `metrics.py` computes IoU, matches detections to ground truth, and builds precision recall curves and mean average precision (mAP). `loss.py` computes YOLO style localization, objectness, and classification losses.
5. **Hard negative mining.** `HardNegativeMiner` ranks images by loss so that the hardest examples, such as crowded scenes or frames with many missed objects, can be prioritized for retraining.

The [case study](techtrack/CaseStudy.md) compares two YOLO models on annotated frames and lays out how to measure per class improvements, robustness to noise, flips, and brightness changes, and the effect of the hard negative mining weight.

## Running it

YOLO weights, the annotated logistics images, and the test videos were provided by the course and are not included. Place them under `techtrack/storage/` as described in [techtrack/README.md](techtrack/README.md), then run the service locally or in Docker.

```bash
pip install -r techtrack/requirements.txt
cd techtrack
python app.py
```

## Built with

Python, OpenCV, NumPy, Docker

## Context

Built for EN.705.603 Creating AI-Enabled Systems at Johns Hopkins University (Summer 2025). Course data, model weights, starter code, and unit tests are not included.
