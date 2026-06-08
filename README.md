# 🚀 YOLOv8 Object Detection — Custom Dataset Training

> Train, evaluate, and deploy YOLOv8 object detection models on custom datasets, with optional auto-labeling via Autodistill.

[![Ultralytics YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-blue?logo=github)](https://github.com/ultralytics/ultralytics)
[![Roboflow](https://img.shields.io/badge/Dataset-Roboflow-purple)](https://roboflow.com)
[![Autodistill](https://img.shields.io/badge/AutoLabel-Autodistill-green)](https://github.com/autodistill/autodistill)
[![Python](https://img.shields.io/badge/Python-3.8%2B-yellow?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Notebooks](#notebooks)
- [Getting Started](#getting-started)
- [Method 1: Manual Training with Custom Dataset](#method-1-manual-training-with-custom-dataset)
- [Method 2: Auto-Training with Autodistill](#method-2-auto-training-with-autodistill)
- [Results](#results)
- [Deployment](#deployment)
- [Resources](#resources)

---

## Overview

This repository provides two end-to-end workflows for training YOLOv8 object detection models:

**Method 1 — Manual Training:** Download a labeled dataset from [Roboflow Universe](https://universe.roboflow.com/), fine-tune a YOLOv8 model, validate it, and deploy it to Roboflow or run it locally via Roboflow Inference.

**Method 2 — Auto-Training with Autodistill:** Start from a folder of raw, unlabeled images (or videos). Use foundation models (GroundedSAM) to automatically annotate your data, then distill a small, fast YOLOv8 model — all with no human labeling required.

Both workflows run seamlessly on Google Colab with GPU acceleration.

---

## Repository Structure

```
├── train_yolov8_object_detection_on_custom_dataset.ipynb   # Manual training notebook
├── train_yolov8_object_detection_on_custom_dataset.py      # Manual training script
├── how_to_auto_train_yolov8_model_with_autodistill.py      # Autodistill auto-training script
└── README.md
```

---

## Notebooks

| Notebook | Description | Open in Colab |
|---|---|---|
| `train_yolov8_object_detection_on_custom_dataset.ipynb` | Train YOLOv8 on a labeled Roboflow dataset | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-yolov8-object-detection-on-custom-dataset.ipynb) |
| Autodistill Auto-Training | Auto-label images and train YOLOv8 end-to-end | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-auto-train-yolov8-model-with-autodistill.ipynb) |

---

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- A free [Roboflow account](https://app.roboflow.com) (for dataset management and deployment)

### Installation

**For manual YOLOv8 training:**
```bash
pip install ultralytics==8.2.103 roboflow==1.1.48
```

**For Autodistill auto-training:**
```bash
pip install autodistill autodistill-grounded-sam autodistill-yolov8 roboflow supervision==0.24.0
```

---

## Method 1: Manual Training with Custom Dataset

This workflow trains a YOLOv8 model on a dataset you've already labeled (or one from Roboflow Universe).

### Step 1 — Download your dataset from Roboflow

```python
import roboflow

roboflow.login()
rf = roboflow.Roboflow()

project = rf.workspace("your-workspace").project("your-project")
dataset = project.version(1).download("yolov8")
```

### Step 2 — Train

```bash
yolo task=detect mode=train model=yolov8s.pt data={dataset.location}/data.yaml epochs=25 imgsz=800 plots=True
```

### Step 3 — Validate

```bash
yolo task=detect mode=val model=runs/detect/train/weights/best.pt data={dataset.location}/data.yaml
```

### Step 4 — Run Inference

```bash
yolo task=detect mode=predict model=runs/detect/train/weights/best.pt conf=0.25 source={dataset.location}/test/images save=True
```

---

## Method 2: Auto-Training with Autodistill

This workflow takes you from **raw images or videos** to a trained, deployable model with no manual annotation.

```
Raw Images/Videos  →  Auto-Label (GroundedSAM)  →  Train YOLOv8  →  Deploy
```

### Step 1 — Prepare your images

Place images in an `images/` folder, or extract frames from videos:

```python
import supervision as sv

VIDEO_DIR_PATH = "videos/"
IMAGE_DIR_PATH = "images/"
FRAME_STRIDE = 10  # save every 10th frame

for video_path in sv.list_files_with_extensions(VIDEO_DIR_PATH, ["mov", "mp4"]):
    with sv.ImageSink(target_dir_path=IMAGE_DIR_PATH) as sink:
        for image in sv.get_video_frames_generator(str(video_path), stride=FRAME_STRIDE):
            sink.save_image(image=image)
```

### Step 2 — Define your ontology and auto-label

```python
from autodistill.detection import CaptionOntology
from autodistill_grounded_sam import GroundedSAM

ontology = CaptionOntology({
    "milk bottle": "bottle",
    "blue cap": "cap"
})

base_model = GroundedSAM(ontology=ontology)
dataset = base_model.label(
    input_folder="images/",
    extension=".png",
    output_folder="dataset/"
)
```

### Step 3 — Train YOLOv8 on the auto-labeled dataset

```python
from autodistill_yolov8 import YOLOv8

target_model = YOLOv8("yolov8n.pt")
target_model.train("dataset/data.yaml", epochs=50)
```

### Step 4 — Run video inference

```bash
yolo predict model=runs/detect/train/weights/best.pt source=your_video.mp4
```

---

## Results

After training, evaluation artifacts are saved to `runs/detect/train/`:

| File | Description |
|---|---|
| `confusion_matrix.png` | Per-class prediction breakdown |
| `results.png` | Loss and mAP curves across epochs |
| `val_batch0_pred.jpg` | Sample validation predictions |
| `weights/best.pt` | Best model checkpoint for deployment |

---

## Deployment

### Deploy to Roboflow (hosted API)

```python
project.version(dataset.version).deploy(
    model_type="yolov8",
    model_path="runs/detect/train/"
)
```

### Self-host with Roboflow Inference (Docker)

```bash
docker pull roboflow/roboflow-inference-server-gpu

# Run inference via HTTP
import requests

res = requests.post(
    "http://localhost:9001/{workspace_id}/{model_id}",
    json={
        "image": {"type": "url", "value": "your_image_url"},
        "confidence": 0.75,
        "iou_threshold": 0.5,
        "api_key": "YOUR_API_KEY"
    }
)
predictions = res.json()
```

Roboflow Inference supports CPU and GPU targets, including the NVIDIA Jetson and TRT-compatible devices.

---

## Resources

- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [Autodistill GitHub](https://github.com/autodistill/autodistill)
- [Roboflow Universe](https://universe.roboflow.com/) — 200,000+ open datasets
- [Roboflow Notebooks](https://github.com/roboflow/notebooks) — 20+ computer vision tutorials
- [Roboflow YouTube](https://www.youtube.com/c/Roboflow)
- [Roboflow Discuss](https://discuss.roboflow.com/)

---

## 🏆 Acknowledgements

Notebooks originally authored by [Roboflow](https://roboflow.com). This repository packages and extends their work for streamlined use.
