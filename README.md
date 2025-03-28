# Eye Detection using YOLO-NAS

## Overview

This project implements **eye detection** using **YOLO-NAS**. It downloads a dataset from **Kaggle**, processes it using **Roboflow**, and performs eye detection on images.

## Features

✅ Download and prepare dataset using Roboflow API\
✅ Use a pre-trained YOLO-NAS model for eye detection\
✅ Predict and save images with detected eyes\
✅ Visualize model performance with training graphs\
✅ Support for further model training and evaluation

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/YogeshGajula/YOLO-NAS-Eye-Detection.git
cd YOLO-NAS-Eye-Detection
```

### Step 2: Install Dependencies

Ensure you have Python installed. Then, install the required packages:

```bash
pip install roboflow matplotlib
```

---

## Dataset Preparation

### 1. Download Dataset

The dataset is sourced from Kaggle:
[https://www.kaggle.com/datasets/ahmadahmadzada/images2000](https://www.kaggle.com/datasets/ahmadahmadzada/images2000)

### 2. Use Roboflow API to Download Dataset

Modify the API key if needed.

```python
from roboflow import Roboflow

rf = Roboflow(api_key="1keefoZmDFOJSw6qft32")
project = rf.workspace("objectdetection-i2zcf").project("object-detection-chiq6")
version = project.version(1)
version.download("yolov9")
```

---

## Eye Detection

### Predict Eyes in an Image

```python
from IPython.display import Image, display

def predict_and_save(model, img_path, save_path):
    display(Image(filename=img_path))
    model.predict(img_path, confidence=40, overlap=30).save(save_path)
    display(Image(filename=save_path))
```

**Usage:**

```python
img_path = "path/to/input.jpg"
save_path = "path/to/output.jpg"
predict_and_save(model, img_path, save_path)
```

---

## Training (Optional)

To train a new YOLO model:

```bash
yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640
```

### Evaluate Model Performance

```bash
yolo task=detect mode=val model=runs/detect/train/weights/best.pt data=data.yaml
```

---

## Visualization

### Loss Graph

```python
import matplotlib.pyplot as plt

epochs = [1, 2, 3, 4, 5]  # Replace with actual epoch values
train_loss = [2.3, 1.8, 1.4, 1.1, 0.9]
val_loss = [2.5, 2.0, 1.6, 1.3, 1.1]

plt.plot(epochs, train_loss, label="Training Loss")
plt.plot(epochs, val_loss, label="Validation Loss", linestyle="dashed")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.show()
```

---

## Folder Structure

```
Eye Detection.v1i.yolov8/
│── data.yaml            # Dataset config file
│── train/               # Training images & labels
│── valid/               # Validation images & labels
│── test/                # Testing images & labels
│── README.dataset.txt   # Dataset description
│── README.roboflow.txt  # Roboflow instructions
```

---

## Contributing

Feel free to fork this repository and submit pull requests.

---

## License

MIT License. See `LICENSE` for more details.

---



