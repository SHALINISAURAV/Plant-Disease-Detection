# 🥔 Potato Leaf Disease Detection using Deep Learning

A complete deep learning pipeline for detecting potato leaf diseases
using AlexNet, Custom CNN, VGG16, and ResNet50.

This system classifies potato leaves into: - 🍂 Early Blight - 🍁 Late
Blight - 🌿 Healthy

------------------------------------------------------------------------

## 📌 Project Overview

This project implements an end-to-end deep learning system for plant
disease detection using convolutional neural networks.

The system: - Loads and preprocesses image datasets - Implements a
custom AlexNet architecture - Compares multiple deep learning
architectures - Applies data augmentation for better generalization -
Visualizes training performance and confusion matrices - Performs
inference on new images

The goal is to assist in early detection of potato leaf diseases and
reduce agricultural losses.

------------------------------------------------------------------------

## 🧠 Models Implemented

### 🔹 AlexNet (Custom Implementation)

-   5 Convolutional Layers
-   Batch Normalization
-   MaxPooling Layers
-   2 Fully Connected Layers (4096 neurons)
-   Dropout Regularization
-   Softmax Output Layer (3 Classes)

### 🔹 Custom CNN

-   Lightweight architecture
-   3 Convolutional layers
-   MaxPooling
-   Dense + Dropout layers

### 🔹 Transfer Learning Models

#### ✅ VGG16

-   Pretrained on ImageNet
-   Frozen base layers
-   Custom classification head

#### ✅ ResNet50

-   Residual connections
-   Global Average Pooling
-   Dense + Dropout classifier

------------------------------------------------------------------------

## 📂 Dataset Structure

    dataset/
    │
    ├── Potato___Early_blight/
    ├── Potato___Late_blight/
    └── Potato___healthy/

------------------------------------------------------------------------

## ⚙️ Tech Stack

-   Python
-   TensorFlow / Keras
-   OpenCV
-   NumPy
-   Pandas
-   Matplotlib
-   Seaborn
-   Scikit-Learn

------------------------------------------------------------------------

## 🚀 Installation

``` bash
git clone https://github.com/yourusername/potato-leaf-disease-detection.git
cd potato-leaf-disease-detection
pip install -r requirements.txt
```

------------------------------------------------------------------------

## ▶️ How to Run

Update dataset path inside script:

``` python
DATASET_PATH = "your_dataset_path"
```

Run training:

``` bash
python PlantDisease.py
```

------------------------------------------------------------------------

## 📊 Results & Outputs

Below sections are reserved for adding your project outputs.

### 🔹 Training Accuracy & Loss Curves

(Add training history plots here)

Example:

![Training History](images/training_history.png)

------------------------------------------------------------------------

### 🔹 Confusion Matrix

(Add confusion matrix image here)

Example:

![Confusion Matrix](images/confusion_matrix.png)

------------------------------------------------------------------------

### 🔹 Model Comparison

(Add model comparison chart here)

Example:

![Model Comparison](images/model_comparison.png)

------------------------------------------------------------------------

### 🔹 Sample Predictions

(Add sample prediction outputs here)

Example:

![Sample Predictions](<img width="4175" height="2970" alt="sample_predictions" src="https://github.com/user-attachments/assets/b54d93a2-1088-4986-98aa-c4be2babb906" />
)

------------------------------------------------------------------------

## 🧪 Evaluation Metrics

-   Accuracy
-   Loss
-   Precision
-   Recall
-   F1-Score
-   Confusion Matrix

The best performing model is automatically selected based on test
accuracy.

------------------------------------------------------------------------

## 🌍 Real-World Impact

-   Helps farmers detect disease early
-   Reduces crop loss
-   Supports precision agriculture
-   Extendable to mobile applications

------------------------------------------------------------------------

## 🔮 Future Improvements

-   Web app deployment (Flask / FastAPI)
-   TensorFlow Lite mobile version
-   Grad-CAM visualization
-   Multi-crop disease detection
-   Hyperparameter tuning

------------------------------------------------------------------------

## 👩‍💻 Author

SHALINI SAURAV
Deep Learning & AI Enthusiast

------------------------------------------------------------------------

⭐ If you like this project, consider giving it a star on GitHub!

