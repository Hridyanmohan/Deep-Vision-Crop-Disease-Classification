# Deep-Vision-Crop-Disease-Classification
A Deep Learning Computer Vision project classifying apple foliar diseases using Custom CNNs and Transfer Learning (ResNet50). Mitigated severe class imbalance using targeted data augmentation and class weights, achieving 85% test accuracy on the Plant Pathology FGVC7 dataset.
# 🍃 Apple Foliar Disease Classification using Deep Learning

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red)
![Machine Learning](https://img.shields.io/badge/Computer%20Vision-Image%20Classification-success)

## 📌 Project Overview
This project applies **Deep Learning and Computer Vision** to automate the detection of apple foliar diseases. Utilizing the Plant Pathology 2020 FGVC7 dataset, the objective is to classify high-resolution orchard images into four categories: `Healthy`, `Rust`, `Scab`, and `Multiple Diseases`. 

The project demonstrates a comprehensive machine learning pipeline, starting from a custom baseline CNN and culminating in a fine-tuned, pre-trained **ResNet50** architecture. A major focus of this project was engineering solutions for **severe dataset class imbalance** and optimizing model capacity for high-variance image data.

## 🛠️ Tech Stack
* **Core Technologies:** Python, TensorFlow, Keras
* **Deep Learning Techniques:** Convolutional Neural Networks (CNN), Transfer Learning (ResNet50), Artificial Neural Networks (ANN), Image Augmentation, Fine-Tuning.
* **Data Engineering:** Stratified Train/Test Splitting, Class Weight Balancing, Image Preprocessing (Z-Score Standardization, ImageNet Mean Subtraction).
* **Evaluation Metrics:** Accuracy, Weighted Precision/Recall, F1-Score, Confusion Matrices.
* **Data Visualization:** Matplotlib, Seaborn, Pandas, NumPy.

## 📊 Dataset & Preprocessing
* **Source:** Plant Pathology 2020 - FGVC7
* **Distribution:** 1,821 expert-annotated images.
* **Challenges Mitigated:** * The dataset is highly imbalanced (e.g., 622 'Rust' samples vs. only 91 'Multiple Diseases' samples).
  * **Solution:** Implemented **stratified splitting** (70/15/15) to preserve minority class distribution across splits and computed **balanced class weights** to penalize the loss function for minority misclassifications.
  * Resized all images to `224x224` to match ResNet50 input tensor requirements.

## 🧠 Model Architecture & Evolution

### Model 1: Custom Baseline CNN
* **Structure:** 3 Convolutional blocks (Conv2D -> BatchNorm -> MaxPooling) + Dense ANN head.
* **Result:** `44.89% Accuracy`. The model severely overfit to the majority classes, proving that shallow architectures lack the parameter capacity to model complex foliar textures from raw data.

### Model 2: Augmented Custom CNN
* **Structure:** Same baseline architecture, but introduced dynamic `ImageDataGenerator` augmentations (rotations, shifts, zooms, flips).
* **Result:** `26.28% Accuracy`. While augmentation forced the model to acknowledge the minority classes (improving recall for 'Multiple Diseases'), the shallow network suffered from underfitting and catastrophic forgetting, highlighting the necessity for a deeper network.

### Model 3: Transfer Learning (ResNet50) - *Optimal Model*
* **Structure:** Pre-trained ResNet50 (ImageNet weights) with a custom Global Average Pooling and Dense classification head (Dropout = 0.5).
* **Strategy:** Two-phase training. Phase 1 froze the base to train the custom head. Phase 2 un-froze the top 20 layers for fine-tuning at a highly reduced learning rate (`1e-5`).
* **Result:** `85.04% Accuracy`. Deep feature hierarchies successfully mapped the augmented textures, drastically improving F1-scores across all classes.

## 📈 Final Results & Comparative Analysis

| Model | Test Accuracy | Weighted Precision | Weighted Recall | Weighted F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| Model 1 (Baseline) | 44.89% | 0.44 | 0.45 | 0.40 |
| Model 2 (Augmented) | 26.28% | 0.22 | 0.26 | 0.21 |
| **Model 3 (ResNet50)** | **85.04%** | **0.86** | **0.85** | **0.85** |

*A detailed misclassification analysis revealed that ResNet50's primary errors occur biologically logical edge cases, such as confusing physically damaged leaves with pathological scab lesions, or struggling when one disease visually dominates a 'Multiple Diseases' sample.*

## 🚀 Future Scope
1. **Advanced Synthetic Data Generation:** Utilizing Generative Adversarial Networks (GANs) or diffusion models to synthetically upscale the `Multiple Diseases` minority class.
2. **Object Detection / Segmentation:** Transitioning from image classification to bounding-box localization using architectures like **YOLOv8** or **Mask R-CNN** to highlight exact lesion locations.
3. **Edge Deployment:** Compressing the fine-tuned ResNet50 model using **TensorFlow Lite (TFLite)** for offline, real-time inference on mobile devices in agricultural environments.

## ⚙️ How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/YourUsername/Apple-Foliar-Disease-Classification.git](https://github.com/YourUsername/Apple-Foliar-Disease-Classification.git)
2. Install required dependencies:
   ```bash
   pip install tensorflow pandas numpy matplotlib seaborn scikit-learn
3. Open and run the Jupyter Notebook:
   ```bash
   jupyter notebook Plant_Disease_Detection.ipynb
