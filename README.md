# 🩺 Pneumonia Detection using Deep Learning (ResNet50)

This project detects **Pneumonia from Chest X-Ray images** using a **Convolutional Neural Network (CNN)** with **ResNet50** as the backbone.  
It leverages **transfer learning** for accurate classification between **Normal** and **Pneumonia** cases.

---

## 📌 Features
- Downloads and processes the **Chest X-Ray Pneumonia dataset** automatically from Kaggle  
- Implements **ResNet50** for feature extraction and classification  
- Supports **Google Colab integration** (dataset & Kaggle API handling included)  
- Training, validation, and testing workflow for reproducibility  
- Visualization of results: sample X-rays, accuracy, and loss curves  

---

## 🗂 Dataset
We use the **Chest X-Ray Images (Pneumonia)** dataset by [Paul Mooney](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).  
- **Classes:** `NORMAL` and `PNEUMONIA`  
- The dataset is automatically downloaded with:
```python
import kagglehub
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
