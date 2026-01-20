# Rice_Disease_Classification_Paddy_Disease 🌾

<p align="center">
  <img src="https://img.shields.io/badge/Deep%20Learning-CNN-blueviolet?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Framework-TensorFlow-orange?style=for-the-badge&logo=tensorflow" />
  <img src="https://img.shields.io/badge/Language-Python-blue?style=for-the-badge&logo=python" />
</p>

---

## 📌 Project Overview

**Rice_Disease_Classification_Paddy_Disease** is a deep learning project that uses a **Convolutional Neural Network (CNN)** built with **TensorFlow/Keras** to classify rice plant diseases from images.

The goal of this project is to help identify rice crop diseases automatically, which can assist farmers and researchers in early detection and better crop management.

---

## 🎯 Objectives

* Classify rice leaf images into **10 disease categories**
* Explore and visualize the rice disease dataset
* Build and train a CNN model using TensorFlow
* Evaluate model performance using accuracy and loss metrics

---

## 🧩 Dataset Description

* 📸 **Total Images:** 10,407
* 🦠 **Disease Classes:** 10
* 🌱 **Rice Varieties:** 10
* 📅 **Plant Age Range:** 45 – 82 days

### 🦠 Disease Classes

* bacterial_leaf_blight
* bacterial_leaf_streak
* bacterial_panicle_blight
* blast
* brown_spot
* dead_heart
* downy_mildew
* hispa
* normal
* tungro

### 🌾 Rice Varieties

ADT45, IR20, KarnatakaPonni, Onthanel, Ponni, Surya, Zonal, AndraPonni, AtchayaPonni, RR

---

## 📁 Project Structure

```
Rice_Disease_Classification_Paddy_Disease/
│-- dataSet/
│   │-- train.csv
│   │-- train_images/
│       │-- bacterial_leaf_blight/
│       │-- bacterial_leaf_streak/
│       │-- ...
│-- notebooks/
│   │-- rice_disease_classification.ipynb
│-- models/
│-- screenshots/
│-- README.md
```

---

## 📊 Exploratory Data Analysis (EDA)

* Distribution of rice varieties
* Distribution of disease classes
* Sample image visualization for normal and diseased leaves
* Statistical summary of rice plant age

---

## 🧠 Model Architecture

The CNN model consists of:

* Image Rescaling (1/255)
* 3 Convolutional layers with ReLU activation
* MaxPooling layers
* Dropout layer to prevent overfitting
* Fully connected Dense layers
* Softmax output layer for multi-class classification

```text
Input (224×224×3)
→ Conv2D → MaxPooling
→ Conv2D → MaxPooling
→ Conv2D → MaxPooling
→ Flatten
→ Dropout (0.25)
→ Dense (128, ReLU)
→ Dense (10, Softmax)
```

---

## ⚙️ Training Configuration

* **Image Size:** 224 × 224
* **Batch Size:** 16
* **Optimizer:** Adam
* **Loss Function:** Sparse Categorical Crossentropy
* **Epochs:** 10
* **Callback:** Early Stopping (patience = 5)
* **Train / Validation Split:** 80% / 20%

---

## 📈 Model Performance

* **Training Accuracy:** ~99%
* **Validation Accuracy:** ~79.8%

Performance is visualized using:

* Training vs Validation Loss graph
* Training vs Validation Accuracy graph

---

## 🧪 Observations

* Total training images: **10,407**
* Number of disease classes: **10**
* Rice plant age ranges from **45 to 82 days**
* **ADT45** is the most common rice variety in the dataset

---

## ▶️ How to Run the Project

### 1️⃣ Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

### 2️⃣ Run the Notebook / Script

* Open the Jupyter Notebook or Python script
* Ensure dataset paths are correctly set
* Train the model using TensorFlow

---

## 🛠 Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

---

## 🚀 Future Improvements

* Apply data augmentation
* Use transfer learning (ResNet, MobileNet, EfficientNet)
* Improve validation accuracy
* Deploy model as a web or mobile application

---

## 📄 License

This project is developed for **academic and research purposes only**.

---

✨ **Healthy Crops, Smarter Farming 🌱**
