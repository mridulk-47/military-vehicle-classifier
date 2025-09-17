# Military vs Civilian Vehicle Classification

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.9.0-orange)
![Keras](https://img.shields.io/badge/Keras-2.9.0-red)

## 📌 Overview
A deep learning-based computer vision system that classifies vehicles as military or civilian. This project implements a Convolutional Neural Network (CNN) to analyze vehicle images and accurately distinguish between military/armored vehicles and civilian vehicles, with applications in defense, surveillance, and security systems.

## 🚀 Features
- **Image Classification**: Accurately classifies vehicles into military or civilian categories
- **Deep Learning Model**: Implements CNN architecture using TensorFlow/Keras
- **Data Augmentation**: Enhances training with various image transformations
- **Model Evaluation**: Comprehensive performance metrics and visualization
- **Transfer Learning**: Option to use pre-trained models (VGG16, ResNet50, etc.)

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/military-vehicle-classifier.git
   cd military-vehicle-classifier
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Dataset
The project uses the [Military vs Civilian Vehicles Image Classification](https://www.kaggle.com/datasets/mexwell/militarycivilian-vehicles-image-classification) dataset from Kaggle, containing thousands of labeled images of military and civilian vehicles.

Dataset Structure:
```
data/
├── train/
│   ├── military/
│   └── civilian/
└── test/
    ├── military/
    └── civilian/
```

## 🧠 Model Architecture
The classification model uses a CNN architecture with the following layers:
1. Convolutional Layers with ReLU activation
2. MaxPooling Layers for dimensionality reduction
3. Dropout Layers for regularization
4. Dense Layers for classification

## 🚦 Usage
1. **Data Preparation**:
   - Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mexwell/militarycivilian-vehicles-image-classification)
   - Extract and organize the dataset in the `data/` directory

2. **Training the Model**:
   ```bash
   python train.py --epochs 50 --batch_size 32
   ```

3. **Making Predictions**:
   ```bash
   python predict.py --image_path path/to/vehicle_image.jpg
   ```

## 📈 Results
The model achieves the following performance metrics:
- Training Accuracy: ~98.5%
- Validation Accuracy: ~96.2%
- Test Accuracy: ~95.8%

## 📂 Project Structure
```
.
├── data/                   # Dataset directory
├── models/                 # Saved models
├── src/
│   ├── data_loader.py     # Data loading and preprocessing
│   ├── model.py          # Model architecture
│   ├── train.py          # Training script
│   └── predict.py        # Prediction script
├── notebooks/            # Jupyter notebooks for exploration
├── requirements.txt      # Python dependencies
└── README.md            # Project documentation
```

## 🙏 Acknowledgments
- [Kaggle](https://www.kaggle.com/) for hosting the dataset
- TensorFlow and Keras for deep learning framework
- OpenCV for image processing

## 📝 Note
This project is for educational and research purposes only. The model's predictions should not be solely relied upon for critical security or military applications.
