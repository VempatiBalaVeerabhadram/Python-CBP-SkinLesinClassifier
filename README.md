# SkinLesionClassifier


A Streamlit-based web application to detect and classify skin lesions as **Benign** or **Malignant** using PCA and SVM models.

## Features
- Upload an image of a skin lesion to predict its classification.
- Provides visual recommendations and actionable insights based on predictions.
- Explore data analytics through an interactive dashboard displaying dataset distribution, model accuracy, and confusion matrix.

---

## Getting Started

### Prerequisites
- Python 3.7 or higher
- Required Python libraries:
  - `numpy`
  - `matplotlib`
  - `Pillow`
  - `joblib`
  - `streamlit`
  - `pandas`
  - `seaborn`
### Usage
Start the Streamlit App:
```bash
streamlit run app.py
```
- Navigate to download the Datasets: Open your browser and go to https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign.

## Choose an Option:

- 🔍 Image Prediction: Upload a skin lesion image to classify it as benign or malignant.
- 📊 Data Insights Dashboard: Analyze dataset distribution and visualize model performance.

## Model Description:
- PCA: Principal Component Analysis is used to reduce the dimensionality of images for efficient processing.
- SVM: Support Vector Machine is utilized for classification of skin lesions.

## How It Works
Image Prediction:
The uploaded image is preprocessed and normalized.
Dimensionality reduction is applied using PCA.
The SVM model predicts the image as either Benign or Malignant.
Displays results with additional recommendations.
Data Insights Dashboard:

Visualizes the training and testing dataset distributions.
Displays model accuracy and confusion matrix.
## Install the dependencies using the following command:

```bash
pip install -r requirements.txt
```
### Clone the Repository
```bash
git clone https://github.com/your-username/Benign_vs_Malignant_Predictor.git
cd Benign_vs_Malignant_Predictor

