# TM10011 Machine Learning Project – ECG Classification

## Project Overview
This project is part of the course TM10011 – Machine Learning. The objective is to develop a machine learning model capable of automatically detecting abnormalities in 12-lead electrocardiograms (ECGs).

The problem is formulated as a binary classification task:
- 0 = Normal ECG
- 1 = Abnormal ECG

Abnormalities may include:
- 1st degree AV block  
- Right bundle branch block (RBBB)  
- Left bundle branch block (LBBB)  
- Sinus bradycardia  
- Atrial fibrillation  
- Sinus tachycardia  

---

## Dataset
- Samples: 827 patients  
- Features: 9000  
- Source: 12-lead ECG signals  
- Feature extraction: Fourier transform (frequency domain features)

The dataset is imbalanced, with the majority of samples representing normal ECGs.

For the ECG dataset:
- Unzip `ecg.zip` before loading the data.

---

## Objective
The goal is to build a model that:
- Accurately classifies ECGs as normal or abnormal  
- Generalizes well to unseen data  
- Handles class imbalance appropriately  

Although a mean accuracy above 85% is achievable, accuracy alone is insufficient due to class imbalance. Therefore, additional performance metrics are required.

---

## Evaluation Metrics
To properly evaluate model performance, the following metrics are used:
- ROC AUC  
- F1-score (minority class)  
- Balanced accuracy  
- Precision and Recall  
- Average precision (PR AUC)  

---

## Methodology
The implemented pipeline includes:

### Preprocessing
- Scaling (e.g., StandardScaler or RobustScaler)  
- Optional log transformation  
- Handling class imbalance (e.g., SMOTE)  

### Dimensionality Reduction
- Principal Component Analysis (PCA)  
  Used to mitigate the curse of dimensionality (9000 features vs 827 samples)

### Feature Selection (optional)
- SelectKBest  
- L1-based feature selection (Logistic Regression)  

### Models Evaluated
- Logistic Regression  
- Support Vector Machine (SVM)  
- Random Forest  
- k-Nearest Neighbors (kNN)  

### Model Selection
- Hyperparameter tuning using GridSearchCV  
- Nested cross-validation for unbiased performance estimation  

---

## Implementation
The models are implemented in separate scripts:

- `Logistic_regression.py` – Logistic Regression  
- `SVM.py` – Support Vector Machine  
- `RandomForest.py` – Random Forest  
- `kNearestNeighbour.py` – k-Nearest Neighbors
- - `Data_visualization.py` – Data exploration and visualization  
- `load_data.py` – Utility script for loading the ECG dataset  
 

Each script contains the full pipeline, including preprocessing, model training, and evaluation.

### Data visualization
The file `Data_visualization.py` is used for exploratory data analysis and visualization of the ECG dataset. It includes:
- Loading and checking the dataset  
- Inspecting class distribution  
- Visualizing the frequency-domain representation of all 12 ECG leads for an example patient  
- Performing a train/test split  
- Analyzing the cumulative explained variance of PCA  

This script was used to better understand the structure of the data and to support methodological choices such as dimensionality reduction.

### Data loading
The file `ecg/load_data.py` contains a helper function for loading the ECG dataset. It:
- Locates the ECG data file automatically  
- Checks whether the extracted CSV file is present  
- Extracts the ZIP file if needed  
- Loads the dataset into a pandas DataFrame  

This utility ensures that the dataset can be loaded consistently across all scripts.
