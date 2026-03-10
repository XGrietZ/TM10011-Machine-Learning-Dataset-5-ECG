#%%
#-----------------------------------
# 0. Imports
#-----------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tqdm import tqdm


from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, learning_curve, cross_val_score, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, FunctionTransformer
from sklearn.metrics import (
    auc, classification_report, confusion_matrix, roc_auc_score, f1_score, balanced_accuracy_score, roc_curve, precision_recall_curve, average_precision_score
)
from sklearn import model_selection, neighbors, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.base import clone

#%%
#-----------------------------------
# 1. Load data
#-----------------------------------

from ecg.load_data import load_data

data = load_data()
print(f'The number of samples: {len(data.index)}')
print(f'The number of columns: {len(data.columns)}')

df = pd.read_csv('ecg\ecg_data\ecg_data.csv',index_col=0)

# Drop index column if present
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

if "label" not in df.columns:
    raise ValueError("Expected a column named 'label' in the CSV.")

X = df.drop(columns=["label"])
y = df["label"].astype(int)

print("\nDataset shape:", X.shape)
print("Label distribution:\n", y.value_counts())

#%%
#-----------------------------------
# 2. Train/Test split (hold-out)
#-----------------------------------   
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    shuffle=True,
    stratify=y,
    random_state=42
)

print("\nTrain label distribution:\n", y_train.value_counts())
print("Test  label distribution:\n", y_test.value_counts())


#%%
#-----------------------------------
# 3. Pipeline
#-----------------------------------
# preprocessing steps:
# - StandardScaler: scales features to have mean=0 and std=1
# - RobustScaler: scales features using statistics that are robust to outliers (median and IQR) 
# - FunctionTransformer: applies a custom function to the data (e.g., log transformation)

preprocessing = Pipeline([
    ('log_transform', 'passthrough'),
    ('scaler', StandardScaler())
])

# Feauture selection with principal component analysis (PCA)
feature_selection = Pipeline([
    ('pca', PCA(n_components=0.95))
])

# # Define the number of features selected by PCA
# pca = PCA(n_components=0.95)
# pca.fit(preprocessing.fit_transform(X_train))
# n_features_selected = pca.n_components_
# print(f"\nNumber of features selected by PCA: {n_features_selected}")

# Define the model
model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')

# Combine preprocessing and model into a single pipeline
pipeline = Pipeline([
    ('preprocessing', preprocessing),
    ('feature_selection', feature_selection),
    ('classifier', model)
    
])

# %%
#-----------------------------------
# 4. Train the model
#-----------------------------------
pipeline.fit(X_train, y_train)

#%%
# -----------------------------------
# 5. Evaluate on test set
# -----------------------------------
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_proba))
print("F1 Score:", f1_score(y_test, y_pred))
print("Balanced Accuracy Score:", balanced_accuracy_score(y_test, y_pred))

#%% 
# -----------------------------------
# 6. Cross-validation on training set
# -----------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)

print("\nCV ROC AUC scores:", cv_scores)
print("Mean CV ROC AUC:", cv_scores.mean())
print("Std CV ROC AUC:", cv_scores.std())

#%%
# -----------------------------------   
# 7. Additional evaluation plots
# -----------------------------------

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

precision, recall, _ = precision_recall_curve(y_test, y_proba)
ap = average_precision_score(y_test, y_proba)
plt.figure(figsize=(6, 4))
plt.plot(recall, precision, lw=2, label=f'PR curve (AP = {ap:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.show()

train_sizes, train_scores, test_scores = learning_curve(
    pipeline, X_train, y_train,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 5)
)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Training score')
plt.plot(train_sizes, test_scores.mean(axis=1), 'o-', label='Cross-validation score')
plt.xlabel('Training examples')
plt.ylabel('ROC AUC')
plt.title('Learning Curve')
plt.legend()
plt.grid()
plt.show()

# %%
#-----------------------------------
# 8. Hyperparameter tuning with graphical progress bar
#-----------------------------------

# param_grid = {
#     'preprocessing__scaler': [
#         StandardScaler(),
#         RobustScaler(),
#         MinMaxScaler(),
#         'passthrough'
#     ],
#     'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
#     'classifier__penalty': ['l1', 'l2'],
#     'classifier__solver': ['liblinear']
#     'feature_selection__pca__n_components': [
#     20, 50, 100, 200, 0.95
#     ],
#     'feature_selection__pca__svd_solver': [
#     'auto',
#     'randomized'
#     ]
# }
param_grid = {
    'preprocessing__scaler': [
        StandardScaler(),
        RobustScaler(),
        MinMaxScaler(),
    ],

    'preprocessing__log_transform': [
        'passthrough',
        FunctionTransformer(np.log1p)
    ],

    'feature_selection__pca__n_components': [
        80, 160, 0.93, 0.95
    ],

    'classifier__C': [
        0.001, 0.01, 0.1, 1, 10
    ],

    'classifier__penalty': [
        'l1', 'l2'
    ],

    'classifier__solver': [
        'liblinear', 'saga'
    ]
}

param_list = list(ParameterGrid(param_grid))
print(f"Total models: {len(param_list)}")
print(f"Total fits (with {cv.get_n_splits()}-fold CV): {len(param_list) * cv.get_n_splits()}")

results = []

start_time = time.time()

for params in tqdm(param_list, desc="Grid search progress", unit="model"):
    model_candidate = clone(pipeline)
    model_candidate.set_params(**params)

    scores = cross_val_score(
        model_candidate,
        X_train,
        y_train,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1
    )

    results.append({
        "params": params,
        "mean_score": scores.mean(),
        "std_score": scores.std()
    })

end_time = time.time()
print(f"\nGrid search took {end_time - start_time:.2f} seconds")

# Best parameter set
best_result = max(results, key=lambda x: x["mean_score"])
print("\nBest params:", best_result["params"])
print("Best CV ROC AUC:", best_result["mean_score"])
print("Best CV ROC AUC std:", best_result["std_score"])

# Optional: results overview as DataFrame
results_df = pd.DataFrame(results).sort_values("mean_score", ascending=False)
print("\nTop 5 models:")
print(results_df.head())
# %%
#-----------------------------------
# 9. Fit best model on full training set and evaluate on test set
#-----------------------------------
best_model = clone(pipeline)
best_model.set_params(**best_result["params"])
best_model.fit(X_train, y_train)

y_pred_best = best_model.predict(X_test)
y_proba_best = best_model.predict_proba(X_test)[:, 1]

print("\nClassification Report (Best Model):\n", classification_report(y_test, y_pred_best))
print("Confusion Matrix (Best Model):\n", confusion_matrix(y_test, y_pred_best))
print("ROC AUC Score (Best Model):", roc_auc_score(y_test, y_proba_best))
print("F1 Score (Best Model):", f1_score(y_test, y_pred_best))
print("Balanced Accuracy Score (Best Model):", balanced_accuracy_score(y_test, y_pred_best))
#%%
#-----------------------------------
# 10. Additional plots for best model
#-----------------------------------
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Best Model)')
plt.show()

fpr, tpr, _ = roc_curve(y_test, y_proba_best)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Best Model)')
plt.legend(loc='lower right')
plt.show()

precision, recall, _ = precision_recall_curve(y_test, y_proba_best)
ap = average_precision_score(y_test, y_proba_best)
plt.figure(figsize=(6, 4))
plt.plot(recall, precision, lw=2, label=f'PR curve (AP = {ap:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Best Model)')
plt.legend(loc='lower left')
plt.show()

# %%
